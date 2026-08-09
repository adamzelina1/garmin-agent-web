"""Browser chat UI (Gradio) over the read-only Garmin agent.

A thin front-end for ``garmin-ask``: the same agent, tools, model config
(cloud / Ollama), long-term memory profile and read-only safety are reused
unchanged; Gradio renders the conversation.

Automatic session persistence: every web chat is written to a session file
after each turn and the last session is loaded back into the UI on startup, so
picking up where you left off is the default. The file can be overridden with
``--session FILE`` (a later console ``garmin-ask --session FILE`` can also
resume it).

Charts: the agent can call the ``chart`` tool, which runs a read-only query
and returns Plotly chart JSON embedded in ``<chart> ... </chart>`` tags. This
module parses the markers and renders each one as a ``gr.Plot`` beneath the
answer text (chart JSON is stripped from the persisted session).
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from .ask import (
    ReadOnlyDB,
    _build_agent,
    _build_chart_figure,
    _load_session,
    _record_turn,
    _save_session,
)
from .config import load_config


def _history_to_messages(
    history: list[dict[str, Any]],
) -> list[Any]:
    """Map Gradio's openai-style history to Pydantic AI messages.

    Only the text of user/assistant turns is kept — tool call/result details
    and chart blocks are dropped, which is all the model needs to hold context.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    msgs: list[Any] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if isinstance(content, list):
            text = " ".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
            content = text.strip()
        if not isinstance(content, str):
            content = str(content)
        if role == "user":
            msgs.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role in ("assistant", "bot"):
            msgs.append(ModelResponse(parts=[TextPart(content=content)]))
    return msgs


def _messages_to_history(messages: list[Any]) -> list[dict[str, str]]:
    """Map persisted Pydantic AI messages back to Gradio's history format."""
    from pydantic_ai.messages import ModelResponse, ModelRequest, TextPart, UserPromptPart

    history: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    history.append(
                        {
                            "role": "user",
                            "content": content if isinstance(content, str) else str(content),
                        }
                    )
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    history.append({"role": "assistant", "content": part.content})
    return history


def _parse_block(raw: str) -> dict[str, Any] | None:
    """Parse a fenced/plain JSON figure block, or None if malformed.

    Tolerates trailing filler after the JSON object (the chart tool appends a
    ``(query returned N rows)`` note, which a model may echo inside the tags):
    first a strict parse, then a raw JSON-prefix decode.
    """
    raw = raw.strip()
    if raw.startswith(("```", "~~~")):
        raw = raw.strip("`~")
    if raw.endswith(("```", "~~~")):
        raw = raw.strip("`~")
    if not raw:
        return None
    if raw.lower().startswith("json"):
        raw = raw[4:].lstrip("\r\n ")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[raw.find("{"):])
        except (json.JSONDecodeError, ValueError):
            return None
        data = obj
    return data if isinstance(data, dict) else None


def _recover_chart_specs(messages: list[Any]) -> list[dict[str, Any]]:
    """Recover validated chart specs from a run's successful ``chart`` tool calls.

    The model is told to embed each validated spec in ``<chart> ... </chart>``
    but occasionally only *says* a chart exists and never emits the spec — the
    tool call itself still holds the spec. This walks the turn's messages and
    returns the spec of every successful ``chart`` return so a dropped chart can
    still be drawn. Duplicate specs will be filtered by the caller.
    """
    rebuilt: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if getattr(part, "tool_name", None) != "chart":
                continue
            if getattr(part, "outcome", None) != "success":
                continue
            content = str(getattr(part, "content", "") or "")
            if not content.startswith("OK: "):
                continue
            data = _parse_block(content[len("OK: ") :])
            if isinstance(data, dict) and data.get("sql"):
                rebuilt.append(data)
    return rebuilt


def _extract_charts(answer: str) -> tuple[str, list[dict[str, Any]]]:
    """Split an agent answer into (plain text, list of plotly figure dicts).

    ``<chart>`` blocks are consumed and their JSON parsed (tolerating a
    surrounding fenced code block). The closing ``</chart>`` tag is *optional*:
    an unclosed block runs to the end of the answer and its JSON prefix is read
    with a raw JSON decoder, so a model that forgets the close tag (or appends
    trailing filler after the JSON) still produces a chart. Malformed blocks
    are dropped along with their marker text.
    """
    charts: list[dict[str, Any]] = []

    def _figure_prefix(raw: str) -> tuple[dict[str, Any] | None, int]:
        """Return the leading dict and the index just past it, else (None, 0)."""
        dec = json.JSONDecoder()
        start = raw.find("{")
        if start == -1:
            return None, 0
        try:
            obj, end = dec.raw_decode(raw[start:])
        except json.JSONDecodeError:
            return None, 0
        if isinstance(obj, dict):
            return obj, start + end
        return None, 0

    pieces: list[str] = []
    low = answer.lower()
    pos = 0
    while True:
        start = low.find("<chart>", pos)
        if start == -1:
            pieces.append(answer[pos:])
            break
        pieces.append(answer[pos:start])
        close = low.find("</chart>", start)
        if close != -1:  # well-formed block: JSON ends at the close tag
            inner = answer[start + len("<chart>") : close]
            data = _parse_block(inner)
            if data is not None:
                charts.append(data)
            end = close + len("</chart>")
        else:  # unclosed block: parse JSON prefix, keep the text tail
            data, end = _figure_prefix(answer[start + len("<chart>") :])
            if data is not None:
                charts.append(data)
                end += start + len("<chart>")
                tail = answer[end:]
                if tail.lstrip().startswith(("```", "~~~")):
                    end = len(answer)  # a leftover closing fence: drop it
            else:
                end = start + len("<chart>")  # malformed: drop the bare tag
        pos = end
    return "".join(pieces), charts


class ChatSession:
    """One chat app instance: lazy agent, auto-persistent conversation."""

    def __init__(
        self,
        cfg: dict[str, Any],
        session_path: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.session_path = session_path or cfg.get("web_session_file")
        self._holder: dict[str, Any] = {}
        self.initial_history: list[dict[str, str]] = []

        if self.session_path:
            resumed = _load_session(self.session_path)
            if resumed:
                self.initial_history = _messages_to_history(resumed)
                print(
                    f"resumed {len(resumed)} prior message(s) from {self.session_path}"
                )

    def _agent(self) -> Any:
        if "agent" not in self._holder:
            db = ReadOnlyDB(self.cfg["db_path"])
            self._holder["db"] = db
            self._holder["agent"] = _build_agent(self.cfg, db)
        return self._holder["agent"]

    def respond(
        self, message: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        history = list(history) or list(self.initial_history)
        result = self._agent().run_sync(
            message, message_history=_history_to_messages(history)
        )
        _record_turn(self.cfg, message, result)
        text, specs = _extract_charts(str(result.output))

        # Fallback: the model may validate a chart via the tool but forget to
        # embed the spec in <chart> tags. Recover any validated specs that were
        # not embedded so the chart still renders.
        embedded_sql = {spec.get("sql") for spec in specs}
        for spec in _recover_chart_specs(result.new_messages()):
            if spec["sql"] not in embedded_sql:
                specs.append(spec)
                embedded_sql.add(spec["sql"])

        if self.session_path:
            full = _history_to_messages(
                history
                + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": text.strip()},
                ]
            )
            _save_session(self.session_path, full)

        db = self._holder.get("db")
        content: list[Any] = []
        if text.strip():
            content.append({"type": "text", "text": text.strip()})
        for spec in specs:
            rendered = _render_chart(spec, db) if db is not None else None
            if rendered is not None:
                content.append(rendered)
        if not content:
            content.append({"type": "text", "text": ""})
        return {"role": "assistant", "content": content}


def make_chat_session(
    cfg: dict[str, Any],
    session_path: str | None = None,
) -> ChatSession:
    """Build a :class:`ChatSession` for ``gr.ChatInterface``.

    ``session_path`` defaults to the configured ``web_session_file``, so
    sessions persist and resume automatically. Pass an explicit path to
    override it.
    """
    return ChatSession(cfg, session_path)


def _render_chart(spec: dict[str, Any], db: ReadOnlyDB) -> Any | None:
    """Turn a chart spec into a ``gr.Plot`` component, or None on any failure.

    The spec is the JSON the agent embedded in ``<chart> ... </chart>``: it
    carries its own ``sql``, so the chart is drawn by re-running the query
    here (read-only) — the agent never passes the raw data.
    """
    try:
        import gradio as gr

        result = db.run_sql(spec["sql"])
        figure = _build_chart_figure(spec, result)
        return gr.Plot(value=figure)
    except Exception:  # noqa: BLE001 - don't break the chat on a bad spec
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="garmin-ask-web",
        description="Browser chat over the read-only Garmin agent (Gradio).",
    )
    parser.add_argument(
        "--session",
        metavar="FILE",
        dest="session_file",
        help="conversation history file to save to / resume from "
        "(default: auto-persist and resume the last session)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="local port to serve the UI on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="create a public share link app",
    )
    args = parser.parse_args()

    try:
        cfg = load_config()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1

    import gradio as gr

    session = make_chat_session(cfg, session_path=args.session_file)
    demo = gr.ChatInterface(
        session.respond,
        chatbot=gr.Chatbot(value=session.initial_history),
        title="Garmin AI",
        description="Ask about your Garmin data. Queries are read-only SELECTs "
        f"against {cfg['db_path']}. Ask for a chart and it will be drawn here.",
        fill_height=True,
    )
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())