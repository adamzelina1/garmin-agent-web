"""Browser chat UI (Gradio) over the read-only Garmin agent.

A thin front-end for ``garmin-ask``: the same agent, tools, model config
(cloud / Ollama), long-term memory profile and read-only safety are reused
unchanged; Gradio renders the conversation.

Automatic session persistence: every web chat is written to the per-user
``user_state`` table after each turn and the last session is loaded back into
the UI on startup, so picking up where you left off is the default.

Charts: the agent can call the ``chart`` tool, which runs a read-only query
and returns Plotly chart JSON embedded in ``<chart> ... </chart>`` tags. This
module parses the markers and renders each one as a ``gr.Plot`` beneath the
answer text (chart JSON is stripped from the persisted session).
"""

from __future__ import annotations

import argparse
from typing import Any

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from .ask import (
    ReadOnlyDB,
    _build_agent,
    _build_chart_figure,
    _final_answer,
    _open_readonly,
    _record_turn,
)
from .chat_format import extract_charts, messages_to_history, recover_chart_specs
from .config import load_config


def _history_to_messages(
    history: list[dict[str, Any]],
) -> list[Any]:
    """Map Gradio's openai-style history to Pydantic AI messages.

    Only the text of user/assistant turns is kept — tool call/result details
    and chart blocks are dropped, which is all the model needs to hold context.
    """
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


class ChatSession:
    """One chat app instance: lazy agent, DB-persisted conversation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from .server.state import TrainingStore, UserState

        self.cfg = cfg
        self._holder: dict[str, Any] = {}
        self._state = UserState(cfg["db_url"])
        self._training = TrainingStore(cfg["db_url"])
        self._user_id = cfg.get("local_user_id") or 1
        self.initial_history: list[dict[str, str]] = []

        resumed = self._state.get_session_messages(self._user_id)
        if resumed:
            self.initial_history = messages_to_history(resumed)
            print(
                f"resumed {len(resumed)} prior message(s) from the database "
                f"(user {self._user_id})"
            )

    def close(self) -> None:
        """Release the shared pools (called when the Gradio server stops)."""
        self._training.close()
        self._state.close()

    def _agent(self) -> Any:
        from .server.state import PgMemory, PgPrinciples, TrainingSeason

        if "agent" not in self._holder:
            db = _open_readonly(self.cfg)
            self._holder["db"] = db
            self._holder["agent"] = _build_agent(
                self.cfg,
                db,
                memory=PgMemory(self._state, self._user_id),
                principles=PgPrinciples(self._state, self._user_id),
                training=TrainingSeason(self._training, self._user_id, today=db.today),
            )
        return self._holder["agent"]

    def respond(
        self, message: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        history = list(history) or list(self.initial_history)
        result = self._agent().run_sync(
            message, message_history=_history_to_messages(history)
        )
        _record_turn(
            self.cfg,
            message,
            result,
            trace_writer=lambda r: self._state.append_trace(self._user_id, r),
        )
        text, specs = extract_charts(_final_answer(result))

        # Fallback: the model may validate a chart via the tool but forget to
        # embed the spec in <chart> tags. Recover any validated specs that were
        # not embedded so the chart still renders.
        embedded_sql = {spec.get("sql") for spec in specs}
        for spec in recover_chart_specs(result.new_messages()):
            if spec["sql"] not in embedded_sql:
                specs.append(spec)
                embedded_sql.add(spec["sql"])

        self._state.set_session_messages(self._user_id, result.all_messages())

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


def make_chat_session(cfg: dict[str, Any]) -> ChatSession:
    """Build a :class:`ChatSession` for ``gr.ChatInterface``."""
    return ChatSession(cfg)


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

    try:
        import gradio as gr
    except ImportError:
        print("error: the Gradio UI is an optional extra — install it with "
              "`uv sync --extra gradio-ui`")
        return 1

    session = make_chat_session(cfg)
    demo = gr.ChatInterface(
        session.respond,
        chatbot=gr.Chatbot(
            value=session.initial_history,
            # Render both inline $...$ and block $$...$$ LaTeX, not just the
            # default block form, so the agent's math markup never shows raw.
            latex_delimiters=[
                {"left": "$", "right": "$", "display": False},
                {"left": "$$", "right": "$$", "display": True},
            ],
        ),
        title="Garmin AI",
        description="Ask about your Garmin data. Queries are read-only SELECTs "
        "against Postgres. Ask for a chart and it will be drawn here.",
        fill_height=True,
    )
    try:
        demo.launch(server_port=args.port, share=args.share, inbrowser=True)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())