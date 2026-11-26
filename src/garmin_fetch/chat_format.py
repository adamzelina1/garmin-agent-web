"""Shaping agent answers and stored sessions for a chat front-end.

Used by the FastAPI server and the agent: turning persisted Pydantic AI
messages back into chat history, pulling ``<chart>`` specs out of an answer,
and swapping pasted training-plan JSON for the ``<plan_table />`` marker.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

#: Tool-call markup a model wrote into its reply *text* instead of making a real
#: call: DeepSeek's DSML tags (``<｜DSML｜invoke …>``, with fullwidth or ASCII
#: bars, in whatever mangled spacing the model produced) and its older
#: ``<｜tool▁calls▁begin｜>``-style tokens.
_LEAK_TAG = re.compile(
    r"</?[\s|｜]*DSML[\s|｜]*[^<>\n]{0,80}>"
    r"|<[|｜]+\s*tool[▁_ ](?:calls?|sep|outputs?)[^<>\n]{0,40}>"
)
#: A line opening a pseudo tag that carries an attribute (``<memory facts='…``):
#: how a mangled call starts when the model drops the opening DSML tags.
_PSEUDO_CALL = re.compile(r"(?m)^[ \t]*<[a-z_][\w.-]*[ \t]+[a-z_]\w*[ \t]*=")
#: Tags the model is *meant* to write in an answer; never a leaked call.
_ANSWER_MARKERS = ("<chart", "<plan_table")


def strip_leaked_tool_call(text: str) -> tuple[str, bool]:
    """Remove a tool call the model wrote as text; returns (clean text, found).

    Some models (DeepSeek in particular) occasionally end a reply with the call
    spelled out — ``<memory facts='{…}'</｜｜DSML｜｜ parameter>…`` — instead of
    using the function-calling interface. The provider then returns it as plain
    content: the call never runs and the markup shows up in the answer. Only
    text containing such markup is touched; everything from the start of the
    block (its pseudo opening tag when there is one, else the first markup tag)
    to the last markup tag is cut, and the prose around it is kept.
    """
    tags = list(_LEAK_TAG.finditer(text))
    if not tags:
        return text, False
    start, end = tags[0].start(), tags[-1].end()
    for match in _PSEUDO_CALL.finditer(text, 0, start):
        if not text[match.start():].lstrip().startswith(_ANSWER_MARKERS):
            start = match.start()
    head, tail = text[:start].rstrip(), text[end:].strip()
    return (f"{head}\n\n{tail}" if head and tail else head or tail), True


def messages_to_history(messages: list[Any]) -> list[dict[str, str]]:
    """Map persisted Pydantic AI messages to chat history (``{role, content}``).

    Assistant responses that also carry tool calls are the model's *working
    narration* ("Let me check…", etc.), not its answer — they are skipped so
    the chat only ever shows the final text-only reply. The terminal answer
    is its own response with no tool calls, so it is always kept.
    """
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
            if any(getattr(p, "tool_name", None) is not None for p in message.parts):
                continue
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


#: How the ``chart`` tool answers a validated chart (see :func:`chart_reply`):
#: the id the model embeds, then the spec on the last line for the server.
_CHART_REPLY = re.compile(r"^OK: chart (c\d+)\b.*\nSpec: (\{.*\})\s*$", re.S)
#: A chart the model placed by reference: ``<chart id="c1"/>`` (also tolerates
#: no quotes, no self-close, or an empty ``</chart>`` right after).
_CHART_REF = re.compile(
    r"""<chart\s+id\s*=\s*["']?(c\d+)["']?\s*/?\s*>(?:\s*</chart\s*>)?""", re.I
)


def chart_reply(chart_id: str, spec: dict[str, Any], detail: str) -> str:
    """The ``chart`` tool's reply for a validated chart.

    The model only has to write ``<chart id="…"/>`` where the chart belongs
    (it used to copy the whole spec into its answer, re-typing every SQL query
    as output tokens); :func:`finish_chart_text` swaps the spec back in. The
    spec rides on the last line so the server can do that from the run's
    messages alone.
    """
    return (
        f"OK: chart {chart_id} ({detail}). Put <chart id=\"{chart_id}\"/> on its "
        "own line where the chart belongs, with one sentence on what it shows; "
        "never paste the spec or the data.\nSpec: "
        + json.dumps(spec, ensure_ascii=False)
    )


def turn_chart_specs(messages: list[Any]) -> dict[str, dict[str, Any]]:
    """``{chart id: spec}`` for every chart the ``chart`` tool validated in
    ``messages`` (one turn's worth). A stored, already-pruned tool reply no
    longer parses, which is fine: its turn was resolved when it was saved."""
    specs: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if getattr(part, "tool_name", None) != "chart":
                continue
            match = _CHART_REPLY.match(str(getattr(part, "content", "") or ""))
            if match:
                data = _parse_block(match.group(2))
                if data is not None:
                    specs[match.group(1)] = data
    return specs


def resolve_chart_refs(text: str, specs: dict[str, dict[str, Any]]) -> str:
    """Replace each ``<chart id="…"/>`` with that chart's full ``<chart>``
    block; a reference to an unknown id is dropped."""

    def block(match: re.Match[str]) -> str:
        spec = specs.get(match.group(1).lower())
        return f"<chart>{json.dumps(spec, ensure_ascii=False)}</chart>" if spec else ""

    return _CHART_REF.sub(block, text)


def unplaced_charts(
    texts: list[str], specs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """The validated charts none of ``texts`` (already resolved) shows,
    matched by SQL."""
    placed = {
        spec.get("sql", "").strip() for text in texts for spec in extract_charts(text)[1]
    }
    return [s for s in specs.values() if s.get("sql", "").strip() not in placed]


def append_charts(text: str, specs: list[dict[str, Any]]) -> str:
    """``text`` with a full ``<chart>`` block per spec added at the end."""
    if not specs:
        return text
    blocks = "\n\n".join(
        f"<chart>{json.dumps(spec, ensure_ascii=False)}</chart>" for spec in specs
    )
    return f"{text.rstrip()}\n\n{blocks}" if text.strip() else blocks


def finish_chart_text(text: str, specs: dict[str, dict[str, Any]]) -> str:
    """An answer with its chart references resolved, plus a block for every
    validated chart it never placed — a model sometimes validates a chart and
    then forgets it (or ends with a short acknowledgement), and the user would
    otherwise never see it. Idempotent: a placed chart is matched by its SQL."""
    text = resolve_chart_refs(text, specs)
    return append_charts(text, unplaced_charts([text], specs))


def extract_charts(
    answer: str, *, slots: bool = False
) -> tuple[str, list[dict[str, Any]]]:
    """Split an agent answer into (plain text, list of plotly figure dicts).

    ``<chart>`` blocks are consumed and their JSON parsed (tolerating a
    surrounding fenced code block). The closing ``</chart>`` tag is *optional*:
    an unclosed block runs to the end of the answer and its JSON prefix is read
    with a raw JSON decoder, so a model that forgets the close tag (or appends
    trailing filler after the JSON) still produces a chart. Malformed blocks
    are dropped along with their marker text. With ``slots`` each chart leaves
    a ``<chart_slot n="i"/>`` marker (its index in the list) where it stood,
    so the web page can draw it in place.
    """
    charts: list[dict[str, Any]] = []
    pieces: list[str] = []

    def keep(data: dict[str, Any] | None) -> None:
        if data is not None:
            charts.append(data)
            if slots:
                pieces.append(f'\n\n<chart_slot n="{len(charts) - 1}"/>\n\n')

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
            keep(_parse_block(inner))
            end = close + len("</chart>")
        else:  # unclosed block: parse JSON prefix, keep the text tail
            data, end = _figure_prefix(answer[start + len("<chart>") :])
            if data is not None:
                keep(data)
                end += start + len("<chart>")
                tail = answer[end:]
                if tail.lstrip().startswith(("```", "~~~")):
                    end = len(answer)  # a leftover closing fence: drop it
            else:
                end = start + len("<chart>")  # malformed: drop the bare tag
        pos = end
    return "".join(pieces), charts


def _workout_dump_list(data: Any) -> Any:
    """The workout list inside a plan-ish dump (or None).

    Handles a bare array and the unified ``training`` tool's windowed
    ``{"weeks": [{"planned": [...]}, ...]}`` agenda (flattened across weeks).
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        weeks = data.get("weeks")
        if isinstance(weeks, list):
            return [
                w for wk in weeks
                if isinstance(wk, dict) and isinstance(wk.get("planned"), list)
                for w in wk["planned"]
            ]
    return None


def _looks_like_workout_dump(data: Any) -> bool:
    """True when ``data`` is training-plan workout JSON."""
    workouts = _workout_dump_list(data)
    if not isinstance(workouts, list) or not workouts:
        return False
    for item in workouts:
        if not isinstance(item, dict):
            return False
        if not (
            isinstance(item.get("planned_date"), str)
            and isinstance(item.get("activity_type"), str)
        ):
            return False
    return True


def _workout_dump_marker(data: Any) -> str:
    """Render the ``<plan_table />`` marker for a plan dump, with date bounds."""
    data = _workout_dump_list(data) or []
    dates = [str(i["planned_date"]) for i in data if i.get("planned_date")]
    attrs = ""
    if dates:
        attrs = f' from="{min(dates)}" to="{max(dates)}"'
    return f"<plan_table{attrs} />"


def replace_workout_dumps(text: str) -> str:
    """Replace verbatim training-plan JSON dumps with a ``<plan_table />`` marker.

    The model is told to show the plan via ``<plan_table />`` (the UI renders
    the stored plan straight from the DB), but it sometimes ignores that and
    pastes the raw JSON the ``training`` tool returned — burning
    tokens and displaying ugly JSON. This post-processes the answer: any JSON
    array whose items look like plan workouts (fenced or bare) is swapped for
    the marker, so the chat always shows the rendered table.
    """
    def _swap(match: re.Match[str]) -> str:
        try:
            data = json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            return match.group(0)
        if not _looks_like_workout_dump(data):
            return match.group(0)
        return _workout_dump_marker(data)

    text = re.sub(r"```(?:json)?\s*\n([\s\S]*?)```", _swap, text)

    out: list[str] = []
    pos = 0
    dec = json.JSONDecoder()
    while True:
        idx = text.find("[", pos)
        if idx == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:idx])
        try:
            data, end = dec.raw_decode(text[idx:])
        except json.JSONDecodeError:
            out.append(text[idx])
            pos = idx + 1
            continue
        if _looks_like_workout_dump(data):
            out.append(_workout_dump_marker(data))
        else:
            out.append(text[idx : idx + end])
        pos = idx + end
    return "".join(out)
