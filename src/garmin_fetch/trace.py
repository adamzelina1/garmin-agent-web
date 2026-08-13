"""Agent turn tracing: a compact, append-only log of how an agent answered.

Every turn records the user's question plus a step-by-step summary of the
conversation the agent had with itself: tool calls (with their JSON args) and
tool returns (outputs/outcome), in order, plus the model's final answer. The
aim is an inspectable record for tuning the tools — what the model asked for
and what each tool actually returned.

Turns are stored per user in Postgres (the ``user_state`` table's ``trace``
key); ``garmin-trace`` prints that transcript. A legacy ``--file`` reads an old
``ask_trace.jsonl`` instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

#: Hard cap for a single tool-return/content payload stored in a trace record.
_MAX_PAYLOAD = 5000


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"…(+{len(text) - n} more)"


def _utf8_console() -> None:
    """Reconfigure stdout/stderr to UTF-8 so emoji in traces render anywhere.

    Windows consoles default to the OEM/cp1252 codepage; printing trace content
    (charts, emoji) then raises UnicodeEncodeError. Overriding the stream's
    encoding sidesteps the console codepage entirely.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def summarize(messages: list[Any]) -> list[dict[str, Any]]:
    """Turn a run's message list into ordered, JSON-safe step records."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    steps: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    steps.append(
                        {"kind": "prompt", "content": _clip(str(part.content), _MAX_PAYLOAD)}
                    )
                elif isinstance(part, ToolReturnPart):
                    steps.append(
                        {
                            "kind": "tool_return",
                            "tool": part.tool_name,
                            "outcome": part.outcome,
                            "content": _clip(str(part.content), _MAX_PAYLOAD),
                        }
                    )
                elif isinstance(part, RetryPromptPart):
                    steps.append(
                        {
                            "kind": "retry",
                            "tool": getattr(part, "tool_name", None) or "",
                            "content": _clip(str(part.content), 1000),
                        }
                    )
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    if hasattr(part.args, "model_dump"):
                        raw: Any = part.args.model_dump()
                    else:
                        raw = str(part.args)
                    steps.append(
                        {
                            "kind": "tool_call",
                            "tool": part.tool_name,
                            "args": _clip(
                                json.dumps(raw, ensure_ascii=False, default=str),
                                2000,
                            ),
                        }
                    )
                elif isinstance(part, TextPart):
                    steps.append(
                        {"kind": "text", "content": _clip(str(part.content), 300)}
                    )
    return steps


def _usage_summary(usage: Any | None) -> dict[str, Any] | None:
    """Project a ``Result.usage()`` object into a small JSON-safe dict."""
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_write_tokens", 0) or 0,
        "cache_hit_ratio": round(getattr(usage, "cache_hit_ratio", 0.0) or 0.0, 3),
        "requests": getattr(usage, "requests", 0) or 0,
        "tool_calls": getattr(usage, "tool_calls", 0) or 0,
    }


def build_trace_record(
    question: str,
    messages: list[Any],
    answer: str = "",
    model: str = "",
    usage: Any | None = None,
    *,
    _ts: str | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe trace record (without writing it anywhere)."""
    record = {
        "ts": _ts or datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": question,
        "model": model,
        "steps": summarize(messages),
        "answer": _clip(answer, _MAX_PAYLOAD),
    }
    if (usage_summary := _usage_summary(usage)) is not None:
        record["usage"] = usage_summary
    return record


def _iter_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def _db_records() -> list[dict[str, Any]]:
    """Read the per-user trace transcript from Postgres (the ``trace`` key of
    ``user_state`` for ``GARMIN_LOCAL_USER_ID``)."""
    from .config import load_config
    from .server.state import UserState

    cfg = load_config()
    state = UserState(cfg["db_url"])
    try:
        raw = state.get(cfg.get("local_user_id") or 1, "trace")
    finally:
        state.close()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _render(record: dict[str, Any], full: bool) -> list[str]:
    limit = None if full else 300
    lines = [f"[{record.get('ts', '')}] Q> {record.get('question', '')}"]
    usage = record.get("usage")
    if usage:
        ratio = usage.get("cache_hit_ratio", 0.0) or 0.0
        lines.append(
            "  [tokens] "
            f"in={usage.get('input_tokens', 0)} "
            f"out={usage.get('output_tokens', 0)} "
            f"cache={ratio * 100:.0f}% "
            f"(read {usage.get('cache_read_tokens', 0)} / "
            f"write {usage.get('cache_write_tokens', 0)}) "
            f"reqs={usage.get('requests', 0)} tools={usage.get('tool_calls', 0)}"
        )
    for step in record.get("steps", []):
        kind = step.get("kind")
        if kind == "tool_call":
            args = step.get("args") or ""
            shown = args if limit is None else _clip(args, limit)
            lines.append(f"  -> {step.get('tool')}({shown})")
        elif kind == "tool_return":
            content = step.get("content") or ""
            shown = content if limit is None else _clip(content, limit)
            prefix = f"  <- {step.get('tool')} [{step.get('outcome', 'success')}]"
            if limit is None or "\n" not in content:
                lines.append(prefix + " " + shown)
            else:
                lines.append(prefix)
                lines.extend(("     " + ln) for ln in shown.splitlines())
        elif kind in ("text", "retry"):
            content = step.get("content") or ""
            shown = content if limit is None else _clip(content, limit)
            lines.append(f"  - {step.get('kind')}: {shown}")
    answer = record.get("answer")
    if answer and "\n" in answer:
        answer = "\n" + answer
    if answer:
        shown = answer if limit is None else _clip(answer, limit)
        lines.append("== answer")
        lines.append(shown)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="garmin-trace",
        description="View the recorded agent tool-call trace (stored in Postgres "
        "for GARMIN_LOCAL_USER_ID).",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        default=None,
        help="read a legacy JSONL trace file instead of the database",
    )
    parser.add_argument(
        "--tail", type=int, default=0,
        help="print only the last N turns (0 = all)",
    )
    parser.add_argument(
        "--tool",
        default=None,
        help="only show turns containing this tool name",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="do not truncate long args/outputs",
    )
    args = parser.parse_args(argv)

    _utf8_console()
    if args.file:
        records = _iter_records(args.file)
    else:
        records = _db_records()
    if args.tool:
        records = [
            r
            for r in records
            if any(
                s.get("kind", "").startswith("tool") and s.get("tool") == args.tool
                for s in r.get("steps", [])
            )
        ]
    if args.tail:
        records = records[-args.tail :]
    for record in records:
        print("\n".join(_render(record, full=args.full)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())