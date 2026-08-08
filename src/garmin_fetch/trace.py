"""Agent turn tracing: a compact, append-only log of how an agent answered.

Every turn records the user's question plus a step-by-step summary of the
conversation the agent had with itself: tool calls (with their JSON args) and
tool returns (outputs/outcome), in order, plus the model's final answer. The
aim is an inspectable record for tuning the tools — what the model asked for
and what each tool actually returned.

Written as one JSON object per turn to ``GARMIN_TRACE_FILE`` (default
``ask_trace.jsonl``); every write is a fresh append. ``garmin-trace`` prints
that file as a readable transcript.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

#: Hard cap for a single tool-return/content payload stored in the JSONL.
_MAX_PAYLOAD = 5000

_trace_env = os.getenv("GARMIN_TRACE_FILE")
_DEFAULT_FILE = Path(_trace_env) if _trace_env else PROJECT_ROOT / "ask_trace.jsonl"


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"…(+{len(text) - n} more)"


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


def append_turn(
    path: str,
    question: str,
    messages: list[Any],
    answer: str = "",
    model: str = "",
    usage: Any | None = None,
    *,
    _ts: str | None = None,
) -> None:
    """Append one trace record (one JSON line) to ``path``."""
    record = {
        "ts": _ts or datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": question,
        "model": model,
        "steps": summarize(messages),
        "answer": _clip(answer, _MAX_PAYLOAD),
    }
    if (usage_summary := _usage_summary(usage)) is not None:
        record["usage"] = usage_summary
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        description="View the recorded agent tool-call trace (ask_trace.jsonl).",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        default=None,
        help="trace file to read (default: GARMIN_TRACE_FILE or ask_trace.jsonl)",
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

    path = args.file or str(_DEFAULT_FILE)
    records = _iter_records(path)
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