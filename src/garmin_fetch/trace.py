"""Agent turn tracing: a compact, append-only log of how an agent answered.

Every turn records the user's question plus a step-by-step summary of the
conversation the agent had with itself: tool calls (with their JSON args) and
tool returns (outputs/outcome), in order, plus the model's final answer. The
aim is an inspectable record for tuning the tools — what the model asked for
and what each tool actually returned.

Turns are stored per user in Postgres (the ``user_state`` table's ``trace``
key); ``garmin-trace`` prints that transcript. A legacy ``--file`` reads an old
``ask_trace.jsonl`` instead. ``garmin-trace --failures`` instead summarises
where the agent struggles with its tools: every failed tool call (an
``ERROR:`` return, a non-success outcome or a schema retry) grouped by tool
and normalised message, with how often the agent never recovered.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict, deque
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


def _db_records(all_users: bool = False) -> list[dict[str, Any]]:
    """Read the trace transcript from Postgres (the ``trace`` key of
    ``user_state``): ``GARMIN_LOCAL_USER_ID``'s, or with ``all_users`` every
    account's, then each record tagged with its ``user_id`` and all of them
    sorted by time."""
    from .config import load_config
    from .server.state import UserState

    cfg = load_config()
    state = UserState(cfg["db_url"])
    try:
        user_ids = state.user_ids() if all_users else [cfg.get("local_user_id") or 1]
        records: list[dict[str, Any]] = []
        for uid in user_ids:
            raw = state.get(uid, "trace")
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                records.extend(
                    {**r, "user_id": uid} if all_users else r
                    for r in data if isinstance(r, dict)
                )
    finally:
        state.close()
    if all_users:
        records.sort(key=lambda r: r.get("ts", ""))
    return records


#: Quoted names/values inside an error message. They are grouped away, so
#: ``column "hr" does not exist`` and ``column "rpe" does not exist`` are one
#: pattern, and kept as that pattern's variants.
_QUOTED = re.compile(r'"([^"\n]{1,80})"|\'([^\'\n]{1,80})\'')
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
#: Label for a retry with no tool: the output validator sent the answer back.
_OUTPUT_RETRY = "(final answer)"


def _normalise(message: str) -> tuple[str, list[str]]:
    """A message as a groupable pattern plus the quoted values taken out."""
    values = [a or b for a, b in _QUOTED.findall(message)]
    pattern = _QUOTED.sub('"…"', message)
    return _clip(_NUMBER.sub("N", pattern).strip(), 160), values


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _failure(step: dict[str, Any]) -> tuple[str, list[str]] | None:
    """A failed step's pattern and variants, or None when the step did not fail.

    A tool fails by returning ``ERROR: ...``, by a non-success outcome, or by
    a retry prompt (its arguments failed the tool's schema, or it raised
    ``ModelRetry``; a retry with no tool is the output validator)."""
    kind = step.get("kind")
    content = step.get("content") or ""
    if kind == "tool_return":
        text = content.lstrip()
        if text.startswith("ERROR"):
            return _normalise(_first_line(text.removeprefix("ERROR").lstrip(": ")))
        outcome = step.get("outcome") or "success"
        if outcome != "success":
            pattern, values = _normalise(_first_line(text))
            return f"[{outcome}] {pattern}", values
        return None
    if kind != "retry":
        return None
    try:
        errors = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        errors = None
    if isinstance(errors, list) and errors and all(isinstance(e, dict) for e in errors):
        # pydantic validation errors: group by where and why, vary by input.
        parts = sorted({
            f"{'.'.join(str(p) for p in e.get('loc', ())) or 'args'}: {e.get('msg', '')}"
            for e in errors
        })
        values = [_clip(str(e["input"]), 60) for e in errors if "input" in e]
        return _normalise("; ".join(parts))[0], values
    return _normalise(_first_line(content))


def _plain_args(args: str) -> str:
    """A call's args as the model sent them: models send a JSON string, which
    the trace stores JSON-encoded a second time (clipped args stay as is)."""
    try:
        decoded = json.loads(args)
    except json.JSONDecodeError:
        return args
    return decoded if isinstance(decoded, str) else args


def _turn_results(record: dict[str, Any]) -> tuple[Counter[str], list[dict[str, Any]]]:
    """One turn's tool-call counts and its results in order, each paired with
    the call's args (calls and results of one tool are matched first in,
    first out) and flagged ``recovered`` when a later call of the same tool in
    the turn succeeded."""
    calls: Counter[str] = Counter()
    queued: dict[str, deque[str]] = defaultdict(deque)
    results: list[dict[str, Any]] = []
    for step in record.get("steps", []):
        kind = step.get("kind")
        if kind == "tool_call":
            calls[step.get("tool") or ""] += 1
            queued[step.get("tool") or ""].append(step.get("args") or "")
        elif kind in ("tool_return", "retry"):
            tool = step.get("tool") or (_OUTPUT_RETRY if kind == "retry" else "")
            pending = queued.get(tool)
            results.append({
                "tool": tool,
                "args": _plain_args(pending.popleft()) if pending else "",
                "failure": _failure(step),
                "content": step.get("content") or "",
            })
    for i, res in enumerate(results):
        res["recovered"] = any(
            later["tool"] == res["tool"] and later["failure"] is None
            for later in results[i + 1:]
        )
    return calls, results


def _when(record: dict[str, Any]) -> str:
    ts = record.get("ts", "")
    who = f" user {record['user_id']}" if "user_id" in record else ""
    return f"{ts[:16].replace('T', ' ')}{who}"


def failure_report(records: list[dict[str, Any]], *, full: bool = False) -> list[str]:
    """Summarise every failed tool call in ``records``: totals, a per-tool
    failure rate, then each failure pattern (most frequent first) with its
    quoted variants and example turns. ``unrecovered`` counts failures the
    agent never followed with a working call of that tool in the same turn —
    the ones that cost an answer, not just a round trip."""
    limit = None if full else 200
    shown_examples = 3 if full else 1
    calls: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for n, record in enumerate(records):
        turn_calls, results = _turn_results(record)
        calls.update(turn_calls)
        for res in results:
            if res["failure"] is None:
                continue
            pattern, values = res["failure"]
            failed[res["tool"]] += 1
            group = groups.setdefault((res["tool"], pattern), {
                "count": 0, "turns": set(), "unrecovered": 0,
                "variants": Counter(), "examples": [],
            })
            group["count"] += 1
            group["turns"].add(n)
            group["unrecovered"] += not res["recovered"]
            group["variants"].update(values)
            group["last"] = _when(record)
            group["examples"].append((record, res))

    if not records:
        return ["No trace records."]
    total_calls, total_failed = sum(calls.values()), sum(failed.values())
    first, last = records[0].get("ts", "")[:10], records[-1].get("ts", "")[:10]
    lines = [
        f"{len(records)} turns ({first} → {last}), {total_calls} tool calls, "
        f"{total_failed} failed ({total_failed / total_calls:.1%})" if total_calls
        else f"{len(records)} turns ({first} → {last}), no tool calls",
        "",
        "By tool:",
    ]
    for tool in sorted(calls.keys() | failed.keys(), key=lambda t: (-failed[t], -calls[t], t)):
        rate = f"{failed[tool] / calls[tool]:6.1%}" if calls[tool] else "     —"
        lines.append(f"  {tool:<24} {calls[tool]:>5} calls {failed[tool]:>5} failed {rate}")
    if not groups:
        lines += ["", "No failures."]
        return lines

    lines += ["", "Failure patterns (most frequent first):"]
    for (tool, pattern), g in sorted(groups.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        lines.append("")
        lines.append(
            f"  [{tool}] {g['count']}× in {len(g['turns'])} turn(s), "
            f"{g['unrecovered']} unrecovered — last {g['last']}"
        )
        lines.append(f"    {pattern}")
        if g["variants"]:
            top = g["variants"].most_common(None if full else 8)
            lines.append("    values: " + ", ".join(
                f"{v} ×{c}" if c > 1 else v for v, c in top
            ) + ("" if full or len(g["variants"]) <= 8 else f", … (+{len(g['variants']) - 8})"))
        # Newest first: the most recent failures say most about the current tools.
        for record, res in reversed(g["examples"][-shown_examples:]):
            question = record.get("question", "")
            lines.append(f"    e.g. [{_when(record)}] Q> {question if limit is None else _clip(question, 120)}")
            if res["args"]:
                lines.append(f"         -> {res['args'] if limit is None else _clip(res['args'], limit)}")
            error = res["content"] if limit is None else _clip(res["content"], limit)
            lines.extend(f"         <- {ln}" if i == 0 else f"            {ln}"
                         for i, ln in enumerate(error.splitlines() or [""]))
    return lines


def _render_body(record: dict[str, Any], limit: int | None) -> list[str]:
    """Render one record's usage + steps + answer (no question header)."""
    lines: list[str] = []
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


def _render(record: dict[str, Any], full: bool) -> list[str]:
    limit = None if full else 300
    who = f" user {record['user_id']}" if "user_id" in record else ""
    lines = [f"[{record.get('ts', '')}{who}] Q> {record.get('question', '')}"]
    lines.extend(_render_body(record, limit))
    return lines


def _uses_tool(record: dict[str, Any], name: str) -> bool:
    """True when ``name`` appears as a tool call in the record."""
    return any(
        step.get("kind", "").startswith("tool") and step.get("tool") == name
        for step in record.get("steps", [])
    )


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
        help="do not truncate long args/outputs (with --failures: more examples)",
    )
    parser.add_argument(
        "--failures",
        action="store_true",
        help="summarise failed tool calls by tool and error pattern instead of "
        "printing the transcript",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="read every account's trace, not just GARMIN_LOCAL_USER_ID's",
    )
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        default=None,
        help="only turns on or after this date",
    )
    args = parser.parse_args(argv)

    _utf8_console()
    if args.file:
        records = _iter_records(args.file)
    else:
        records = _db_records(all_users=args.all_users)
    if args.since:
        records = [r for r in records if r.get("ts", "")[:10] >= args.since]
    if args.tool:
        records = [r for r in records if _uses_tool(r, args.tool)]
    if args.tail:
        records = records[-args.tail :]
    if args.failures:
        print("\n".join(failure_report(records, full=args.full)))
        return 0
    for record in records:
        print("\n".join(_render(record, full=args.full)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())