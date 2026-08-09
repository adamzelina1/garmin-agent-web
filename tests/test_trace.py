from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from garmin_fetch.ask import ReadOnlyDB, build_agent
from garmin_fetch.trace import _iter_records, append_turn, main as trace_main


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "g.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE daily_metrics (
            calendar_date TEXT PRIMARY KEY,
            resting_hr NUMERIC
        );
        INSERT INTO daily_metrics VALUES ('2026-08-01', 51), ('2026-08-02', 50);
        """
    )
    conn.commit()
    conn.close()
    return str(p)


def _run_agent(db_path: str, question: str) -> object:
    db = ReadOnlyDB(db_path)
    agent = build_agent(
        db,
        model_name="test",
        base_url=None,
        api_key="x",
        model=TestModel(call_tools=["date_range"], custom_output_text="1..2"),
    )
    return agent.run_sync(question)


def test_append_turn_records_steps_and_usage(db_path: str, tmp_path) -> None:
    result = _run_agent(db_path, "what range?")
    path = tmp_path / "trace.jsonl"
    append_turn(
        str(path),
        "what range?",
        result.new_messages(),
        answer=str(result.output),
        usage=result.usage,
    )

    records = _iter_records(str(path))
    assert len(records) == 1
    rec = records[0]
    assert rec["question"] == "what range?"
    assert rec["answer"] == "1..2"
    kinds = [s["kind"] for s in rec["steps"]]
    assert "tool_call" in kinds
    assert "tool_return" in kinds
    call = next(s for s in rec["steps"] if s["kind"] == "tool_call")
    assert call["tool"] == "date_range"
    rtrn = next(s for s in rec["steps"] if s["kind"] == "tool_return")
    assert rtrn["tool"] == "date_range"
    assert rtrn["outcome"] == "success"
    # Token usage is projected when provided.
    assert "input_tokens" in rec["usage"]
    assert "output_tokens" in rec["usage"]
    assert "cache_read_tokens" in rec["usage"]
    assert "cache_hit_ratio" in rec["usage"]


def test_append_turn_appends_jsonl_and_missing_file_is_empty(
    db_path: str, tmp_path
) -> None:
    assert _iter_records(str(tmp_path / "nope.jsonl")) == []

    path = tmp_path / "trace.jsonl"
    for q in ("one", "two"):
        result = _run_agent(db_path, q)
        append_turn(str(path), q, result.new_messages(), answer=str(result.output))

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["steps"]


def test_trace_main_renders(db_path: str, tmp_path, capsys) -> None:
    result = _run_agent(db_path, "what range?")
    path = tmp_path / "trace.jsonl"
    append_turn(str(path), "what range?", result.new_messages(), answer=str(result.output))

    assert trace_main(["--file", str(path), "--tail", "1"]) == 0
    out = capsys.readouterr().out
    assert "Q> what range?" in out
    assert "date_range" in out
    assert "1..2" in out

    assert trace_main(["--file", str(path), "--tool", "run_sql"]) == 0
    assert "Q>" not in capsys.readouterr().out