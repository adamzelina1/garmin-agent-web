from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from garmin_fetch.ask import _load_session, build_agent
from garmin_fetch.ask_web import (
    _extract_charts,
    _history_to_messages,
    _messages_to_history,
    _recover_chart_specs,
    make_chat_session,
)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "g.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE daily_metrics (
            calendar_date TEXT PRIMARY KEY,
            sleep_time_hours NUMERIC,
            resting_hr NUMERIC,
            active_hours NUMERIC
        );
        INSERT INTO daily_metrics VALUES
            ('2026-08-01', 7.5, 51, 1.5),
            ('2026-08-02', 8.25, 50, 2.0),
            ('2026-08-03', NULL, 52, NULL);
        """
    )
    conn.commit()
    conn.close()
    return str(p)


def _fake_build(cfg: dict[str, str], db):
    return build_agent(
        db,
        model_name="test",
        base_url=None,
        api_key="x",
        model=TestModel(call_tools=["date_range"], custom_output_text="mock final answer"),
    )


def test_history_roundtrip_and_mapping() -> None:
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    msgs = _history_to_messages(history)
    assert _messages_to_history(msgs) == history
    assert _messages_to_history([]) == []

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart

    msgs = _history_to_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": 42},  # non-str content is coerced
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "here is the answer"},
                    {"type": "some-component"},
                ],
            },
        ]
    )
    assert isinstance(msgs[0], ModelRequest)
    assert isinstance(msgs[1], ModelResponse)
    assert isinstance(msgs[2], ModelRequest)
    assert isinstance(msgs[3], ModelResponse)
    assert msgs[3].parts[0].content == "here is the answer"
    assert isinstance(msgs[3].parts[0], TextPart)
    assert len(msgs) == 4


def test_extract_charts_parses_closed_blocks() -> None:
    figure = {"data": [{"type": "bar", "x": ["a"], "y": [1]}]}
    raw = (
        "Sleep by week:\n"
        f"<chart>{json.dumps(figure)}</chart>\n"
        "That finishes the chart."
    )
    text, charts = _extract_charts(raw)
    assert text == "Sleep by week:\n\nThat finishes the chart."
    assert charts == [figure]

    # Fences and bad JSON inside a closed block are tolerated.
    good = {"layout": {"title": {"text": "x"}}}
    raw = (
        "<chart>\n```json\n"
        f"{json.dumps(good)}"
        "\n```\n</chart>\n"
        "<chart>not json at all</chart>\n"
        "tail"
    )
    text, charts = _extract_charts(raw)
    assert text == "\n\ntail"
    assert len(charts) == 1
    assert charts[0]["layout"]["title"]["text"] == "x"

    # A model piping the tool's ``OK: <spec> (query returned N rows)`` reply
    # verbatim must still render: the JSON prefix survives the trailing note.
    spec = {
        "sql": "SELECT calendar_date FROM daily_metrics",
        "traces": [{"go": "Bar", "x": "calendar_date"}],
        "layout": {},
    }
    raw = f"<chart>  {json.dumps(spec)} (4 rows)</chart>"
    text, charts = _extract_charts(raw)
    assert text == ""
    assert len(charts) == 1
    assert charts[0]["sql"] == spec["sql"]


def test_extract_charts_handles_unclosed_and_malformed() -> None:
    figure = {"data": [{"type": "line", "x": [1, 2], "y": [3, 4]}]}
    raw = (
        "Load by week:\n<chart>"
        f"{json.dumps(figure)}"
        "\ntrailing filler after the JSON with no close tag"
    )
    text, charts = _extract_charts(raw)
    assert charts == [figure]
    assert "trailing filler after the JSON" in text

    # A fenced block without a closing tag is still parsed.
    parsed = (
        "Here is the chart:\n"
        "<chart>\n```json\n"
        f"{json.dumps(figure)}"
        "\n```\n"
    )
    text, charts = _extract_charts(parsed)
    assert charts == [figure]
    assert text == "Here is the chart:\n"

    # Truncated JSON mid-token: no hang, no chart.
    raw = "Some text\n<chart>{this is truncated json mid-"
    text, charts = _extract_charts(raw)
    assert charts == []
    assert text == "Some text\n{this is truncated json mid-"


def test_recover_chart_specs_from_tool_returns() -> None:
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    spec = {
        "sql": "SELECT calendar_date, resting_hr FROM daily_metrics ORDER BY calendar_date",
        "traces": [{"go": "Scatter", "x": "calendar_date", "y": "resting_hr"}],
        "layout": {"title": {"text": "Resting HR"}},
    }
    ok = "OK: " + json.dumps(spec) + " (query returned 3 rows)"
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="chart", content=ok, tool_call_id="1"),
            ]
        )
    ]
    recovered = _recover_chart_specs(messages)
    assert len(recovered) == 1
    assert recovered[0]["sql"] == spec["sql"]

    # A failed chart call contributes nothing.
    failed = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="chart",
                content="ERROR: spec is not valid JSON",
                tool_call_id="2",
                outcome="failed",
            )
        ]
    )
    assert _recover_chart_specs([failed]) == []

    # A non-chart tool's success return is ignored.
    other = ModelRequest(
        parts=[
            ToolReturnPart(tool_name="run_sql", content=ok, tool_call_id="3"),
        ]
    )
    assert _recover_chart_specs([other]) == []


def test_responder_answers_and_keeps_context(db_path: str, monkeypatch) -> None:
    monkeypatch.setattr("garmin_fetch.ask_web._build_agent", _fake_build)
    cfg = {"db_path": db_path, "llm_model": "test", "llm_api_key": "x", "llm_base_url": None}

    session = make_chat_session(cfg)
    msg = session.respond("what is the date range?", [])
    assert msg["role"] == "assistant"
    assert msg["content"][0]["text"] == "mock final answer"

    session.respond(
        "again",
        [
            {"role": "user", "content": "what is the date range?"},
            {"role": "assistant", "content": "mock final answer"},
        ],
    )


def test_responder_persists_and_autoloads_session(db_path: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("garmin_fetch.ask_web._build_agent", _fake_build)
    cfg = {"db_path": db_path, "llm_model": "test", "llm_api_key": "x", "llm_base_url": None}
    session = tmp_path / "web_session.json"

    first = make_chat_session(cfg, session_path=str(session))
    assert first.initial_history == []
    first.respond("what is the date range?", [])
    first.respond(
        "and later?",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}],
    )

    assert session.exists()
    assert len(_load_session(str(session))) == 4  # 2 user + 2 assistant turns

    # The last session is loaded back into the UI on startup.
    second = make_chat_session(cfg, session_path=str(session))
    assert second.initial_history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
        {"role": "user", "content": "and later?"},
        {"role": "assistant", "content": "mock final answer"},
    ]
    msg = second.respond("again", [])
    assert msg["content"][0]["text"] == "mock final answer"


def test_responder_renders_chart_and_ignores_bad_spec(db_path: str, monkeypatch) -> None:
    cfg = {"db_path": db_path, "llm_model": "test", "llm_api_key": "x", "llm_base_url": None}

    spec = {
        "sql": "SELECT calendar_date, resting_hr FROM daily_metrics ORDER BY calendar_date",
        "traces": [
            {"type": "line", "x": "calendar_date", "y": "resting_hr", "name": "Resting HR"}
        ],
        "layout": {"title": {"text": "Resting HR"}},
    }

    def good_build(cfg: dict[str, str], db):
        return build_agent(
            db,
            model_name="test",
            base_url=None,
            api_key="x",
            model=TestModel(
                call_tools=[],
                custom_output_text="Answer." + f"<chart>{json.dumps(spec)}</chart>",
            ),
        )

    monkeypatch.setattr("garmin_fetch.ask_web._build_agent", good_build)
    msg = make_chat_session(cfg).respond("show me a chart", [])
    assert "Plot" in [type(c).__name__ for c in msg["content"]]

    # A spec whose SQL is invalid renders no chart (and does not crash).
    bad = {"sql": "SELECT nope FROM daily_metrics", "traces": []}

    def bad_build(cfg: dict[str, str], db):
        return build_agent(
            db,
            model_name="test",
            base_url=None,
            api_key="x",
            model=TestModel(
                call_tools=[],
                custom_output_text="Answer." + f"<chart>{json.dumps(bad)}</chart>",
            ),
        )

    monkeypatch.setattr("garmin_fetch.ask_web._build_agent", bad_build)
    msg = make_chat_session(cfg).respond("show me a chart", [])
    assert "Plot" not in [type(c).__name__ for c in msg["content"]]