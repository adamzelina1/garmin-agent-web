from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from garmin_fetch.ask import (
    Memory,
    ReadOnlyDB,
    Weather,
    _ask_session,
    _build_chart_figure,
    _chart_spec_error,
    build_agent,
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
        CREATE TABLE metrics (
            data_type TEXT,
            calendar_date TEXT,
            raw_data TEXT
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


def test_tables_and_columns(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    # Only the agent-facing tables are visible.
    assert db.tables() == ["daily_metrics"]
    assert "metrics" not in db.tables()
    names = {c["name"] for c in db.columns("daily_metrics")}
    assert {"calendar_date", "sleep_time_hours", "resting_hr"} <= names
    with pytest.raises(ValueError):
        db.columns("metrics")


def test_hr_zones_table_visible_to_agent(tmp_path: Path) -> None:
    p = tmp_path / "g.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE daily_metrics (
            calendar_date TEXT PRIMARY KEY,
            resting_hr NUMERIC
        );
        CREATE TABLE hr_zones (
            sport TEXT PRIMARY KEY, zone2_min REAL, zone2_max REAL
        );
        INSERT INTO hr_zones VALUES ('DEFAULT', 134, 147);
        """
    )
    conn.commit()
    conn.close()

    db = ReadOnlyDB(str(p))
    assert "hr_zones" in db.tables()
    out = db.run_sql("SELECT sport, zone2_min, zone2_max FROM hr_zones")
    assert out["rows"] == [["DEFAULT", 134, 147]]


def test_activity_detail_series_table_visible_to_agent(tmp_path: Path) -> None:
    p = tmp_path / "g.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE daily_metrics (
            calendar_date TEXT PRIMARY KEY
        );
        CREATE TABLE activity_detail_series (
            activity_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            heart_rate REAL,
            cadence REAL,
            PRIMARY KEY (activity_id, tick)
        );
        INSERT INTO activity_detail_series VALUES (1001, 0, 150, 84);
        INSERT INTO activity_detail_series VALUES (1001, 1, 165, 88);
        """
    )
    conn.commit()
    conn.close()

    db = ReadOnlyDB(str(p))
    assert "activity_detail_series" in db.tables()
    out = db.run_sql(
        "SELECT COUNT(*) AS n FROM activity_detail_series "
        "WHERE activity_id = 1001 AND heart_rate > 160"
    )
    assert out["rows"] == [[1]]


def test_run_sql_rejects_non_allowed_tables(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    bad = [
        "SELECT * FROM metrics",
        "SELECT * FROM metrics AS m WHERE m.resting_hr > 50",
        "SELECT * FROM sqlite_master",
        "PRAGMA table_info(metrics)",
        "SELECT data_type, COUNT(*) FROM metrics GROUP BY data_type",
    ]
    for sql in bad:
        with pytest.raises(ValueError, match="outside the allowed set"):
            db.run_sql(sql)


def test_date_range(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    assert db.date_range() == {"min": "2026-08-01", "max": "2026-08-03", "rows": 3}


def test_run_sql_select(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    out = db.run_sql(
        "SELECT calendar_date, sleep_time_hours FROM daily_metrics "
        "ORDER BY calendar_date"
    )
    assert out["columns"] == ["calendar_date", "sleep_time_hours"]
    assert out["rows"][0] == ["2026-08-01", 7.5]
    assert out["truncated"] is False


def test_run_sql_rejects_writes_and_multi(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    bad = [
        "DELETE FROM daily_metrics",
        "UPDATE daily_metrics SET resting_hr=0",
        "INSERT INTO daily_metrics VALUES ('x',1,2)",
        "DROP TABLE daily_metrics",
        "ALTER TABLE daily_metrics ADD COLUMN c",
        "CREATE TABLE foo (x)",
        "SELECT 1; SELECT 2",
    ]
    for sql in bad:
        with pytest.raises(ValueError):
            db.run_sql(sql)


def test_authorizer_blocks_writes_on_connection(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    with db._connect() as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE daily_metrics SET resting_hr=0").fetchall()
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("INSERT INTO metrics VALUES ('x','y','z')").fetchall()
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DROP TABLE daily_metrics").fetchall()
    # Reads still work afterwards.
    assert db.run_sql("SELECT COUNT(*) FROM daily_metrics")["rows"] == [[3]]


def test_agent_constructs_offline(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    agent = build_agent(
        db,
        model_name="deepseek-v4-flash",
        base_url="http://localhost:11434/v1",
        api_key="x",
    )
    assert agent is not None


def test_agent_end_to_end_with_test_model(db_path: str) -> None:
    from pydantic_ai.models.test import TestModel

    db = ReadOnlyDB(db_path)
    agent = build_agent(
        db,
        model_name="test",
        base_url=None,
        api_key="x",
        model=TestModel(call_tools=["date_range"], custom_output_text="mock final answer"),
    )
    res = agent.run_sync("what is the date range?")
    assert res.output == "mock final answer"


def _chart_result(db: ReadOnlyDB, sql: str) -> dict:
    return db.run_sql(sql)


def test_chart_spec_builds_line_figure(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    result = _chart_result(
        db,
        "SELECT calendar_date, resting_hr FROM daily_metrics ORDER BY calendar_date",
    )
    spec = {
        "sql": "SELECT calendar_date, resting_hr FROM daily_metrics ORDER BY calendar_date",
        "traces": [
            {
                "type": "line",
                "x": "calendar_date",
                "y": "resting_hr",
                "name": "Resting HR",
            }
        ],
        "layout": {"title": {"text": "Resting HR"}},
    }
    fig = _build_chart_figure(spec, result)
    data = fig.to_dict()["data"]
    assert data[0]["type"] == "scatter"
    assert data[0]["mode"] == "lines+markers"
    assert len(data[0]["y"]) == 3
    assert fig.layout.title.text == "Resting HR"


def test_chart_spec_builds_bar(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    result = _chart_result(db, "SELECT calendar_date, resting_hr FROM daily_metrics")
    fig = _build_chart_figure(
        {
            "sql": "SELECT calendar_date, resting_hr FROM daily_metrics",
            "traces": [{"type": "bar", "x": "calendar_date", "y": "resting_hr"}],
        },
        result,
    )
    assert fig.to_dict()["data"][0]["type"] == "bar"


def test_chart_spec_builds_histogram(db_path: str) -> None:
    db = ReadOnlyDB(db_path)
    result = _chart_result(db, "SELECT resting_hr FROM daily_metrics")
    fig = _build_chart_figure(
        {
            "sql": "SELECT resting_hr FROM daily_metrics",
            "traces": [{"type": "histogram", "x": "resting_hr"}],
        },
        result,
    )
    assert fig.to_dict()["data"][0]["type"] == "histogram"


def test_chart_spec_errors() -> None:
    spec = {
        "sql": "SELECT calendar_date, resting_hr FROM daily_metrics",
        "traces": [{"type": "line", "x": "nope", "y": "resting_hr"}],
    }
    assert "not in" in _chart_spec_error(spec, {"columns": ["calendar_date"], "rows": [["a"]]})
    assert "traces" in _chart_spec_error({}, {"columns": [], "rows": [[1]]})
    assert (
        "one of" in _chart_spec_error(
            {"traces": [{"type": "bogus", "x": "a", "y": "b"}]},
            {"columns": ["a", "b"], "rows": [[1, 2]]},
        )
    )
    assert (
        _chart_spec_error(
            {"traces": [{"type": "histogram", "x": "calendar_date"}]},
            {"columns": ["calendar_date"], "rows": [["a"]]},
        )
        is None
    )
    assert "layout" in _chart_spec_error(
        {"sql": "x", "traces": [{"type": "bar", "x": "a", "y": "b"}], "layout": []},
        {"columns": ["a", "b"], "rows": [[1, 2]]},
    )
    assert _chart_spec_error(
        {"sql": "x", "traces": [{"type": "bar", "x": "a", "y": "b"}], "layout": {}},
        {"columns": ["a", "b"], "rows": [[1, 2]]},
    ) is None
    assert _chart_spec_error(
        {"sql": "x", "traces": [{"type": "bar", "x": "a", "y": "b"}]},
        {"columns": ["a", "b"], "rows": []},
    )


def test_agent_chart_tool_runs_offline(db_path: str) -> None:
    from pydantic_ai.models.test import TestModel

    db = ReadOnlyDB(db_path)
    agent = build_agent(
        db,
        model_name="test",
        base_url=None,
        api_key="x",
        model=TestModel(call_tools=["chart"], custom_output_text="done"),
    )
    res = agent.run_sync("plot sleep by day")
    assert res.output == "done"


def test_memory_roundtrip_and_get(tmp_path) -> None:
    path = tmp_path / "memory.json"
    mem = Memory(str(path))
    assert mem.get() == {}
    mem.remember("training_goal", "run 10k under 45 min")
    mem.remember("sleep_issue", "often wakes around 3am")
    assert mem.get() == {
        "training_goal": "run 10k under 45 min",
        "sleep_issue": "often wakes around 3am",
    }
    assert mem.forget("training_goal")
    assert not mem.forget("training_goal")
    assert mem.get() == {"sleep_issue": "often wakes around 3am"}
    # the file on disk is what a later session reads
    assert Memory(str(path)).get() == {"sleep_issue": "often wakes around 3am"}


def test_memory_remembers_tolerates_corruption(tmp_path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("not json", encoding="utf-8")
    assert Memory(str(path)).get() == {}
    Memory(str(path)).remember("k", "v")
    assert Memory(str(path)).get() == {"k": "v"}


def test_memory_validates_keys_and_values(tmp_path) -> None:
    mem = Memory(str(tmp_path / "memory.json"))
    with pytest.raises(ValueError):
        mem.remember("   ", "x")
    with pytest.raises(ValueError):
        mem.remember("k" * 81, "x")
    with pytest.raises(ValueError):
        mem.remember("k", "v" * 2001)


def test_memory_write_retries_when_destination_locked(tmp_path, monkeypatch) -> None:
    """Windows can transiently hold the target file (WinError 32); the write
    must retry the replace rather than fail the whole turn."""
    import os

    import garmin_fetch.ask as ask_mod

    path = tmp_path / "memory.json"
    mem = Memory(str(path))

    real_replace = os.replace
    attempts = {"n": 0}

    def flaky_replace(src, dst):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise PermissionError(32, "being used by another process", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(ask_mod.os, "replace", flaky_replace)

    mem.remember("k", "v")
    assert attempts["n"] == 3
    assert mem.get() == {"k": "v"}
    # no temp files left behind
    assert [p.name for p in tmp_path.iterdir()] == ["memory.json"]


def test_agent_memory_get_tool_offline(db_path: str, tmp_path) -> None:
    from pydantic_ai.models.test import TestModel

    mem = Memory(str(tmp_path / "memory.json"))
    mem.remember("topic", "marathon training")
    db = ReadOnlyDB(db_path)
    agent = build_agent(
        db,
        model_name="test",
        base_url=None,
        api_key="x",
        model=TestModel(call_tools=["get_memory"], custom_output_text="ok"),
        memory=mem,
    )
    res = agent.run_sync("do you remember anything about me?")
    assert res.output == "ok"


def test_session_loop_keeps_context(db_path: str, monkeypatch, capsys) -> None:
    from pydantic_ai.models.test import TestModel

    def fake_build(db, *, model_name, base_url=None, api_key=None, model=None, memory=None, weather=None):
        return build_agent(
            db,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            model=TestModel(
                call_tools=["date_range"], custom_output_text="mock final answer"
            ),
        )

    monkeypatch.setattr("garmin_fetch.ask.build_agent", fake_build)
    answers = iter(["what is the date range?", "", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    cfg = {
        "db_path": db_path,
        "llm_model": "test",
        "llm_api_key": "x",
        "llm_base_url": None,
    }
    _ask_session(cfg)
    out = capsys.readouterr().out
    assert out.count("mock final answer") == 1  # one answer, then exit on "quit"


def test_session_file_roundtrip(db_path: str, tmp_path) -> None:
    from pydantic_ai.models.test import TestModel

    from garmin_fetch.ask import _load_session, _save_session

    db = ReadOnlyDB(db_path)
    agent = build_agent(
        db,
        model_name="test",
        base_url=None,
        api_key="x",
        model=TestModel(call_tools=["date_range"], custom_output_text="mock final answer"),
    )
    res = agent.run_sync("what is the date range?")
    path = tmp_path / "session.json"
    _save_session(str(path), res.all_messages())
    loaded = _load_session(str(path))
    assert len(loaded) == len(res.all_messages())
    assert _load_session(str(tmp_path / "missing.json")) == []


def test_session_with_file_persistence(db_path: str, tmp_path, monkeypatch, capsys) -> None:
    from pydantic_ai.models.test import TestModel

    from garmin_fetch.ask import _ask_session

    def fake_build(db, *, model_name, base_url=None, api_key=None, model=None, memory=None, weather=None):
        return build_agent(
            db,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            model=TestModel(
                call_tools=["date_range"], custom_output_text="mock final answer"
            ),
        )

    monkeypatch.setattr("garmin_fetch.ask.build_agent", fake_build)
    session = tmp_path / "session.json"
    cfg = {
        "db_path": db_path,
        "llm_model": "test",
        "llm_api_key": "x",
        "llm_base_url": None,
    }

    answers = iter(["first question", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    _ask_session(cfg, str(session))
    assert session.exists()

    capsys.readouterr()
    answers = iter(["follow up", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    _ask_session(cfg, str(session))
    out = capsys.readouterr().out
    assert "Resumed" in out  # history was loaded from the file on resume


def test_session_clear_command_drops_context(db_path, tmp_path, monkeypatch, capsys) -> None:
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from pydantic_ai.models.test import TestModel

    from garmin_fetch.ask import _load_session

    def fake_build(db, *, model_name, base_url=None, api_key=None, model=None, memory=None, weather=None):
        return build_agent(
            db,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            model=TestModel(call_tools=["date_range"], custom_output_text="mock final answer"),
        )

    monkeypatch.setattr("garmin_fetch.ask.build_agent", fake_build)
    session = tmp_path / "session.json"
    cfg = {
        "db_path": db_path,
        "llm_model": "test",
        "llm_api_key": "x",
        "llm_base_url": None,
    }

    answers = iter(["question one", "/clear", "question two", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    _ask_session(cfg, str(session))

    out = capsys.readouterr().out
    assert "no prior context" in out
    dumped = ModelMessagesTypeAdapter.dump_json(_load_session(str(session))).decode()
    assert "question two" in dumped
    assert "question one" not in dumped  # /clear wiped the earlier context


def test_session_new_command_compacts_context(db_path, tmp_path, monkeypatch, capsys) -> None:
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from pydantic_ai.models.test import TestModel

    from garmin_fetch.ask import _load_session

    def fake_build(db, *, model_name, base_url=None, api_key=None, model=None, memory=None, weather=None):
        return build_agent(
            db,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            model=TestModel(call_tools=["date_range"], custom_output_text="COMPACTED SUMMARY"),
        )

    monkeypatch.setattr("garmin_fetch.ask.build_agent", fake_build)
    session = tmp_path / "session.json"
    cfg = {
        "db_path": db_path,
        "llm_model": "test",
        "llm_api_key": "x",
        "llm_base_url": None,
    }

    answers = iter(["question one", "/new", "question two", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    _ask_session(cfg, str(session))

    out = capsys.readouterr().out
    assert "Compacted" in out
    dumped = ModelMessagesTypeAdapter.dump_json(_load_session(str(session))).decode()
    assert "COMPACTED SUMMARY" in dumped  # the summary seeds the new session
    assert "question two" in dumped
    assert "question one" not in dumped  # raw history was folded into the summary


def test_session_new_with_empty_history_starts_fresh(db_path, tmp_path, monkeypatch, capsys) -> None:
    from pydantic_ai.models.test import TestModel

    def fake_build(db, *, model_name, base_url=None, api_key=None, model=None, memory=None, weather=None):
        return build_agent(
            db,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            model=TestModel(call_tools=["date_range"], custom_output_text="mock final answer"),
        )

    monkeypatch.setattr("garmin_fetch.ask.build_agent", fake_build)
    cfg = {
        "db_path": db_path,
        "llm_model": "test",
        "llm_api_key": "x",
        "llm_base_url": None,
    }

    answers = iter(["/new", "question one", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    _ask_session(cfg)
    out = capsys.readouterr().out
    assert "Nothing to compact" in out
    assert out.count("mock final answer") == 1


_OM_PAYLOAD = {
    "latitude": 51.5,
    "longitude": -0.1,
    "timezone": "Europe/London",
    "daily": {
        "time": ["2026-07-01", "2026-07-02"],
        "temperature_2m_max": [22.3, 24.1],
        "temperature_2m_min": [13.2, 14.0],
        "precipitation_sum": [0.0, 2.5],
        "wind_speed_10m_max": [18.0, 25.3],
    },
}


def test_weather_historical_range_from_archive() -> None:
    calls: list[tuple[str, dict]] = []

    def fake_get(url: str, params: dict) -> dict:
        calls.append((url, params))
        return _OM_PAYLOAD

    w = Weather(default_lat=51.5, default_lon=-0.1, http_get=fake_get, today=date(2026, 8, 8))
    out = w.query("2026-07-01", "2026-07-02")

    assert out["source"] == "historical"
    assert out["location"] == {"lat": 51.5, "lon": -0.1}
    assert out["days"] == [
        {"date": "2026-07-01", "temp_max_c": 22.3, "temp_min_c": 13.2, "precip_mm": 0.0, "wind_max_kmh": 18.0},
        {"date": "2026-07-02", "temp_max_c": 24.1, "temp_min_c": 14.0, "precip_mm": 2.5, "wind_max_kmh": 25.3},
    ]
    url, params = calls[0]
    assert url == Weather.ARCHIVE_URL
    assert params["start_date"] == "2026-07-01" and params["end_date"] == "2026-07-02"
    assert params["latitude"] == 51.5 and params["longitude"] == -0.1


def test_weather_forecast_default_and_override() -> None:
    calls: list[tuple[str, dict]] = []

    def fake_get(url: str, params: dict) -> dict:
        calls.append((url, params))
        return _OM_PAYLOAD

    w = Weather(default_lat=51.5, default_lon=-0.1, http_get=fake_get, today=date(2026, 8, 8))
    out = w.query(lat=48.8, lon=2.3)

    assert out["source"] == "forecast"
    assert out["location"] == {"lat": 48.8, "lon": 2.3}
    assert len(out["days"]) == 2
    assert calls[0][1]["start_date"] == "2026-08-08" and calls[0][1]["end_date"] == "2026-08-09"
    assert calls[0][0] == Weather.FORECAST_URL


def test_weather_from_config_and_no_default_location() -> None:
    w = Weather.from_config({"weather_home_lat": "51.5", "weather_home_lon": "-0.1"})
    assert (w._default_lat, w._default_lon) == (51.5, -0.1)

    bare = Weather(http_get=lambda *_: _OM_PAYLOAD, today=date(2026, 8, 8))
    with pytest.raises(ValueError, match="no location configured"):
        bare.query("2026-07-01", "2026-07-02")


def test_weather_query_validates_input() -> None:
    w = Weather(default_lat=51.5, default_lon=-0.1, http_get=lambda *_: _OM_PAYLOAD, today=date(2026, 8, 8))
    with pytest.raises(ValueError, match="not be before"):
        w.query("2026-07-02", "2026-07-01")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        w.query("july", "2026-07-02")
    with pytest.raises(ValueError, match="crosses today"):
        w.query("2026-08-01", "2026-08-09")
    with pytest.raises(ValueError, match="forecast only covers"):
        w.query("2026-08-08", "2026-08-20")
    with pytest.raises(ValueError, match="at most 92 days"):
        w.query("2026-01-01", "2026-08-08")
    with pytest.raises(ValueError, match="both lat and lon"):
        w.query("2026-07-01", "2026-07-02", lat=51.5)
    with pytest.raises(ValueError, match="both date_start and date_end"):
        w.query("2026-07-01")
    with pytest.raises(ValueError, match="out of range"):
        w.query("2026-07-01", "2026-07-02", lat=95.0, lon=0.0)


def test_agent_weather_tool_runs_offline(db_path: str) -> None:
    from pydantic_ai.models.test import TestModel

    class FakeWeather:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def query(self, date_start=None, date_end=None, lat=None, lon=None):
            self.calls.append((date_start, date_end, lat, lon))
            return {"source": "forecast", "days": [{"date": "2026-08-08", "temp_max_c": 21.0}]}

    fake = FakeWeather()
    db = ReadOnlyDB(db_path)
    agent = build_agent(
        db,
        model_name="test",
        base_url=None,
        api_key="x",
        model=TestModel(call_tools=["weather"], custom_output_text="done"),
        weather=fake,
    )
    res = agent.run_sync("what's the weather tomorrow?")
    assert res.output == "done"
    assert fake.calls  # the model actually called the weather tool