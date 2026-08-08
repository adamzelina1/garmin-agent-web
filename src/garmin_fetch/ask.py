"""AI agent that answers questions by querying the local Garmin database.

The agent layer is deliberately thin: a ``ReadOnlyDB`` executor that exposes
schema introspection and safe ``SELECT`` queries, wrapped in a Pydantic AI
agent whose tools let the model inspect and query the SQLite store. All data
access is read-only by construction (``PRAGMA query_only`` + an authorizer
that denies write opcodes + a statement gate), so a model can never mutate
``garmin.db``.

The model/provider is swappable: point ``LLM_BASE_URL`` at Ollama (local,
data stays on-machine) or leave it unset to use the OpenAI API
(``OPENAI_API_KEY``).

The agent also has a stateless ``weather`` tool (Open-Meteo archive + short
forecast) to contextualise stored facts — it never writes anything, so the
SQLite store stays the sole source of truth.

``garmin-ask`` runs a one-shot query, or (with no question argument) an
interactive session that threads the conversation history through every turn
so the model keeps context across questions. Pass ``--session FILE`` to
persist that history after each turn and resume it on a later run.

The interactive session understands two extra commands:
``/clear`` wipes the context and starts a new session with no prior history;
``/new`` asks the model to collapse the current context into a compact summary
and starts a new session seeded with that summary (so nothing is lost, but the
token footprint shrinks).
"""

from __future__ import annotations

import json
import re
import os
import secrets
import sqlite3
import time
from argparse import ArgumentParser, Namespace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from .config import load_config

#: Guard a statement is a read-only query and not a write.
_SELECT_PREFIX = re.compile(r"^\s*(?:SELECT|WITH|EXPLAIN|PRAGMA)\b", re.IGNORECASE)
_WRITE_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REINDEX|VACUUM|"
    r"REPLACE|TRIGGER)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 200

#: Tables the agent may see and query. Everything else (raw ``metrics``,
#: ``activities``, ``user_profile``, ``sync_state``) stays invisible to the model.
_ALLOWED_TABLES = ("daily_metrics", "activity_summaries", "activity_detail_series", "hr_zones")

#: How the model should interpret the stored metrics (units, NULL semantics).
_SYSTEM_PROMPT = """\
You are an analyst with read-only access to a local SQLite database of personal
Garmin health data. Today is {today}. Answer the user's questions by inspecting
the schema and running read-only SELECT queries. You may run several queries.
You can never write.

Only four tables are available and you may only query these:
- daily_metrics: one row per calendar_date, wide columns.
  calendar_date is YYYY-MM-DD.
- activity_summaries: one row per activity (activity_id), curated summary fields.
- activity_detail_series: the intra-activity time series, one row per tick
  (activity_id + tick). Columns: heart_rate, cadence, power_w, speed_mps,
  elevation_m, distance_m (cumulative metres), latitude, longitude, ts_ms
  (epoch ms). A sport's device may not record a metric, in which case that
  column is NULL for every tick. Prefer aggregate/interval queries over this
  table (e.g. per-km buckets via distance_m, or WHERE heart_rate > X).
- hr_zones: the user's configured heart-rate zone boundaries, one row per
  sport (sport: DEFAULT/RUNNING/CYCLING/...). zoneN_min..zoneN_max give each
  zone's bpm range (e.g. zone2_min..zone2_max is the user's Zone 2). This is
  the *current* device configuration snapshot, not historical.

Units and semantics:
- Sleep durations (sleep_time_hours, nap_time_hours, deep_sleep_hours,
  light_sleep_hours, rem_sleep_hours, awake_sleep_hours,
  unmeasurable_sleep_hours) are in HOURS.
- sleep_start_local / sleep_end_local are wall-clock 'HH:MM' strings.
- daily_metrics.total_distance_m is in METRES. activity_summaries.distance_km
  is in KM; its durations (duration_hours etc.) are in HOURS.
- body battery is 0..100; stress and heart rate are bpm; hrv_last_night_avg is
  an HRV score; sleep_score is 0..100.
- Almost every column may be NULL on days where the value is N/A. Only use
  aggregate functions over the rows you have.

Result sets are capped at 200 rows, so prefer aggregated queries (GROUP BY
weeks/months/weekdays) over dumping raw rows. Never use SELECT *; list only
the columns you need (daily_metrics is ~50 columns wide and activity_summaries
~38). Keep result sets tiny: select a handful of columns and aggregate to at
most ~100-200 rows, ORDER BY time columns so trends read naturally. Prefer
small probes — e.g. query with LIMIT 10 to sanity-check columns before a
full aggregation.

When asked for a chart or visualization: call the chart tool to validate a
spec. The spec is a JSON object of your OWN design (you write the graph, not
the data): {{"sql": "SELECT ...", "traces": [{{"type": "...", "x": "...", "y":
"...", "name": "..."}}], "layout": {{"title": {{"text": "..."}}}}}}. The `sql`
must return the plotted data itself (column names x/y reference its result).
Keep points to at most ~200 by aggregating (GROUP BY week/month/weekday) and
ORDER BY time columns. The chart tool runs the query, checks your columns
exist, and returns "OK: <spec>" if valid. Embed that returned spec JSON
VERBATIM in the final answer wrapped in <chart> ... </chart> (one per chart)
and add a short one-sentence description of what it shows as normal text.
Available trace types: line, scatter, area, bar, pie, histogram, box. Do NOT
paste query data or Python code into your answer — only the spec.

Weather context: a `weather` tool returns daily weather for a place: min/max
temperature (degC), precipitation (mm) and max wind (km/h). Pass `date_start`/
`date_end` as YYYY-MM-DD — a range fully before today is historical weather, and
a range starting today or later is the short forecast (today/tomorrow); with no
dates it defaults to the forecast. Coordinates come from the configured home
location (GARMIN_HOME_LAT/LON) unless you pass both `lat` and `lon` explicitly.
Use weather only to explain or contextualise stored facts (e.g. a slow run on a
windy or hot day). It is never a substitute for a database value: exact answers
must still come from the tables.

Long-term memory: you keep a persistent profile of the user across sessions.
Call get_memory() when personal context may matter. Save stable, useful facts
the user volunteers (goals, preferences, habits, equipment, lifestyle) with
remember_memory(key, value) using short lowercase keys and concise values;
overwrite a key when a fact changes. NEVER store anything already queryable
from the database, and skip ephemeral or one-off details.

When returning an answer, refer to every number with its unit (e.g. "7.6
hours of sleep", "22:35")."""

#: Label placed ahead of a compacted summary when it seeds a new session.
_SUMMARY_LABEL = "This is a compact summary of our previous conversation. Read it as context and continue normally."

#: Instruction used for the ``/new`` command: condense the conversation into a
#: self-contained summary that a fresh session can pick up from.
_COMPACT_INSTRUCTION = """\
Condense the conversation so far into a compact but complete summary that will
seed a new session. Capture the user's questions, anything they stated about
themselves, and every key finding or number that came up (always with its unit).
Drop tool-call details, intermediate reasoning and redundancy. Do NOT call any
tools. Return ONLY the summary, with no preamble or commentary."""


def _deny(action: int, *args: Any) -> int:
    """Authorizer: deny write opcodes, allow everything else.

    ``return sqlite3.SQLITE_DENY`` aborts the offending statement. The sqlite
    authorizer callback is invoked with 5 positional arguments; only ``action``
    is relevant here.
    """
    blocked = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
    }
    if action in blocked:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _jsonable(value: Any) -> Any:
    """Coerce a SQLite value into a JSON-safe one."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return value


_CHART_TYPES = ("line", "scatter", "bar", "pie", "histogram", "area", "box")


def _chart_spec_error(spec: dict[str, Any], result: dict[str, Any]) -> str | None:
    """Validate a chart spec against an executed query; return an error string
    or None if the spec is usable."""
    columns = result["columns"]
    rows = result["rows"]
    if not rows:
        return "ERROR: the query returned no rows"
    traces = spec.get("traces")
    if not isinstance(traces, list) or not traces:
        return "ERROR: spec needs a non-empty 'traces' list"
    for i, tr in enumerate(traces):
        if not isinstance(tr, dict):
            return f"ERROR: trace {i} must be an object"
        kind = tr.get("type")
        if kind not in _CHART_TYPES:
            return (
                f"ERROR: trace {i} type must be one of: {', '.join(_CHART_TYPES)}"
            )
        for col in (tr.get("x"), tr.get("y")):
            if col is not None and col not in columns:
                return (
                    f"ERROR: trace {i} references column {col!r} which is not in "
                    f"the query result columns {columns}"
                )
        if kind == "histogram":
            if not tr.get("x"):
                return f"ERROR: trace {i} (histogram) needs an 'x' column"
        elif not tr.get("x") or not tr.get("y"):
            return f"ERROR: trace {i} needs both 'x' and 'y' columns"
    layout = spec.get("layout")
    if layout is not None and not isinstance(layout, dict):
        return "ERROR: 'layout' must be a JSON object"
    return None


def _build_chart_figure(
    spec: dict[str, Any], result: dict[str, Any]
) -> Any:
    """Build a Plotly figure from a validated chart spec + query result.

    ``result`` is what ``ReadOnlyDB.run_sql`` returns. Raises ``ValueError``
    with a model-facing message if the spec is unusable.
    """
    import plotly.graph_objects as go

    error = _chart_spec_error(spec, result)
    if error:
        raise ValueError(error)
    columns = result["columns"]
    rows = result["rows"]

    def _data(col: str) -> list[Any]:
        return [r[columns.index(col)] for r in rows]

    traces: list[Any] = []
    for tr in spec["traces"]:
        kind = tr["type"]
        name = tr.get("name")
        if kind == "line":
            traces.append(
                go.Scatter(x=_data(tr["x"]), y=_data(tr["y"]), mode="lines+markers", name=name)
            )
        elif kind == "scatter":
            traces.append(
                go.Scatter(x=_data(tr["x"]), y=_data(tr["y"]), mode="markers", name=name)
            )
        elif kind == "area":
            traces.append(
                go.Scatter(x=_data(tr["x"]), y=_data(tr["y"]), mode="lines", fill="tozeroy", name=name)
            )
        elif kind == "bar":
            traces.append(go.Bar(x=_data(tr["x"]), y=_data(tr["y"]), name=name))
        elif kind == "pie":
            traces.append(go.Pie(labels=_data(tr["x"]), values=_data(tr["y"]), name=name))
        elif kind == "histogram":
            traces.append(go.Histogram(x=_data(tr["x"]), name=name))
        elif kind == "box":
            traces.append(go.Box(x=_data(tr["x"]), y=_data(tr["y"]), name=name))

    fig = go.Figure(data=traces)
    layout = spec.get("layout")
    if layout:
        fig.update_layout(**layout)
    return fig


class ReadOnlyDB:
    """Read-only handle over the Garmin database plus safe query execution.

    Every call opens its own short-lived connection with ``PRAGMA query_only``
    and an authorizer that denies all write opcodes; ``run_sql`` additionally
    guards the statement. Connections are per-call because Pydantic AI runs
    tools from a worker thread (SQLite connections are thread-bound).
    """

    def __init__(self, db_path: str) -> None:
        self.path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_deny)
        return conn

    def tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r["name"] for r in rows if r["name"] in _ALLOWED_TABLES]

    def _forbidden_table_regex(self) -> re.Pattern | None:
        """Regex matching any table the agent must not reference, or None."""
        with self._connect() as conn:
            known = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
        banned = [n for n in known if n not in _ALLOWED_TABLES]
        banned.extend(["sqlite_master", "sqlite_sequence"])
        if not banned:
            return None
        alternatives = "|".join(
            sorted({re.escape(n) for n in banned}, key=len, reverse=True)
        )
        return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)

    def columns(self, table: str) -> list[dict[str, Any]]:
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"unknown table: {table!r}")
        with self._connect() as conn:
            rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [dict(r) for r in rows]

    def date_range(self) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT MIN(calendar_date) AS min_date, MAX(calendar_date) "
                    "AS max_date, COUNT(*) AS n FROM daily_metrics"
                ).fetchone()
        except sqlite3.DatabaseError:
            return {"min": None, "max": None, "rows": 0}
        return {
            "min": row["min_date"],
            "max": row["max_date"],
            "rows": row["n"],
        }

    def run_sql(self, sql: str) -> dict[str, Any]:
        """Validate and execute a read-only query, returning rows as JSON-safe."""
        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise ValueError("empty statement")
        if ";" in statement:
            raise ValueError("only a single statement is allowed")
        if not _SELECT_PREFIX.match(statement):
            raise ValueError(
                "only SELECT / WITH / EXPLAIN / PRAGMA statements are allowed"
            )
        write_word = _WRITE_WORDS.search(statement)
        if write_word:
            raise ValueError(
                f"statement contains a write keyword and was rejected: "
                f"{write_word.group(0)}"
            )
        forbidden = self._forbidden_table_regex()
        if forbidden and forbidden.search(statement):
            raise ValueError(
                "statement references a table outside the allowed set "
                f"({', '.join(_ALLOWED_TABLES)})"
            )
        with self._connect() as conn:
            try:
                cur = conn.execute(statement)
            except sqlite3.DatabaseError as exc:
                raise sqlite3.DatabaseError(
                    f"{exc} | statement: {statement!r}"
                ) from exc
            columns = [d[0] for d in (cur.description or [])]
            rows = [list(row) for row in cur.fetchmany(_MAX_ROWS + 1)]
        truncated = len(rows) > _MAX_ROWS
        return {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows[: _MAX_ROWS]],
            "truncated": truncated,
            "note": (
                f"showing up to {_MAX_ROWS} rows".lower()
                if truncated
                else f"{len(rows) if not truncated else _MAX_ROWS} rows"
            ),
        }

    def close(self) -> None:
        """Connections are per-call; nothing to release."""


class Memory:
    """A tiny persistent key/value store for long-term user facts.

    Persisted as a JSON dict at ``path``. Reads and writes go straight to the
    file per call (like the DB connections) so it is safe from agent worker
    threads; writes are atomic-ish (tmp file then replace). Corruption or a
    missing file degrades to an empty profile.
    """

    _MAX_KEY = 80
    _MAX_VALUE = 2000

    def __init__(self, path: str) -> None:
        self.path = path

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)
        }

    def _write(self, data: dict[str, str]) -> None:
        """Atomic-ish write with a unique tmp + retry, for Windows file locking.

        The destination may be transiently held by another process (antivirus,
        indexer, or a concurrent turn), which makes ``os.replace`` fail with
        ``PermissionError [WinError 32]``. Each write uses its own temp name so
        concurrent writers never clobber each other, and the replace is retried
        briefly before re-raising.
        """
        path = Path(self.path)
        tmp = path.with_name(f"{path.stem}.{secrets.token_hex(6)}.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            for attempt in range(5):
                try:
                    os.replace(tmp, path)
                    return
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def get(self) -> dict[str, str]:
        return self._read()

    def remember(self, key: str, value: str) -> None:
        key, value = key.strip(), value.strip()
        if not key or len(key) > self._MAX_KEY:
            raise ValueError(f"key must be 1..{self._MAX_KEY} characters")
        if len(value) > self._MAX_VALUE:
            raise ValueError(f"value must be at most {self._MAX_VALUE} characters")
        data = self._read()
        data[key] = value
        self._write(data)

    def forget(self, key: str) -> bool:
        data = self._read()
        if key not in data:
            return False
        del data[key]
        self._write(data)
        return True


class Weather:
    """Stateless Open-Meteo access for the agent (archive + short forecast).

    Every call is one small HTTP request; nothing is cached or stored, so the
    tool can never be a source of truth — only context for the stored data.
    Coordinates come from a configured home location and can be overridden per
    request (both ``lat`` and ``lon`` together).
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    MAX_DAYS = 92
    FORECAST_DAYS = 2
    _DAILY_FIELDS = (
        "temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "wind_speed_10m_max"
    )
    _FIELD_ALIASES = {
        "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c",
        "precipitation_sum": "precip_mm",
        "wind_speed_10m_max": "wind_max_kmh",
    }

    def __init__(
        self,
        *,
        default_lat: float | None = None,
        default_lon: float | None = None,
        http_get: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        today: date | None = None,
    ) -> None:
        self._default_lat = default_lat
        self._default_lon = default_lon
        self._http_get = http_get
        self._today = today or date.today()

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._http_get is not None:
            return self._http_get(url, params)
        import httpx

        resp = httpx.get(url, params=params, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def from_config(
        cls, cfg: dict[str, str], **kwargs: Any
    ) -> "Weather":
        """Build from config strings (empty strings become no default)."""
        return cls(
            default_lat=_float_or_none(cfg.get("weather_home_lat")),
            default_lon=_float_or_none(cfg.get("weather_home_lon")),
            **kwargs,
        )

    def _resolve(self, lat: float | None, lon: float | None) -> tuple[float, float]:
        if lat is None and lon is None:
            if self._default_lat is None or self._default_lon is None:
                raise ValueError(
                    "no location configured: set GARMIN_HOME_LAT and "
                    "GARMIN_HOME_LON in .env (or pass explicit lat and lon)"
                )
            return float(self._default_lat), float(self._default_lon)
        if lat is None or lon is None:
            raise ValueError("pass either both lat and lon, or neither")
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError) as exc:
            raise ValueError("lat and lon must be numbers") from exc
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("lat/lon out of range")
        return lat, lon

    def query(
        self,
        date_start: str | None = None,
        date_end: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> dict[str, Any]:
        """Return per-day weather for ``date_start``..``date_end`` (inclusive).

        A range fully before *today* uses the historical archive; a range
        starting today or later is the forecast (forecast days only). With no
        dates the forecast for today/tomorrow is returned. Raises ``ValueError``
        with a model-facing message on invalid input.
        """
        lat, lon = self._resolve(lat, lon)
        today = self._today
        if date_start is None and date_end is None:
            start = today
            end = today + timedelta(days=self.FORECAST_DAYS - 1)
        elif date_start is None or date_end is None:
            raise ValueError(
                "pass either both date_start and date_end, or neither (forecast)"
            )
        else:
            try:
                start = date.fromisoformat(date_start)
                end = date.fromisoformat(date_end)
            except ValueError as exc:
                raise ValueError("date_start/date_end must be YYYY-MM-DD") from exc
        if end < start:
            raise ValueError("date_end must not be before date_start")
        days = (end - start).days + 1
        if days > self.MAX_DAYS:
            raise ValueError(f"request at most {self.MAX_DAYS} days at a time")

        if end < today:
            url = self.ARCHIVE_URL
        elif start >= today:
            forecast_end = today + timedelta(days=self.FORECAST_DAYS - 1)
            if end > forecast_end:
                raise ValueError(
                    f"the forecast only covers up to {forecast_end.isoformat()} "
                    f"({self.FORECAST_DAYS} days)"
                )
            url = self.FORECAST_URL
        else:
            raise ValueError(
                "the range crosses today: pick range wholly before today "
                "(history) or wholly today-and-later (forecast)"
            )

        payload = self._get(
            url,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": self._DAILY_FIELDS,
                "timezone": "auto",
            },
        )
        return self._compact(payload, lat, lon, start, end, url)

    def _compact(
        self,
        payload: dict[str, Any],
        lat: float,
        lon: float,
        start: date,
        end: date,
        url: str,
    ) -> dict[str, Any]:
        daily = payload.get("daily") or {}
        times = daily.get("time") or []
        days: list[dict[str, Any]] = []
        for i, day in enumerate(times):
            row: dict[str, Any] = {"date": day}
            for field, alias in self._FIELD_ALIASES.items():
                series = daily.get(field) or []
                row[alias] = series[i] if i < len(series) else None
            days.append(row)
        return {
            "source": "forecast" if url == self.FORECAST_URL else "historical",
            "location": {"lat": lat, "lon": lon},
            "days": days[: self.MAX_DAYS],
            "note": (
                "Open-Meteo daily values; metric units (degC, mm, km/h). "
                f"Request covered {start.isoformat()}..{end.isoformat()}."
            ),
        }


def _float_or_none(value: str | None) -> float | None:
    """Parse a possibly-empty config string into a float, or None."""
    if value is None or not str(value).strip():
        return None
    return float(value)


def _tool_error(exc: BaseException) -> str:
    """Format a tool exception, avoiding a duplicated ``ERROR: `` prefix."""
    msg = str(exc)
    return msg if msg.startswith("ERROR: ") else f"ERROR: {msg}"


def _schema_text(db: ReadOnlyDB) -> str:
    """Render the agent-facing schema as ``table: col1, col2`` lines."""
    return "\n".join(
        f"{table}: {', '.join(c['name'] for c in db.columns(table))}"
        for table in db.tables()
    )


def build_agent(
    db: ReadOnlyDB,
    *,
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    model: Any = None,
    memory: Memory | None = None,
    weather: "Weather | None" = None,
) -> Any:
    """Build the Pydantic AI agent wired to ``db`` tools.

    Pass ``model`` (e.g. ``TestModel``) to override the transport for tests.
    Pass ``memory`` to give the agent get/remember/forget tools over a durable
    user profile (long-term memory between sessions).
    """
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if model is None:
        model = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )
    system_prompt = _SYSTEM_PROMPT.format(today=date.today().isoformat())
    schema = _schema_text(db)
    if schema:
        system_prompt += (
            "\n\nThe current database schema is given below. You do NOT need to "
            "call table_schema for these tables — the columns are already listed "
            "here. Write queries directly against these columns; if a column you "
            "expect is missing, re-check with table_schema once.\n\n" + schema
        )
    agent = Agent(model, system_prompt=system_prompt)

    @agent.tool_plain
    def list_tables() -> str:
        """Return the names of all tables."""

        return json.dumps(db.tables())

    @agent.tool_plain
    def table_schema(table: str) -> str:
        """Return the columns of a table (names and types)."""
        try:
            return json.dumps(db.columns(table))
        except ValueError as exc:
            return _tool_error(exc)

    @agent.tool_plain
    def date_range() -> str:
        """Return the minimum and maximum calendar_date in daily_metrics."""
        return json.dumps(db.date_range())

    @agent.tool_plain
    def run_sql(sql: str) -> str:
        """Run a read-only SQL query. Returns columns and rows as JSON.

        Only SELECT / WITH / EXPLAIN / PRAGMA statements are allowed. List only
        the columns you need (never SELECT *), aggregate with GROUP BY, and use
        small limits — results are capped at 200 rows.
        """
        try:
            result = db.run_sql(sql)
        except (ValueError, sqlite3.DatabaseError) as exc:
            return _tool_error(exc)
        return json.dumps(result)

    @agent.tool_plain
    def chart(spec: str) -> str:
        """Validate a free-form chart spec and return it ready to embed.

        ``spec`` is a JSON object describing a Plotly figure:
          {
            "sql": "SELECT ...",                     # read-only; must return the data
            "traces": [                            # one or more traces
              {"type": "bar",         # line|scatter|area|bar|pie|histogram|box
               "x": "<result column>",
               "y": "<numeric result column>",     # omit y for histogram
               "name": "optional legend label"}
            ],
            "layout": {"title": {"text": "..."}, ...}   # optional Plotly layout
          }
        This runs ``sql`` to confirm it works and columns exist, then returns the
        same spec (with your title layout) for you to embed VERBATIM in the
        final answer wrapped in <chart> ... </chart> tags. It never returns the
        data itself — the UI reruns the query to draw the chart.
        """
        try:
            import json as _json

            parsed = _json.loads(spec)
        except _json.JSONDecodeError as exc:
            return f"ERROR: spec is not valid JSON: {exc}"
        if not isinstance(parsed, dict):
            return "ERROR: spec must be a JSON object"
        sql = parsed.get("sql")
        if not isinstance(sql, str):
            return "ERROR: spec needs a string 'sql' key"
        try:
            result = db.run_sql(sql)
        except (ValueError, sqlite3.DatabaseError) as exc:
            return f"ERROR: {exc}"
        try:
            _build_chart_figure(parsed, result)
        except ValueError as exc:
            return _tool_error(exc)
        return "OK: " + _json.dumps(parsed, ensure_ascii=False)

    if weather is not None:

        @agent.tool_plain(name="weather")
        def weather_fc(
            lat: float | None = None,
            lon: float | None = None,
            date_start: str | None = None,
            date_end: str | None = None,
        ) -> str:
            """Return daily weather (min/max degC, precip mm, max wind km/h).

            ``date_start``/``date_end`` are inclusive YYYY-MM-DD bounds: a range
            fully before today is historical weather; a range starting today or
            later is the forecast (today/tomorrow only). With neither date given
            the forecast for today/tomorrow is returned. Coordinates default to
            the configured home location (GARMIN_HOME_LAT/GARMIN_HOME_LON);
            override by passing both ``lat`` and ``lon``. Returns JSON of daily
            rows plus the location used. Use it only to contextualise stored
            data — never as the source of an answer.

            Example: weather_fc(date_start="2026-07-01", date_end="2026-07-07")
            returns that week's observed weather around home.
            """
            try:
                return json.dumps(
                    weather.query(date_start, date_end, lat, lon),
                    ensure_ascii=False,
                )
            except ValueError as exc:
                return f"ERROR: {exc}"

    if memory is not None:

        @agent.tool_plain
        def get_memory() -> str:
            """Return the user's long-term memory profile as JSON (facts saved in earlier sessions)."""  # noqa: E501
            return json.dumps(memory.get())

        @agent.tool_plain
        def remember_memory(key: str, value: str) -> str:
            """Save or update one durable fact about the user in long-term memory.

            Use short lowercase keys and concise values. Overwrite an existing
            key with remember_memory when the fact has changed.
            """
            try:
                memory.remember(key, value)
                return "saved"
            except ValueError as exc:
                return f"ERROR: {exc}"

        @agent.tool_plain
        def forget_memory(key: str) -> str:
            """Remove a fact from long-term memory."""
            try:
                return "forgotten" if memory.forget(key) else "ERROR: no such key"
            except OSError as exc:
                return f"ERROR: {exc}"

    return agent


def _build_agent(cfg: dict[str, str], db: ReadOnlyDB) -> Any:
    api_key = cfg["llm_api_key"] or None
    base_url = cfg["llm_base_url"] or None
    if not api_key and not base_url:
        raise RuntimeError(
            "no LLM configured: set OPENAI_API_KEY / LLM_API_KEY for cloud, "
            "or LLM_BASE_URL (e.g. http://host.docker.internal:11434/v1) with LLM_MODEL for a local Ollama model"
        )
    memory = Memory(cfg["memory_file"]) if cfg.get("memory_file") else None
    weather = Weather.from_config(cfg)
    return build_agent(
        db,
        model_name=cfg["llm_model"],
        base_url=base_url,
        api_key=api_key,
        memory=memory,
        weather=weather,
    )


def _record_turn(
    cfg: dict[str, str],
    question: str,
    result: Any,
) -> None:
    """Append a trace entry for this turn (no-op when tracing is disabled)."""
    path = cfg.get("trace_file")
    if not path:
        return
    from .trace import append_turn

    append_turn(
        path,
        question,
        result.new_messages(),
        answer=str(result.output),
        model=cfg.get("llm_model", ""),
        usage=result.usage,
    )


def _ask(cfg: dict[str, str], question: str) -> str:
    db = ReadOnlyDB(cfg["db_path"])
    try:
        result = _build_agent(cfg, db).run_sync(question)
        _record_turn(cfg, question, result)
        return str(result.output)
    finally:
        db.close()


def _load_session(path: str) -> list[Any]:
    """Load a persisted conversation history from ``path`` (or [] if absent)."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    if not Path(path).exists():
        return []
    raw = Path(path).read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return ModelMessagesTypeAdapter.validate_json(raw)


def _save_session(path: str, history: list[Any]) -> None:
    """Persist ``history`` to ``path`` (atomic-ish: write then replace)."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(ModelMessagesTypeAdapter.dump_json(history).decode(), encoding="utf-8")
    tmp.replace(Path(path))


def _seed_history_from_summary(summary: str) -> list[Any]:
    """Wrap a compacted summary as a one-message history for a new session."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    return [
        ModelRequest(
            parts=[UserPromptPart(content=f"{_SUMMARY_LABEL}\n\n{summary}")]
        )
    ]


def _ask_session(cfg: dict[str, str], session_path: str | None = None) -> None:
    """Interactive multi-turn session; history persists across questions.

    With ``session_path`` the conversation is loaded on start and saved to the
    file after every turn, so a later ``garmin-ask --session FILE`` resumes it.

    Two commands reset the context mid-session:

    - ``/clear``  drops all history — the session starts over with no memory
      of what came before (and the persisted file is wiped too).
    - ``/new``    asks the model to fold the conversation into one compact
      summary, then starts a new session seeded with that summary, so context
      survives in compressed form.
    """
    db = ReadOnlyDB(cfg["db_path"])
    try:
        agent = _build_agent(cfg, db)
        history = _load_session(session_path) if session_path else []
        if history:
            print(f"Resumed {len(history)} prior message(s) from {session_path}")
        print(
            "Ask about your Garmin data, one question per line "
            "(exit/quit or EOF to leave; /clear starts fresh; "
            "/new compact the context into a new session)."
        )
        while True:
            try:
                prompt = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not prompt:
                continue
            low = prompt.lower()
            if low in ("exit", "quit"):
                return
            if low == "/clear":
                history = []
                if session_path:
                    _save_session(session_path, history)
                print("Cleared — new session with no prior context.")
                continue
            if low == "/new":
                if not history:
                    history = []
                    if session_path:
                        _save_session(session_path, history)
                    print("Nothing to compact yet — starting a new empty session.")
                    continue
                try:
                    summary = agent.run_sync(
                        _COMPACT_INSTRUCTION, message_history=history or None
                    ).output
                    summary = str(summary).strip()
                except Exception as exc:
                    print(f"error: {exc}")
                    continue
                if not summary:
                    print("Compaction produced no summary; keeping the current context.")
                    continue
                n_before = len(history)
                history = _seed_history_from_summary(summary)
                if session_path:
                    _save_session(session_path, history)
                print(
                    f"Compacted {n_before} message(s) into 1; "
                    "new session seeded with the summary."
                )
                continue
            try:
                result = agent.run_sync(prompt, message_history=history or None)
            except Exception as exc:
                print(f"error: {exc}")
                continue
            _record_turn(cfg, prompt, result)
            history = result.all_messages()
            if session_path:
                _save_session(session_path, history)
            print(result.output)
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        prog="garmin-ask",
        description="Ask questions about your Garmin data (select-only agent). "
        "Run with no QUESTION to start an interactive session that keeps context "
        "(commands: /clear = fresh session with no context, "
        "/new = compact the context into a new session).",
    )
    parser.add_argument(
        "question", nargs="*", metavar="QUESTION",
        help="question to ask; omit it to start an interactive session",
    )
    parser.add_argument(
        "--session", metavar="FILE", dest="session_file",
        help="in the interactive session, persist/resume the conversation "
        "history in FILE (saved after each turn)",
    )
    args = parser.parse_args(argv)
    if args.question and args.session_file:
        parser.error("--session only applies to the interactive session; omit QUESTION")

    cfg = load_config()
    try:
        if args.question:
            print(_ask(cfg, " ".join(args.question)))
        else:
            _ask_session(cfg, args.session_file)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())