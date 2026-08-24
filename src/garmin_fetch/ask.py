"""AI agent that answers questions by querying the per-user Garmin database.

The agent layer is deliberately thin: a ``ReadOnlyDB`` executor that exposes
schema introspection and safe ``SELECT`` queries, wrapped in a Pydantic AI
agent whose tools let the model inspect and query the Postgres store. All data
access is read-only by construction (the agent connects as a SELECT-only PG
role, and Row-Level Security scopes every row to ``current_setting('app.user_id')``)
plus a statement gate as a second layer, so a model can never mutate the DB or
read another account's rows.

The model/provider is swappable: point ``LLM_BASE_URL`` at Ollama (local,
data stays on-machine) or leave it unset to use the OpenAI API
(``OPENAI_API_KEY``).

The agent also has a stateless ``weather`` tool (Open-Meteo archive + short
forecast) to contextualise stored facts — it never writes anything, so the
store stays the sole source of truth.

``garmin-ask`` runs a one-shot query, or (with no question argument) an
interactive session that threads the conversation history through every turn
so the model keeps context across questions. Both the conversation and the
long-term memory profile are persisted per user in Postgres (the ``user_state``
table), so a later ``garmin-ask`` resumes exactly where the last one left off.

The interactive session understands two extra commands:
``/clear`` wipes the context and starts a new session with no prior history;
``/new`` asks the model to collapse the current context into a compact summary
and starts a new session seeded with that summary (so nothing is lost, but the
token footprint shrinks).
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from decimal import Decimal
from argparse import ArgumentParser, Namespace
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Protocol

import psycopg

from .config import load_config


class _Memory(Protocol):
    """The durable per-user memory interface (implemented by
    ``server.state.PgMemory``)."""

    def get(self) -> dict[str, str]: ...

    def remember(self, key: str, value: str) -> None: ...

    def forget(self, key: str) -> bool: ...


class _Plan(Protocol):
    """The per-user training-plan interface (implemented by
    ``server.state.TrainingPlan``)."""

    def list(
        self,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]: ...


#: Guard a statement is a read-only query and not a write.
_SELECT_PREFIX = re.compile(r"^\s*(?:SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)
_WRITE_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REINDEX|VACUUM|"
    r"REPLACE|TRIGGER)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 200

#: Training-plan tool guardrails: by default ``get_training_plan`` returns only
#: this window (recent past + upcoming) so a long completed history is never
#: pulled into the model context, and caps the rows at the safety limit.
_PLAN_PAST_DAYS = 30
_PLAN_FUTURE_DAYS = 180
_PLAN_MAX_WORKOUTS = 200

#: Tables the agent may see and query. Everything else (raw ``metrics``,
#: ``activities``, ``user_profile``, ``sync_state``) stays invisible to the
#: model — and, at the database level, unreadable by the agent's SELECT-only
#: role (real REVOKEs). This list is the second-layer statement gate.
_ALLOWED_TABLES = (
    "daily_metrics", "activity_summaries", "activity_detail_series",
    "activity_splits", "hr_zones", "power_zones", "race_predictions", "gear",
    "devices", "derived_metrics", "training_plan",
)

#: Data-driven schema annotations: table-level notes and per-column unit/semantic
#: hints. ``_schema_text`` generates the agent-facing schema from these plus the
#: live columns, so a schema change needs at most one dict entry here — never a
#: prose edit that can drift out of sync.
_TABLE_NOTES: dict[str, str] = {
    "daily_metrics": "one row per calendar_date (YYYY-MM-DD)",
    "activity_summaries": "one row per activity",
    "activity_detail_series": (
        "one row per tick (activity_id + tick); prefer aggregate/interval "
        "queries; a metric may be NULL on every tick if the device didn't record it"
    ),
    "activity_splits": "one row per split (activity_id + split_type + split_number)",
    "hr_zones": "configured HR zone boundaries, one row per sport; current snapshot",
    "power_zones": "configured power zones (watts) + FTP, one row per sport; current snapshot",
    "race_predictions": "single current-fitness snapshot",
    "gear": "one row per item; current snapshot (each sync replaces it)",
    "devices": "one row per device; current snapshot",
    "derived_metrics": (
        "one row per (calendar_date, metric); daily derived scores recomputed "
        "and stored each sync. metric = 'readiness' (0-100 score) with "
        "readiness_hrv/readiness_rhr/readiness_sleep (raw values), "
        "readiness_z_hrv/readiness_z_rhr/readiness_z_sleep (z-scores), "
        "readiness_composite, readiness_samples_hrv/rhr/sleep; 'acwr' (ratio) "
        "with acwr_acute_load / acwr_chronic_load / acwr_daily_load. Pivot with "
        "WHERE metric = '<name>' (a day's components share its calendar_date)"
    ),
    "training_plan": "planned workouts; READ-ONLY via SQL — write only through the plan tools",
}

#: Per-column hints for columns whose unit or meaning is non-obvious. Keyed by
#: table, then column; an entry only shows up when that column actually exists.
_COLUMN_DOCS: dict[str, dict[str, str]] = {
    "daily_metrics": {
        "total_distance_m": "METRES (activity_summaries.distance_km is km)",
        "sleep_start_local": "wall-clock HH:MM",
        "sleep_end_local": "wall-clock HH:MM",
        "sleep_score": "0-100",
        "hrv_last_night_avg": "HRV score (ms)",
        "vo2max": "precise VO2max estimate, forward-filled",
        "resting_hr": "bpm; no 7-day avg stored — compute rolling averages yourself",
        "lactate_threshold_hr": "bpm, forward-filled",
        "lactate_threshold_speed_kmh": "km/h, forward-filled",
        "running_ftp_watts": "running FTP (watts), forward-filled",
        "cycling_ftp_watts": "cycling FTP (watts); absent without a power meter",
        "sweat_loss_ml": "estimated sweat loss (ml)",
    },
    "activity_summaries": {
        "activity_type": "Garmin typeKey (e.g. running, road_biking)",
        "distance_km": "km",
        "duration_hours": "hours",
        "elapsed_hours": "hours",
        "moving_hours": "hours",
        "avg_hr": "bpm",
        "max_hr": "bpm",
        "pace_min_km": "running min/km (decimal, 5.5 = 5:30); NULL for non-running",
        "vo2max": "the activity's own VO2max estimate",
        "weather_temp_c": "degC",
        "weather_apparent_c": "degC",
        "weather_humidity": "0-100",
        "weather_wind_kmh": "km/h",
        "weather_description": "e.g. 'Fair'; NULL for indoor/weatherless",
        "is_pr": "personal record flag",
    },
    "activity_detail_series": {
        "distance_m": "cumulative metres",
        "ts_ms": "epoch ms",
        "speed_kmh": "km/h",
        "power_w": "watts",
        "accumulated_power_w": "watts",
    },
    "activity_splits": {
        "distance_m": "metres",
        "duration_s": "seconds",
        "start_time_s": "offset into the activity (seconds)",
        "pace_sec_per_km": "seconds/km (lower = faster)",
        "split_type": "'distance' (work) or 'rest'",
    },
    "race_predictions": {
        "time_5k_min": "minutes",
        "time_10k_min": "minutes",
        "time_half_marathon_min": "minutes",
        "time_marathon_min": "minutes",
    },
    "derived_metrics": {
        "metric": "which derived metric the row holds — see the table note for known names",
        "value": "the metric's value; units differ per metric (readiness 0-100, acwr ratio, loads arbitrary)",
        "qualifier": "category label: readiness = Prime/Moderate/Low/Depleted; acwr = Sweet Spot/Elevated/Danger/Detraining",
    },
    "training_plan": {
        "activity_type": "run/cycle/swim/strength/rest/other",
        "duration_min": "planned minutes",
        "distance_km": "planned km",
        "completed_activity_id": (
            "activity_summaries.activity_id that satisfied it — join for actual stats"
        ),
    },
}

#: The prompt, split into independently-editable sections. ``_PROMPT_ROLE`` is
#: the only one with a ``{today}`` placeholder (formatted at run time as a
#: dynamic system prompt, so the date is always fresh); the rest are plain
#: strings concatenated in order.
_PROMPT_ROLE = """\
You are an analyst with read-only access to a per-user Postgres database of personal
Garmin health data. Today is {today}. Answer the user's questions by inspecting
the schema and running read-only SELECT queries. You may run several queries.
You can never write — except through the dedicated training-plan tools (see below).

The date above is refreshed on every turn. Whenever the answer depends on
"today"/"tomorrow"/"this week" or other relative dates, call the ``today`` tool
and use its result — it is the authoritative current date.

Column suffixes are the units: *_kmh = km/h, *_m = metres, *_w = watts, *_hours
= hours, *_min = minutes, *_c = degC, *_ml = ml, *_kcal = kcal. *_pct = percent
(zone *_pct = % of activity duration spent in that zone). Heart rate is bpm,
cadence is spm, body battery and stress are 0-100. Columns with non-obvious
units or special meaning carry a parenthetical note in the schema below.
"""

_PROMPT_QUERY = """\
Almost every column may be NULL on days/rows where the value is N/A; only use
aggregate functions over the rows you have. Result sets are capped at 200 rows,
so prefer aggregated queries (GROUP BY weeks/months/weekdays) over dumping raw
rows. Never use SELECT *; list only the columns you need (daily_metrics is ~60
columns wide and activity_summaries ~58). Keep result sets tiny: select a handful
of columns and aggregate to at most ~100-200 rows, ORDER BY time columns so
trends read naturally. Prefer small probes — e.g. query with LIMIT 10 to
sanity-check columns before a full aggregation.

Write the SQL yourself for every data question. There are no ready-made query
tools and no shortcuts: use the schema given above and the ``run_sql`` tool for
all data access (metric time series, recent activities, splits, per-tick series,
per-day summaries — everything). Derive each answer directly from the tables.

One exception: for "how was my day X" questions, call ``get_day_summary(YYYY-MM-DD)``
instead of writing a SELECT — it returns the whole per-day bundle (every stored
daily metric plus that day's activities) in one call without a ~60-column
daily_metrics dump. For anything else, write the SQL yourself.
"""

_PROMPT_CHARTS = """\
When asked for a chart or visualization: call the chart tool to validate a
spec. The spec is a JSON object of your OWN design (you write the graph, not
the data): {"sql": "SELECT ...", "traces": [{...}], "layout": {"title":
{"text": "..."}}}. The `sql` must return the plotted data itself (column
names in the traces reference its result). You are free to draw ANY Plotly
figure: each trace names a plotly.graph_objects class via "go" (e.g. "Scatter",
"Scattergl", "Violin", "Heatmap", "Pie", "Bar", "Candlestick") — the compact
aliases line / scatter / area / bar / pie / histogram / box also work. Data
columns are referenced by name in x / y / z, and any other numeric trace
argument can reference a column as {"column": "<name>"} (e.g. for a pie:
{"labels": {"column": "sport_type"}, "values": {"column": "hours"}}).
Every remaining key in a trace is passed straight to the Plotly constructor
(mode, marker, line, opacity, orientation, colorscale, text, ...). Keep points
to at most ~200 by aggregating (GROUP BY week/month/weekday) and ORDER BY time
columns. The chart tool runs the query, checks every referenced column exists
and the trace can be built, and returns "OK: <spec> (query returned N rows)" if
valid. Embed only the spec JSON — the object after "OK: " — VERBATIM in the
final answer wrapped in <chart> ... </chart> (one per chart) and add a short
one-sentence description of what it shows as normal text. Do NOT paste query
data or Python code into your answer — only the spec.
"""

_PROMPT_WEATHER = """\
Weather: observed conditions during a workout are already stored per activity
in activity_summaries (weather_temp_c, weather_apparent_c, weather_humidity,
weather_wind_kmh, weather_description) — prefer those columns whenever a
question is about the weather at the time/location of a stored activity. The
`weather` tool is for what the stored data does NOT cover: (1) the short forecast
(up to 16 days ahead), (2) historical weather for days with no activity (e.g.
correlate a restless night with the temperature), and (3) weather at any location
via explicit `lat`/`lon` (default: GARMIN_HOME_LAT/GARMIN_HOME_LON). It returns
per-day min/max temp (degC), precipitation (mm) and max wind (km/h) for a range
fully before today (historical) or starting today (forecast). It is never a
substitute for a database value: exact answers must still come from the tables.
"""

_PROMPT_MEMORY = """\
Long-term memory: you keep a persistent profile of the user across sessions.
Call get_memory() when personal context may matter. Save with
remember_memory(key, value) — short lowercase keys, concise values — only
facts the user volunteers that are not in the database (goals, preferences,
habits, equipment, lifestyle). NEVER save anything you could query from the
database, including anything inferred from it: no fitness markers (VO2max,
FTP, LTHR), no personal records, no race predictions, no training zones, no
baseline resting HR/HRV — those all belong in the tables and are re-derived
on demand. Memory exists solely to hold what the database cannot tell you.
Overwrite a key when a fact changes, and skip ephemeral or one-off details.
"""

_PROMPT_PLAN = """\
Training plan: the user keeps a training plan (a set of planned workouts) shown
on the site and editable both manually and through you. Read it with
get_training_plan() and update it with update_training_plan(). When the user
asks to create or adjust a training plan (e.g. "build a 12-week half-marathon
plan starting Monday", "move my Thursday run to Friday", "add an easy swim on
Wednesday"), plan the workouts yourself — dates (YYYY-MM-DD), activity_type
(run/cycle/swim/strength/rest/other), title, duration_min, distance_km,
intensity (easy/moderate/hard/race_pace) — and call update_training_plan with
the complete set. Pass "replace": true when the user wants a brand-new plan
(so stale workouts are cleared first). Use the schema and existing data (e.g.
recent weekly mileage, race predictions, VO2max, hr_zones) to make the plan
sensible and specific to the user. After updating, tell the user what you
added/changed (a concise weekly summary is ideal).

When the user wants to SEE the plan (or a slice of it) in the chat, call
get_training_plan() to inspect it, then do NOT transcribe the workouts into a
markdown table and NEVER paste the raw JSON the tool returns. Instead embed
this exact marker in your answer: <plan_table /> — optionally with date
bounds, e.g. <plan_table from="2026-08-10" to="2026-08-16" />. The UI replaces
the marker with the user's stored plan rendered directly from the database
(identical to the Training Plan tab), so you never burn tokens writing the
table out. Keep your narrative answer short — the marker renders the table
for you.
"""

_PROMPT_ANSWER = """\
When returning an answer, refer to every number with its unit (e.g. "7.6
hours of sleep", "22:35").
"""

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


def _jsonable(value: Any) -> Any:
    """Coerce a driver value into a JSON-safe one."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, Decimal):
        # psycopg returns ``numeric`` (e.g. EXTRACT(...)) as Decimal, which the
        # json module can't serialize; collapse to float (NaN -> null).
        if value != value:
            return None
        try:
            return float(value)
        except (OverflowError, ValueError):
            return str(value)
    if isinstance(value, (datetime, date, time)):
        # psycopg returns ``date``/``timestamp``/``time`` as their Python
        # counterparts, which json can't serialize; emit ISO-8601 strings.
        return value.isoformat()
    return value


def _jsonify_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce every value in a list of dict rows to a JSON-safe form."""
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]


_CHART_TYPES = (
    "line", "scatter", "area", "bar", "pie", "histogram", "box",
)

#: Legacy trace aliases mapped to plotly.graph_objects classes, with the default
#: styling the old hand-built builder chose. New specs may pass any Plotly trace
#: class via the "go" key; these aliases just keep the compact shorthand working.
_LEGACY_TRACES: dict[str, dict[str, Any]] = {
    "line": {"go": "Scatter", "defaults": {"mode": "lines+markers"}},
    "scatter": {"go": "Scatter", "defaults": {"mode": "markers"}},
    "area": {"go": "Scatter", "defaults": {"mode": "lines", "fill": "tozeroy"}},
    "bar": {"go": "Bar"},
    "pie": {"go": "Pie"},
    "histogram": {"go": "Histogram"},
    "box": {"go": "Box"},
}


def _go_class_name(go: Any, name: str) -> str | None:
    """Return a valid plotly.graph_objects trace class name for ``name`` or None.

    Plotly's trace classes are CamelCase and lazily exposed as attributes, so
    accept the exact name or a lowercase variant (``scatter`` -> ``Scatter``).
    """
    for candidate in (name, name[:1].upper() + name[1:]):
        cls = getattr(go, candidate, None)
        if isinstance(cls, type):
            return candidate
    return None


def _trace_column_refs(trace: dict[str, Any]) -> list[str]:
    """Collect the result columns a trace references.

    ``x``/``y``/``z`` set to a column name are references; every nested value
    written as ``{"column": "col"}`` is one too (how data reaches arbitrary
    trace params, e.g. ``{"labels": {"column": "sport"}}``).
    """
    refs: list[str] = []
    for key in ("x", "y", "z"):
        value = trace.get(key)
        if isinstance(value, str):
            refs.append(value)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {"column"} and isinstance(value.get("column"), str):
                refs.append(value["column"])
            else:
                for child in value.values():
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for value in trace.values():
        walk(value)
    return refs


def _chart_spec_error(spec: dict[str, Any], result: dict[str, Any]) -> str | None:
    """Validate a chart spec against an executed result; return an error string
    or None if the spec is usable."""
    import plotly.graph_objects as go

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
        name = tr.get("go") if isinstance(tr.get("go"), str) else tr.get("type")
        if not isinstance(name, str) or not name:
            return f"ERROR: trace {i} needs a 'go' or a 'type' name"
        if name not in _LEGACY_TRACES and _go_class_name(go, name) is None:
            return (
                f"ERROR: trace {i} class {name!r} is not a valid chart type. "
                f"Use one of: {', '.join(_CHART_TYPES)} "
                "or a plotly.graph_objects class name "
                "(e.g. Scatter, Scattergl, Violin, Heatmap, Pie)"
            )
        for col in _trace_column_refs(tr):
            if col not in columns:
                return (
                    f"ERROR: trace {i} references column {col!r} which is not in "
                    f"the query result columns {columns}"
                )
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

    def _resolve(value: Any) -> Any:
        """Replace data references with their column values, pass rest through.

        A dict of the exact shape ``{"column": "name"}`` becomes that column's
        data; everything else (raw strings, numbers, marker/line/color config
        dicts, arrays) is passed verbatim to the Plotly constructor.
        """
        if isinstance(value, dict):
            if set(value) == {"column"} and isinstance(value.get("column"), str):
                return _data(value["column"])
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        return value

    traces: list[Any] = []
    for tr in spec["traces"]:
        if isinstance(tr.get("go"), str):
            legacy = None
            go_name = tr["go"]
        else:
            name = tr.get("type")
            legacy = _LEGACY_TRACES.get(name) if isinstance(name, str) else None
            go_name = name if legacy is None else legacy["go"]
        kwargs: dict[str, Any] = {}
        for key, value in tr.items():
            if key in ("type", "go"):
                continue
            if key in ("x", "y", "z") and isinstance(value, str) and value in columns:
                kwargs[key] = _data(value)
            else:
                kwargs[key] = _resolve(value)
        if legacy is not None:
            for key, value in legacy.get("defaults", {}).items():
                kwargs.setdefault(key, value)
            if legacy["go"] == "Pie":
                if "labels" not in kwargs and "x" in kwargs:
                    kwargs["labels"] = kwargs.pop("x")
                if "values" not in kwargs and "y" in kwargs:
                    kwargs["values"] = kwargs.pop("y")
        cls = getattr(go, go_name, None) or getattr(go, _go_class_name(go, go_name))
        traces.append(cls(**kwargs))

    fig = go.Figure(data=traces)
    layout = spec.get("layout")
    if layout:
        fig.update_layout(**layout)
    return fig


class QueryError(Exception):
    """A read-only query was rejected by the Postgres driver."""


#: Driver-level errors raised by the read-only PG role (psycopg is a hard
#: dependency; there is no SQLite fallback).
_DB_ERROR_TYPES: tuple[type[BaseException], ...] = (psycopg.Error,)


class _PgReadOnlyBackend:
    """Postgres read-only backend: pooled connections, ``information_schema``.

    Read-only enforcement comes from the read-only PG role (+ Row-Level
    Security, Phase 3): the role has SELECT on the five agent tables and
    nothing else, and RLS filters every row by ``current_setting('app.user_id')``.
    The agent's statement gate stays on top as a second layer. Per-call
    connections are drawn from a shared ``psycopg_pool`` so concurrent chat
    requests don't each open a fresh TCP connection.
    """

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None

    def _get_pool(self) -> Any:
        if self._pool is None:
            import psycopg
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self.dsn,
                min_size=1,
                max_size=8,
                open=True,
                configure=lambda conn: setattr(
                    conn, "row_factory", psycopg.rows.dict_row
                ),
            )
        return self._pool

    def connect(self) -> Any:
        """A pooled connection as a context manager (returns to pool on exit)."""
        return self._get_pool().connection()

    def table_names(self, conn: Any) -> list[str]:
        return [
            r["name"]
            for r in conn.execute(
                "SELECT table_name AS name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            ).fetchall()
        ]

    def columns(self, conn: Any, table: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT column_name AS name, data_type AS type "
                "FROM information_schema.columns WHERE table_name = %s "
                "ORDER BY ordinal_position",
                (table,),
            ).fetchall()
        ]

    def row_values(self, row: Any) -> list[Any]:
        return list(row.values())

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


class ReadOnlyDB:
    """Read-only handle over one account's Garmin data plus safe query execution.

    Bound to a ``user_id``: every call opens its own short-lived pooled
    connection, runs ``SET LOCAL app.user_id = <uid>`` so Row-Level Security
    scopes the transaction, and executes the query as the read-only PG role.
    Connections are per-call because Pydantic AI runs tools from a worker
    thread.
    """

    def __init__(self, url: str, *, user_id: int | None = None) -> None:
        self.path = url
        self.user_id = user_id
        self._backend = _PgReadOnlyBackend(url)

    @classmethod
    def from_url(cls, url: str, *, user_id: int | None = None) -> "ReadOnlyDB":
        """A Postgres-backed read-only handle from a ``postgres://`` DSN."""
        return cls(url, user_id=user_id)

    @contextmanager
    def _connect(self) -> Any:
        with self._backend.connect() as conn:
            if self.user_id is not None:
                conn.execute(
                    "SELECT set_config('app.user_id', %s, true)",
                    (str(self.user_id),),
                )
            yield conn

    def tables(self) -> list[str]:
        with self._connect() as conn:
            names = self._backend.table_names(conn)
        return [n for n in names if n in _ALLOWED_TABLES]

    def _forbidden_table_regex(self) -> re.Pattern | None:
        """Regex matching any table the agent must not reference, or None."""
        with self._connect() as conn:
            known = self._backend.table_names(conn)
        banned = [n for n in known if n not in _ALLOWED_TABLES]
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
            return self._backend.columns(conn, table)

    def date_range(self) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT MIN(calendar_date) AS min_date, MAX(calendar_date) "
                    "AS max_date, COUNT(*) AS n FROM daily_metrics"
                ).fetchone()
        except _DB_ERROR_TYPES:
            return {"min": None, "max": None, "rows": 0}
        return {
            "min": row["min_date"],
            "max": row["max_date"],
            "rows": row["n"],
        }

    def day_summary(self, calendar_date: str) -> dict[str, Any]:
        """Every stored daily metric plus activities for one calendar date."""
        if not isinstance(calendar_date, str) or not calendar_date.strip():
            raise ValueError("calendar_date must be YYYY-MM-DD")
        day = calendar_date.strip()
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError(
                f"calendar_date must be YYYY-MM-DD, got {calendar_date!r}"
            ) from exc
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM daily_metrics WHERE calendar_date = %s",
                (day,),
            ).fetchone()
            activities = conn.execute(
                "SELECT activity_id, activity_name, activity_type, "
                "start_time_local, duration_hours, distance_km, avg_hr, "
                "max_hr, pace_min_km, training_load "
                "FROM activity_summaries WHERE start_date = %s "
                "ORDER BY start_time_local DESC NULLS LAST",
                (day,),
            ).fetchall()
        metrics: dict[str, Any] = {}
        if row:
            metrics = {
                k: _jsonable(v)
                for k, v in dict(row).items()
                if v is not None and k not in ("user_id", "calendar_date", "fetched_at")
            }
        return {
            "calendar_date": day,
            "metrics": metrics,
            "activities": _jsonify_rows([dict(r) for r in activities]),
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
                "only SELECT / WITH / EXPLAIN statements are allowed"
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
            except _DB_ERROR_TYPES as exc:
                raise QueryError(
                    f"{exc} | statement: {statement!r}"
                ) from exc
            columns = [d[0] for d in (cur.description or [])]
            rows = [
                self._backend.row_values(row) for row in cur.fetchmany(_MAX_ROWS + 1)
            ]
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
        """Release the pooled connections (no-op if never opened)."""
        self._backend.close()


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
    FORECAST_DAYS = 16
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
        starting today or later is the forecast (up to ``FORECAST_DAYS`` days
        ahead). With no dates the forecast from today onward is returned. Raises
        ``ValueError`` with a model-facing message on invalid input.
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

        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "daily": self._DAILY_FIELDS,
            "timezone": "auto",
        }
        if url == self.ARCHIVE_URL:
            params["start_date"] = start.isoformat()
            params["end_date"] = end.isoformat()
        else:
            # Open-Meteo only returns `forecast_days` (default 7) unless asked,
            # and `forecast_days` is mutually exclusive with start/end dates.
            # Request from today and trim to the requested range in _compact.
            params["forecast_days"] = (end - today).days + 1
        payload = self._get(url, params)
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
            if start <= date.fromisoformat(day) <= end:
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
    """Render the agent-facing schema, annotating tables and tricky columns.

    Table notes come from ``_TABLE_NOTES`` and per-column unit/semantic hints
    from ``_COLUMN_DOCS``, so the prompt can never drift from the real columns:
    a new column simply appears (with a hint only if one is defined here).
    """
    lines: list[str] = []
    for table in db.tables():
        note = _TABLE_NOTES.get(table)
        header = f"{table}: " + (f"[{note}] " if note else "")
        parts: list[str] = []
        for col in db.columns(table):
            name = col["name"]
            hint = _COLUMN_DOCS.get(table, {}).get(name)
            parts.append(name if hint is None else f"{name} ({hint})")
        lines.append(header + ", ".join(parts))
    return "\n".join(lines)


def _current_date_prompt() -> str:
    """The role prompt with today's date, re-evaluated as a dynamic prompt.

    Includes the weekday name (e.g. ``2026-08-16 (Sunday)``) so the model never
    has to compute the day of week from the date itself — models reliably get
    that arithmetic wrong by a day.
    """
    return _PROMPT_ROLE.format(
        today=date.today().strftime("%Y-%m-%d (%A)")
    )


def _refresh_resumed_prompt(messages: list[Any]) -> list[Any]:
    """Stamp the current date onto any stored system prompt in resumed history.

    Sessions persisted before the date prompt became dynamic carry a
    ``SystemPromptPart`` with an old date and no ``dynamic_ref``, so Pydantic AI
    can never re-evaluate it — the resumed agent keeps thinking "today" is the
    day the session started. Rewrite those parts in place with the fresh date
    and the current ``dynamic_ref`` so they keep refreshing on later turns.
    """
    from pydantic_ai.messages import SystemPromptPart

    fresh = _current_date_prompt()
    ref = _current_date_prompt.__qualname__
    for message in messages:
        for part in getattr(message, "parts", []):
            if not isinstance(part, SystemPromptPart):
                continue
            if part.content == fresh and part.dynamic_ref == ref:
                continue
            part.content = fresh
            part.dynamic_ref = ref
    return messages


def build_agent(
    db: ReadOnlyDB,
    *,
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    model: Any = None,
    memory: _Memory | None = None,
    weather: "Weather | None" = None,
    plan: _Plan | None = None,
) -> Any:
    """Build the Pydantic AI agent wired to ``db`` tools.

    Pass ``model`` (e.g. ``TestModel``) to override the transport for tests.
    Pass ``reasoning_effort`` to request a reasoning effort level from the
    underlying model (``low``/``medium``/``high``). Pass ``memory`` to give
    the agent get/remember/forget tools over a durable user profile (long-term
    memory between sessions). Pass ``plan`` (a ``server.state.TrainingPlan``)
    to give the agent read/update tools over the user's training plan.
    """
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if model is None:
        model_settings = (
            {"openai_reasoning_effort": reasoning_effort}
            if reasoning_effort
            else None
        )
        model = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            settings=model_settings,
        )
    schema = _schema_text(db)
    system_prompt = (
        "The current database schema is given below. You do NOT need to "
        "call table_schema for these tables — the columns (with units/notes) "
        "are already listed here. Write queries directly against these "
        "columns; if a column you expect is missing, re-check with "
        "table_schema once.\n\n" + schema
    ) if schema else ""
    system_prompt += "\n\n" + "\n\n".join(
        [
            _PROMPT_QUERY,
            _PROMPT_CHARTS,
            _PROMPT_WEATHER,
            _PROMPT_MEMORY,
            _PROMPT_PLAN,
            _PROMPT_ANSWER,
        ]
    )
    agent = Agent(model, system_prompt=system_prompt.strip())

    # The date must be a dynamic system prompt: static prompts are evaluated
    # once and stored in the message history, so a cached agent (or a resumed
    # session) would keep "Today is <yesterday>" forever. A dynamic prompt is
    # re-evaluated on every model request, so the date can never go stale.
    # (dynamic=True only works in the decorator form; the module-level function
    # keeps the ``dynamic_ref`` stable so resumed sessions can match it.)
    agent.system_prompt(dynamic=True)(_current_date_prompt)

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
    def today() -> str:
        """Return the current date as YYYY-MM-DD (Weekday).

        Use this (not the date in the system prompt) for any relative-date
        question — the prompt's date can be stale in a long-running session.
        The weekday name is included so you never have to derive it from the
        date yourself.
        """
        return date.today().strftime("%Y-%m-%d (%A)")

    @agent.tool_plain
    def run_sql(sql: str) -> str:
        """Run a read-only SQL query. Returns columns and rows as JSON.

        Only SELECT / WITH / EXPLAIN statements are allowed. List only
        the columns you need (never SELECT *), aggregate with GROUP BY, and use
        small limits — results are capped at 200 rows.
        """
        try:
            result = db.run_sql(sql)
        except (ValueError, QueryError) as exc:
            return _tool_error(exc)
        return json.dumps(result)

    @agent.tool_plain
    def get_day_summary(calendar_date: str) -> str:
        """Return every stored daily metric plus activities for one date.

        ``calendar_date`` is YYYY-MM-DD. Returns the day's non-null
        daily_metrics values (sleep, resting HR, HRV, steps, weight, ...) and
        the activities recorded that day. Use this instead of hand-writing a
        SELECT for "how was my day X" — it returns exactly the per-day bundle
        you need without selecting the ~60-column wide daily_metrics row.
        """
        try:
            return json.dumps(db.day_summary(calendar_date), ensure_ascii=False)
        except (ValueError, QueryError, psycopg.Error) as exc:
            return _tool_error(exc)

    @agent.tool_plain
    def chart(spec: str) -> str:
        """Validate a free-form chart spec and return it ready to embed.

        ``spec`` is a JSON object describing a Plotly figure:
          {
            "sql": "SELECT ...",              # read-only; must return the data
            "traces": [                      # one or more traces
              {
                "go": "Scatter",   # any plotly.graph_objects class name
                                   # ("Scattergl", "Violin", "Heatmap", "Pie",
                                   #  "Bar", "Candlestick", ...) or the compact
                                   #  alias "type": line|scatter|area|bar|pie|
                                   #  histogram|box
                "x": "<result column>",      # column: use x / y / z by name
                "y": "<numeric result column>",
                "mode": "markers",           # any other key is passed straight
                "marker": {"color": "red"}   # to the Plotly constructor
              }
            ],
            "layout": {"title": {"text": "..."}, ...}   # optional Plotly layout
          }
        Any numeric trace argument may also reference a column explicitly as
        {"column": "<name>"} (e.g. pie labels/values). This runs ``sql`` to
        confirm it works and that every referenced column exists, builds the
        figure to check the trace is constructible, then returns the same spec
        (with your title layout) for you to embed VERBATIM in the final answer
        wrapped in <chart> ... </chart> tags. It never returns the data
        itself — the UI reruns the query to draw the chart.
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
        except (ValueError, QueryError) as exc:
            return f"ERROR: {exc}"
        try:
            _build_chart_figure(parsed, result)
        except ValueError as exc:
            return _tool_error(exc)
        rows = result.get("rows") or []
        return (
            "OK: " + _json.dumps(parsed, ensure_ascii=False)
            + f" (query returned {len(rows)} rows)"
        )

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
            later is the forecast (up to 16 days ahead). With neither date given
            the forecast from today onward is returned. Coordinates default to
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

    if plan is not None:

        @agent.tool_plain
        def get_training_plan(
            date_start: str | None = None,
            date_end: str | None = None,
        ) -> str:
            """Return the user's training plan as JSON (recent + upcoming).

            ``date_start``/``date_end`` are optional inclusive YYYY-MM-DD
            bounds. With neither given, only the last 30 days plus the next
            180 days are returned — older completed history is NOT included;
            pass explicit dates to page further back. With both given, that
            exact range is returned. Results are capped at 200 workouts (the
            ``truncated`` flag is set when rows were dropped, so pass a
            narrower range to see the rest). Each workout has id, planned_date,
            activity_type (run/cycle/swim/strength/rest/other), title,
            description, duration_min, distance_km, intensity, completed, and
            completed_activity_id (the activity_summaries.activity_id that
            satisfied it, when auto-detected — join on it for the actual
            workout's stats).
            """
            try:
                if (date_start is None) != (date_end is None):
                    return "ERROR: pass both date_start and date_end, or neither"
                if date_start is None:
                    date_start = (
                        date.today() - timedelta(days=_PLAN_PAST_DAYS)
                    ).isoformat()
                    date_end = (
                        date.today() + timedelta(days=_PLAN_FUTURE_DAYS)
                    ).isoformat()
                else:
                    date.fromisoformat(date_start)
                    date.fromisoformat(date_end)
                workouts = plan.list(date_start, date_end)
                truncated = len(workouts) > _PLAN_MAX_WORKOUTS
                if truncated:
                    workouts = workouts[:_PLAN_MAX_WORKOUTS]
                return json.dumps(
                    {"workouts": workouts, "truncated": truncated},
                    ensure_ascii=False,
                )
            except ValueError as exc:
                return _tool_error(exc)

        @agent.tool_plain
        def update_training_plan(spec: str) -> str:
            """Add, edit or delete workouts in the user's training plan.

            ``spec`` is a JSON object:
            - "replace": optional bool — when true, the whole plan is wiped
              first (use for "create a brand-new plan").
            - "workouts": optional list of workout objects. An object with an
              "id" updates that existing workout (the full desired state must
              be given); one without an "id" creates a new workout. Fields:
              planned_date (required, YYYY-MM-DD), activity_type (required:
              run/cycle/swim/strength/rest/other), title, description,
              duration_min, distance_km, intensity (easy/moderate/hard/
              race_pace), completed (bool).
            - "delete_ids": optional list of workout ids to delete.
            Returns a JSON summary of what changed.
            """
            try:
                parsed = json.loads(spec)
            except json.JSONDecodeError as exc:
                return f"ERROR: spec is not valid JSON: {exc}"
            if not isinstance(parsed, dict):
                return "ERROR: spec must be a JSON object"
            if "replace" in parsed and not isinstance(parsed["replace"], bool):
                return "ERROR: 'replace' must be a boolean"
            for key in ("workouts", "delete_ids"):
                if key in parsed and not isinstance(parsed[key], list):
                    return f"ERROR: '{key}' must be a list"
            try:
                result = plan.apply(parsed)
            except ValueError as exc:
                return _tool_error(exc)
            return json.dumps(result, ensure_ascii=False)

    return agent


def _build_agent(
    cfg: dict[str, str],
    db: ReadOnlyDB,
    *,
    memory: Any | None = None,
    plan: Any | None = None,
) -> Any:
    api_key = cfg["llm_api_key"] or None
    base_url = cfg["llm_base_url"] or None
    if not api_key and not base_url:
        raise RuntimeError(
            "no LLM configured: set OPENAI_API_KEY / LLM_API_KEY for cloud, "
            "or LLM_BASE_URL (e.g. http://host.docker.internal:11434/v1) with LLM_MODEL for a local Ollama model"
        )
    weather = Weather.from_config(cfg)
    return build_agent(
        db,
        model_name=cfg["llm_model"],
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=cfg.get("llm_reasoning_effort") or None,
        memory=memory,
        weather=weather,
        plan=plan,
    )


def _open_readonly(cfg: dict[str, Any]) -> ReadOnlyDB:
    """Read-only handle for the configured backend (read-only PG role)."""
    url = cfg.get("readonly_db_url") or cfg.get("db_url")
    if not url:
        raise RuntimeError(
            "GARMIN_DB_URL (or GARMIN_READONLY_DB_URL) must be set — "
            "Postgres is the only supported backend"
        )
    return ReadOnlyDB.from_url(url, user_id=cfg.get("local_user_id") or 1)


def _record_turn(
    cfg: dict[str, str],
    question: str,
    result: Any,
    *,
    trace_writer: Any,
) -> None:
    """Append a trace entry for this turn via ``trace_writer`` (which persists
    the record to the per-user ``user_state`` ``trace`` key)."""
    from .trace import build_trace_record

    record = build_trace_record(
        question,
        result.new_messages(),
        answer=str(result.output),
        model=cfg.get("llm_model", ""),
        usage=result.usage,
    )
    trace_writer(record)


def _ask(cfg: dict[str, str], question: str) -> str:
    db = _open_readonly(cfg)
    from .server.state import PgMemory, TrainingPlan, TrainingPlanStore, UserState

    state = UserState(cfg["db_url"])
    plan_store = TrainingPlanStore(cfg["db_url"])
    user_id = cfg.get("local_user_id") or 1
    try:
        result = _build_agent(
            cfg,
            db,
            memory=PgMemory(state, user_id),
            plan=TrainingPlan(plan_store, user_id),
        ).run_sync(question)
        _record_turn(
            cfg, question, result, trace_writer=lambda r: state.append_trace(user_id, r)
        )
        return str(result.output)
    finally:
        db.close()
        state.close()
        plan_store.close()


def _seed_history_from_summary(summary: str) -> list[Any]:
    """Wrap a compacted summary as a one-message history for a new session."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    return [
        ModelRequest(
            parts=[UserPromptPart(content=f"{_SUMMARY_LABEL}\n\n{summary}")]
        )
    ]


def _ask_session(cfg: dict[str, str]) -> None:
    """Interactive multi-turn session; history persists across questions.

    Both the conversation and the long-term memory profile live in Postgres
    (``user_state``, scoped to ``GARMIN_LOCAL_USER_ID``), so a later
    ``garmin-ask`` resumes exactly where the last session left off.

    Two commands reset the context mid-session:

    - ``/clear``  drops all history — the session starts over with no memory
      of what came before (and the stored conversation is wiped too).
    - ``/new``    asks the model to fold the conversation into one compact
      summary, then starts a new session seeded with that summary, so context
      survives in compressed form.
    """
    from .server.state import PgMemory, TrainingPlan, TrainingPlanStore, UserState

    db = _open_readonly(cfg)
    state = UserState(cfg["db_url"])
    plan_store = TrainingPlanStore(cfg["db_url"])
    user_id = cfg.get("local_user_id") or 1
    try:
        agent = _build_agent(
            cfg,
            db,
            memory=PgMemory(state, user_id),
            plan=TrainingPlan(plan_store, user_id),
        )
        history = state.get_session_messages(user_id) or []
        if history:
            _refresh_resumed_prompt(history)
            print(
                f"Resumed {len(history)} prior message(s) from the database "
                f"(user {user_id})"
            )

        def _persist(messages: list[Any]) -> None:
            state.set_session_messages(user_id, messages)

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
                state.clear_session(user_id)
                print("Cleared — new session with no prior context.")
                continue
            if low == "/new":
                if not history:
                    history = []
                    state.clear_session(user_id)
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
                _persist(history)
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
            _record_turn(
                cfg,
                prompt,
                result,
                trace_writer=lambda r: state.append_trace(user_id, r),
            )
            history = result.all_messages()
            _persist(history)
            print(result.output)
    finally:
        db.close()
        state.close()
        plan_store.close()


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
    args = parser.parse_args(argv)

    cfg = load_config()
    try:
        if args.question:
            print(_ask(cfg, " ".join(args.question)))
        else:
            _ask_session(cfg)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())