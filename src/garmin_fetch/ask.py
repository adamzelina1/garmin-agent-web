"""AI agent that answers questions by querying the per-user Garmin database.

The agent layer is deliberately thin: a ``ReadOnlyDB`` executor that exposes
schema introspection and safe ``SELECT`` queries, wrapped in a Pydantic AI
agent whose tools let the model inspect and query the Postgres store. All data
access is read-only by construction (the agent connects as a SELECT-only PG
role, and Row-Level Security scopes every row to ``current_setting('app.user_id')``)
plus a statement gate as a second layer, so a model can never mutate the DB or
read another account's rows.

Prompting uses Pydantic AI *instructions*, not system prompts: instructions are
re-sent on every run and never stored in the message history, so a resumed or
compacted session always gets the current role, date, memory and principles
with no re-stamping of persisted history.

The model/provider is swappable: point ``LLM_BASE_URL`` at Ollama (local,
data stays on-machine) or leave it unset to use the OpenAI API
(``OPENAI_API_KEY``).

``garmin-ask`` runs a one-shot query, or (with no question argument) an
interactive session that threads the conversation history through every turn.
The conversation and the long-term memory profile are persisted per user in
Postgres (the ``user_state`` table), so a later ``garmin-ask`` resumes where
the last one left off. In the interactive session ``/clear`` wipes the context
and ``/new`` folds it into a compact summary that seeds a new session.
"""

from __future__ import annotations

import functools
import json
import math
import re
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Callable, Iterable, Protocol

import psycopg
import psycopg.rows

from .config import load_config
from .parser import TYPE_COLUMNS


class _Memory(Protocol):
    """The durable per-user memory interface (implemented by
    ``server.state.PgMemory``)."""

    #: Soft cap on the number of facts, used to nudge consolidation.
    max_facts: int

    def get(self) -> dict[str, str]: ...

    def remember(self, facts: dict[str, Any]) -> int: ...

    def forget(self, keys: list[str]) -> list[str]: ...

    def replace(self, facts: dict[str, Any]) -> int: ...


class _Principles(Protocol):
    """The athlete-authored training principles (implemented by
    ``server.state.PgPrinciples``). Unlike memory, these are directives the
    athlete controls from Settings, not facts the agent records itself."""

    def get(self) -> str: ...


class _Training(Protocol):
    """The per-user training season (implemented by
    ``server.state.TrainingSeason``): one read (the windowed agenda) and one
    atomic write (a season spec, or ``{"undo": true}``)."""

    def agenda(self, **window: Any) -> dict[str, Any]: ...

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]: ...


#: Guard a statement is a read-only query and not a write.
_SELECT_PREFIX = re.compile(r"^\s*(?:SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)
_WRITE_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REINDEX|VACUUM|"
    r"REPLACE|TRIGGER)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 500

#: A ``daily_metrics`` column counts as empty for an account (hidden like a
#: disabled type's columns) when under this share of its days hold a value AND
#: it has fewer than ``_SPARSE_MAX_VALUES`` values — so a handful of monthly
#: weigh-ins still stays visible once enough of them accumulate.
_SPARSE_MAX_SHARE = 0.01
_SPARSE_MAX_VALUES = 10

#: Tables the agent may see and query. Everything else (raw ``metrics``,
#: ``activities``, ``user_profile``, ``sync_state``) stays invisible to the
#: model — and, at the database level, unreadable by the agent's SELECT-only
#: role (real REVOKEs). This list is the second-layer statement gate.
_ALLOWED_TABLES = (
    "daily_metrics", "activity_summaries", "activity_detail_series",
    "activity_splits", "hr_zones", "power_zones", "race_predictions", "gear",
    "devices", "derived_metrics", "training_workout", "training_goal",
    "training_block", "training_week", "weather_forecast",
)

#: Metrics ``get_metric_trend`` supports: derived scores, then daily columns
#: (a daily column is only offered when the account's enabled data types can
#: write it).
_DERIVED_TREND_METRICS = ("acwr", "run_acwr", "run_cadence_drift")
_DAILY_TREND_METRICS = (
    "sleep_score", "sleep_time_hours", "resting_hr", "hrv_last_night_avg",
    "vo2max", "total_steps", "total_distance_m", "stress_avg",
    "body_battery_max", "body_battery_min", "weight_kg", "sweat_loss_ml",
)

#: Data-driven schema annotations: table-level overviews and per-column
#: unit/semantic hints. ``_schema_text`` renders the agent-facing *table
#: overview* from ``_TABLE_NOTES`` (only a short list of common columns is in
#: the prompt; the rest are fetched live via ``table_schema``, so auto-created
#: columns are always picked up); ``_COLUMN_DOCS`` feeds ``table_schema``'s
#: per-column descriptions.
_TABLE_NOTES: dict[str, str] = {
    "daily_metrics": (
        "one row per calendar_date (YYYY-MM-DD). The day's core health + "
        "activity scalars: sleep stages/score/duration, resting/max/min HR, HRV, "
        "stress, respiration, SpO2, steps, intensity minutes, body battery, "
        "VO2max, lactate threshold, sweat loss, weight/body composition. Wide "
        "table — never SELECT *; pick columns."
    ),
    "activity_summaries": (
        "one row per activity: type, start time, duration/elapsed/moving, "
        "distance, speed, running pace, HR avg/max + time-in-zone, power, "
        "cadence, respiration, calories, training load, elevation, per-activity "
        "weather"
    ),
    "activity_detail_series": (
        "one row per sample (every ~2-10 s) of an activity (activity_id + tick): "
        "elapsed_s, HR, cadence, power, speed, elevation, cumulative distance. "
        "For ONE activity use get_activity_detail instead. Query this table only "
        "to chart a session or compare sessions, and always bucket by time "
        "(GROUP BY FLOOR(elapsed_s / 60)) — never select raw samples. A metric "
        "is NULL on every sample when the device didn't record it"
    ),
    "activity_splits": (
        "one row per lap (activity_id + split_number): auto-laps (usually 1 km) "
        "or manual lap presses, with distance, duration, pace, start offset, "
        "HR/power/cadence and elevation gain"
    ),
    "hr_zones": (
        "configured HR zone boundaries (zone1..zone5 min/max) + training method, "
        "one row per sport; current snapshot"
    ),
    "power_zones": (
        "configured power zones (watts) + functional threshold power, one row per "
        "sport; current snapshot"
    ),
    "race_predictions": "single current-fitness snapshot of 5k/10k/half/marathon finish times",
    "gear": "one row per equipment item (type, name, cumulative distance, activity count, last use, retired); current snapshot",
    "devices": "one row per Garmin device (model name); current snapshot",
    "derived_metrics": (
        "one row per (calendar_date, metric); daily derived scores recomputed "
        "and stored each sync. metric = 'acwr' (ratio) with acwr_acute_load / "
        "acwr_chronic_load / acwr_daily_load, qualifier Sweet "
        "Spot/Elevated/Danger/Detraining. "
        "RUNNING-SPECIFIC (running activities only, so cycling/swimming cannot "
        "inflate it): 'run_acwr' (foot-strike volume ratio) with "
        "run_acute_km / run_chronic_km (km/day, 7d/28d EMA); 'run_cadence_drift' "
        "(pace-normalised cadence z-score over the trailing 90d fit — negative = "
        "overstriding / worse mechanics) with run_cadence (spm PER LEG; total "
        "both-feet = 2x) and run_gct_ms, "
        "qualifier Form Up/Stable/Form Dropping/Breaking Down. Pivot with "
        "WHERE metric = '<name>' (a day's components share its calendar_date)"
    ),
    "training_workout": (
        "planned workouts; READ-ONLY via SQL — write only through the training "
        "tool. planned_date and activity_type associate workouts to the active "
        "block and goal (goal_id is an optional manual override). status is "
        "planned/completed/partial/skipped; target_* columns hold the prescribed pace/HR-zone/power"
    ),
    "training_goal": (
        "one row per long-term goal/target (an event such as a marathon or a "
        "half-marathon) over [start_date, target_date]; an account may hold many. read-only via "
        "SQL — write only through the training tool. sport is "
        "run/cycle/swim/strength/rest/other"
    ),
    "training_block": (
        "one row per periodized phase of a goal: name/focus are free text, "
        "start_date/end_date define the span (ordered chronologically by start_date), "
        "target_weekly_km provides an optional baseline weekly volume for weeks in this phase; "
        "read-only via SQL — write only through the training tool"
    ),
    "training_week": (
        "one row per (block_id, week_start): explicit targets for one calendar "
        "week (Mon..Sun) of a block — distance_km and/or duration_min, optional "
        "is_deload flag; falls back to block's target_weekly_km when distance_km is NULL. "
        "read-only via SQL — write only through the training tool"
    ),
    "weather_forecast": (
        "stored daily Open-Meteo weather forecast refreshed each sync (calendar_date, "
        "temp_max_c, temp_min_c, precip_mm, wind_max_kmh, condition_code WMO integer)"
    ),
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
        "vo2max": "precise VO2max estimate; Garmin's most-recent value, already populated daily",
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
        "avg_cadence": (
            "SHARED across sports. RUNNING = steps/min PER LEG (single foot, "
            "~80-90); total both-feet cadence = 2x this value (give the 2x number "
            "as 'total spm' when reporting cadence for a run). CYCLING/ROWING = rpm "
            "(~85-100)."
        ),
        "max_cadence": "same unit as avg_cadence (per-leg spm running / rpm cycling)",
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
        "ts_ms": "epoch ms (wall-clock); use elapsed_s for time within the activity",
        "elapsed_s": "seconds since the activity's first sample — bucket by this",
        "speed_kmh": "km/h",
        "power_w": "watts",
        "cadence": "steps/min PER LEG for running (total both-feet = 2x); rpm for cycling; NULL if not recorded",
        "accumulated_power_w": "watts",
    },
    "activity_splits": {
        "distance_m": "metres",
        "duration_s": "seconds",
        "start_time_s": "offset into the activity (seconds)",
        "pace_sec_per_km": "seconds/km (lower = faster)",
        "avg_cadence": "steps/min PER LEG for running (total = 2x); rpm for cycling",
        "max_cadence": "same unit as avg_cadence",
        "split_type": "Garmin lap intensity: 'interval' (running laps), 'distance' (other active laps), 'rest', or 'split' (untyped)",
    },
    "race_predictions": {
        "time_5k_min": "minutes",
        "time_10k_min": "minutes",
        "time_half_marathon_min": "minutes",
        "time_marathon_min": "minutes",
    },
    "hr_zones": {
        "sport": (
            "UPPERCASE Garmin sport key. Values seen: RUNNING, CYCLING, DEFAULT. "
            "Match the stored case exactly (e.g. sport = 'RUNNING'), never lowercase."
        ),
        "training_method": "UPPERCASE: HR_RESERVE or LACTATE_THRESHOLD",
    },
    "power_zones": {
        "sport": (
            "UPPERCASE Garmin sport key. Values seen: RUNNING, CYCLING, ROWING, "
            "CROSS_COUNTRY_SKIING. Match the stored case exactly, never lowercase."
        ),
    },
    "derived_metrics": {
        "metric": "which derived metric the row holds — see the table note for known names",
        "value": "the metric's value; units differ per metric (acwr ratio, loads arbitrary)",
        "qualifier": "category label: acwr = Sweet Spot/Elevated/Danger/Detraining; run_cadence_drift = Form Up/Stable/Form Dropping/Breaking Down",
    },
    "training_workout": {
        "activity_type": "run/cycle/swim/strength/rest/other",
        "duration_min": "planned minutes",
        "distance_km": "planned km",
        "target_pace_min_km": "prescribed running pace as DECIMAL min/km (5.5 = 5:30)",
        "target_hr_zone": "prescribed HR zone/target, free text (e.g. 'Z2' or '140-150')",
        "target_power_w": "prescribed power in watts (int)",
        "steps": (
            "ordered JSON array of the session's segments, present whenever a "
            "workout has more than one (warm-up/work/recovery/cool-down, "
            "intervals, strides, ...); stored as "
            "{kind: warmup/steady/work/recovery/cooldown/rest, duration_sec, "
            "repeat?, label?, intensity?, target_pace_min_km?, target_hr_zone?, "
            "target_power_w?}; label is a short segment name (no duration/rep). "
            "When WRITING via the training tool pass a friendly `duration` "
            "string ('15m', '90s', '1:30') instead of seconds; NULL/[] only for "
            "a single continuous effort"
        ),
        "status": "planned/completed/partial/skipped; 'partial' and 'completed' both count as done",
        "goal_id": "training_goal.id the workout optionally overrides to (usually resolved automatically by planned_date)",
        "completed_activity_id": (
            "activity_summaries.activity_id that satisfied it — join for actual stats"
        ),
    },
    "training_goal": {
        "sport": "run/cycle/swim/strength/rest/other (same vocabulary as workouts); may be NULL",
        "start_date": "YYYY-MM-DD; campaign start date",
        "target_date": "YYYY-MM-DD; the event/race day, may be NULL",
        "target_time": "free text goal finish time (e.g. 3:45:00)",
    },
    "training_block": {
        "name": "free text phase label (Base/Build/Peak/Taper or any custom name)",
        "focus": "free text training focus",
        "start_date": "YYYY-MM-DD, may be NULL (undated block, which cannot carry week targets)",
        "end_date": "YYYY-MM-DD, may be NULL (undated block); must be on/after start_date",
        "target_weekly_km": "optional baseline weekly km for weeks in this block",
    },
    "training_week": {
        "week_start": "Monday of this calendar week (YYYY-MM-DD); derived from the block start + the week's position",
        "distance_km": "target km for this week (NULL if only time is tracked)",
        "duration_min": "target minutes for this week; careful about per-week totals vs a single session",
        "is_deload": "true when this week is a planned recovery/deload week",
    },
    "weather_forecast": {
        "calendar_date": "YYYY-MM-DD",
        "temp_max_c": "degC",
        "temp_min_c": "degC",
        "precip_mm": "mm",
        "wind_max_kmh": "km/h",
        "condition_code": "WMO weather condition code (0=clear, 1-3=cloudy, 51-67=rain, 71-77=snow)",
    },
}

#: The agent's static instructions, filled once per agent with ``str.format``
#: (``{overview}`` = the per-user table overviews, ``{training}`` = the training
#: bullet or nothing). The prompt only says WHEN to use each tool; HOW to call
#: one lives in that tool's docstring, which the model receives as the tool
#: description — so every rule is stated exactly once.
_PROMPT_TEMPLATE = """\
# Role

You are an expert sports scientist and data analyst with read-only access to the
user's personal Garmin health and fitness database (PostgreSQL). Give thorough,
accurate, insight-driven answers the user can act on: connect metrics, flag
trends, anomalies and relationships, and add concise context rather than just
returning raw numbers.

# Database

{overview}

Common columns (write SQL with these directly; call `table_schema` for any other):
- **daily_metrics**: `calendar_date`, `resting_hr` (bpm), `hrv_last_night_avg` (ms), `sleep_score` (0-100), `sleep_time_hours`, `vo2max`, `total_steps`, `total_distance_m` (METRES), `body_battery_max/min`, `stress_avg`, `weight_kg`
- **activity_summaries**: `activity_id`, `activity_name`, `activity_type` (running, cycling, ...), `start_date`, `duration_hours`, `distance_km` (KM), `avg_hr`, `max_hr`, `pace_min_km` (decimal min/km; 5.5 = 5:30), `avg_cadence` (PER LEG), `avg_power_w`, `training_load`, `elevation_gain_m`, `weather_temp_c`
- **derived_metrics**: `calendar_date`, `metric` ('acwr', 'run_acwr', 'run_cadence_drift'), `value`, `qualifier`
- **weather_forecast**: `calendar_date`, `temp_max_c`, `temp_min_c`, `precip_mm`, `wind_max_kmh`, `condition_code`

Query rules:
1. PostgreSQL dialect. Never `SELECT *`; pick only the columns you need.
2. Results are capped at 500 rows — aggregate (GROUP BY week/month/sport),
   `ORDER BY` time, and `LIMIT` to probe a table's shape first.
3. Almost any column may be NULL; aggregate over the rows you have and wrap
   denominators in `NULLIF(col, 0)`.
4. Date-like columns (`calendar_date`, `start_date`, `start_time_local`, ...) are
   TEXT (`'YYYY-MM-DD'` / `'HH:MM'`). Cast with `::date` before `DATE_TRUNC` /
   `EXTRACT`, or compare them as strings.
5. Measurements are `DOUBLE PRECISION`: cast before rounding
   (`ROUND(AVG(heart_rate)::numeric, 1)`).
6. Query freely: probing, correcting and follow-up queries are cheap and safe —
   accuracy matters more than the number of tool calls.

# Tools

- `get_day_summary` for one day; `get_metric_trend` for a core metric over time;
  `get_recent_activities` for a list of activities; `get_activity_detail` for
  how one session went. Otherwise write SQL with `run_sql`.
- `chart` when the user asks for a chart. On `OK: <spec>`, embed the spec
  verbatim in `<chart> ... </chart>` with one descriptive sentence; never paste
  the data.
- `weather` only for a day or place the database does not cover. A stored
  activity has its own weather columns; the home forecast is `weather_forecast`.
- `memory` for durable facts the user tells you about themselves.
{training}
# Output conventions

1. Always include units (`7.6 hours`, `154 bpm`, `245 W`).
2. Give running as pace (`MM:SS /km`), not km/h; lead with it.
3. Durations as `HH:MM:SS` or `Xh Ym`.
4. Running cadence is stored PER LEG — quote the total (2x) as spm; cycling and
   rowing cadence is rpm.
5. For running load prefer `run_acwr` (foot-strike volume) over `acwr` (all
   activities, blind to tissue stress); use `run_cadence_drift` (negative =
   overstriding) when the user mentions leg/joint/tendon aches or form breaking down.
6. Plain text for numbers, units and ranges (`20 km/wk → 75–78 km/wk`) — no LaTeX
   or `$...$` math markup, which renders as raw text in the chat.
"""

#: The training bullet of the prompt, emitted only when a season is wired.
_TRAINING_BULLET = """\
- `training` for everything about the training season (goals, blocks, weekly
  targets, planned workouts): read it before building or reshaping a week and
  to answer "how is my week going?". To show a plan in chat, embed
  `<plan_table />` (or `<plan_table from="YYYY-MM-DD" to="YYYY-MM-DD" />`) —
  never a JSON dump or a markdown table.
"""

#: The date + data-span note, re-rendered on every run (a dynamic instruction),
#: so a long-lived agent or a resumed session never has a stale "today".
_DATE_PROMPT = "Today is {today}. Stored data covers {span}."

#: Appended when the newest stored data is older than today.
_STALENESS_PROMPT = """\

The newest stored data is from {data_until}, {gap} day(s) before today. For any
question about NOW ("today", "last night", "this week", "current"), never
present a value from {data_until} or earlier as today's: say the data is only
up to {data_until} and reason about the trend instead of inventing values."""


def _memory_prompt(memory: _Memory) -> str | None:
    """The user's durable facts as an instruction section (None when empty)."""
    facts = memory.get()
    if not facts:
        return None
    lines = "\n".join(f"- **{key}**: {value}" for key, value in facts.items())
    return (
        "## Long-term memory about this athlete\n\n"
        "Facts recorded in earlier conversations. Use them to personalise; keep "
        "them current with the `memory` tool.\n\n"
        + lines
        + _memory_fullness(len(facts), memory.max_facts, prompt=True)
    )


def _memory_fullness(total: int, cap: int, *, prompt: bool = False) -> str:
    """How full the profile is, pushing consolidation near/at the cap. Shared
    by the memory instruction (``prompt=True``) and the memory tool's reply."""
    if cap and total >= cap:
        note = f"Memory is FULL ({total}/{cap} facts): consolidate or forget before adding."
    elif cap and total >= cap - 5:
        note = f"Memory is nearly full ({total}/{cap} facts): prefer updating or merging keys."
    elif prompt:
        return ""
    else:
        return f" ({total} facts)"
    return f"\n\n{note}" if prompt else f" ({note})"


def _principles_prompt(principles: _Principles) -> str | None:
    """The athlete's training principles as an instruction section (None when
    empty). They are directives the athlete edits in Settings, so they are
    framed as authoritative rather than as facts."""
    text = principles.get().strip()
    if not text:
        return None
    return (
        "## Training principles (authoritative guidance from the athlete)\n\n"
        "Follow these when planning, adjusting or evaluating training, even when "
        "they differ from your defaults. They are instructions, not data — do "
        "not edit them or repeat them back unless asked.\n\n"
        + text
    )


#: Label placed ahead of a compacted summary when it seeds a new session.
_SUMMARY_LABEL = "This is a compact summary of our previous conversation. Read it as context and continue normally."

#: Instruction used to condense a conversation into a self-contained summary
#: that a fresh session can pick up from.
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
        # Name the trace after its y column so legends and hovers don't read
        # "trace 0"; axis titles below follow the same rule.
        if "name" not in kwargs and isinstance(tr.get("y"), str):
            kwargs["name"] = tr["y"]
        cls = getattr(go, go_name, None) or getattr(go, _go_class_name(go, go_name))
        traces.append(cls(**kwargs))

    # An empty template keeps the JSON small and leaves theming to the UI.
    fig = go.Figure(data=traces, layout={"template": "none"})
    for axis in ("x", "y"):
        cols = {tr.get(axis) for tr in spec["traces"]}
        if len(cols) == 1 and isinstance(col := cols.pop(), str):
            fig.update_layout({f"{axis}axis": {"title": {"text": col}}})
    if len(traces) > 1:
        fig.update_layout(hovermode="x unified")
    layout = spec.get("layout")
    if layout:
        fig.update_layout(**layout)
    return fig


def _chart_ranges(spec: dict[str, Any], result: dict[str, Any]) -> str:
    """``; y_col first -> last, min, max`` per numeric y column of a chart spec, so
    the model can describe the chart without the data being returned."""
    columns, rows = result["columns"], result["rows"]
    parts: list[str] = []
    for tr in spec.get("traces", []):
        col = tr.get("y") if isinstance(tr, dict) else None
        if not isinstance(col, str) or col not in columns:
            continue
        i = columns.index(col)
        vals = [
            r[i] for r in rows
            if isinstance(r[i], int | float) and not isinstance(r[i], bool)
        ]
        if vals:
            parts.append(
                f"{col} {vals[0]:g} -> {vals[-1]:g}, min {min(vals):g}, max {max(vals):g}"
            )
    return "; " + "; ".join(parts) if parts else ""


#: The activity_summaries columns ``_format_activity`` reads.
_ACTIVITY_COLUMNS = (
    "activity_id, activity_name, activity_type, start_date, start_time_local, "
    "duration_hours, distance_km, avg_hr, max_hr, pace_min_km, avg_speed_kmh, "
    "avg_cadence, avg_power_w, training_load, elevation_gain_m, weather_temp_c, is_pr"
)

#: Most time buckets ``activity_detail`` returns (the bucket widens to fit).
_MAX_BUCKETS = 200


def _is_running(activity_type: str | None) -> bool:
    return "run" in (activity_type or "").lower()


def _round(value: Any, digits: int = 0) -> Any:
    if value is None:
        return None
    return round(float(value), digits) if digits else round(float(value))


def _fmt_duration(seconds: Any) -> str | None:
    """``H:MM:SS`` (or ``M:SS`` under an hour)."""
    if seconds is None:
        return None
    total = int(round(float(seconds)))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_pace(min_per_km: Any) -> str | None:
    """Decimal min/km -> ``M:SS /km``."""
    if not min_per_km or min_per_km <= 0:
        return None
    total = int(round(float(min_per_km) * 60))
    return f"{total // 60}:{total % 60:02d} /km"


def _fmt_cadence(per_leg: Any, running: bool) -> str | None:
    """Stored cadence (per leg for running) as the number to quote."""
    if not per_leg:
        return None
    return f"{round(per_leg * 2)} spm" if running else f"{round(per_leg)} rpm"


def _format_activity(r: dict[str, Any]) -> dict[str, Any]:
    """One activity_summaries row with pace, duration and cadence ready to quote."""
    running = _is_running(r["activity_type"])
    pace = r["pace_min_km"]
    if pace is None and r["avg_speed_kmh"]:
        pace = 60.0 / r["avg_speed_kmh"]
    duration = r["duration_hours"]
    return {
        "activity_id": r["activity_id"],
        "activity_name": r["activity_name"],
        "activity_type": r["activity_type"],
        "start_date": r["start_date"],
        "start_time": r["start_time_local"],
        "duration": _fmt_duration(duration * 3600) if duration else None,
        "distance_km": _round(r["distance_km"], 2) if r["distance_km"] else None,
        "pace": _fmt_pace(pace) if running else None,
        "speed_kmh": _round(r["avg_speed_kmh"], 1) if r["avg_speed_kmh"] and not running else None,
        "cadence": _fmt_cadence(r["avg_cadence"], running),
        "avg_hr": _round(r["avg_hr"]),
        "max_hr": _round(r["max_hr"]),
        "avg_power_w": _round(r["avg_power_w"]),
        "training_load": _round(r["training_load"]),
        "elevation_gain_m": _round(r["elevation_gain_m"]),
        "weather_temp_c": r["weather_temp_c"],
        "is_pr": bool(r["is_pr"]),
    }


class QueryError(Exception):
    """A read-only query was rejected by the Postgres driver."""


class ReadOnlyDB:
    """Read-only handle over one account's Garmin data plus safe query execution.

    Bound to a ``user_id``: every call draws a pooled connection, runs
    ``set_config('app.user_id', ...)`` so Row-Level Security scopes the
    transaction, and executes as the read-only PG role (SELECT on the agent
    tables only). Connections are per-call because Pydantic AI runs tools from a
    worker thread.

    ``excluded_types`` is the account's disabled daily data types (a comma list
    string or an iterable): ``daily_metrics`` columns that no *enabled* type can
    write — plus columns that are (almost) empty for this account — are hidden
    from the schema and day summaries and rejected in SQL, so the agent never
    sees metrics the account turned off (even before the next sync prunes their
    stored values) or that its devices don't record.
    """

    def __init__(
        self,
        url: str,
        *,
        user_id: int | None = None,
        excluded_types: Iterable[str] | str = (),
    ) -> None:
        self.url = url
        self.user_id = user_id
        if isinstance(excluded_types, str):
            excluded_types = excluded_types.split(",")
        excluded = {s.strip() for s in excluded_types if s.strip()}
        # Mirrors ``db.prune_excluded_types``: a column survives while any
        # enabled type still writes it.
        enabled = set().union(
            *(cols for name, cols in TYPE_COLUMNS.items() if name not in excluded)
        )
        self._excluded_columns: frozenset[str] = frozenset(
            set().union(*(TYPE_COLUMNS.get(name, set()) for name in excluded)) - enabled
        )
        self._pool: Any = None
        self._table_names: list[str] | None = None
        self._forbidden: re.Pattern | None = None
        self._hidden: frozenset[str] | None = None
        self._hidden_re: re.Pattern | None = None

    @contextmanager
    def _connect(self) -> Any:
        if self._pool is None:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self.url,
                min_size=1,
                max_size=8,
                open=True,
                configure=lambda conn: setattr(conn, "row_factory", dict_row),
            )
        with self._pool.connection() as conn:
            if self.user_id is not None:
                conn.execute(
                    "SELECT set_config('app.user_id', %s, true)", (str(self.user_id),)
                )
            yield conn

    def _hidden_columns(self) -> frozenset[str]:
        """``daily_metrics`` columns the agent must not see for this account.

        The disabled types' columns plus every column that is (almost) empty
        for this account (see ``_SPARSE_MAX_SHARE``): the table is shared, so
        it carries columns other accounts' devices create. One counting query
        per handle; the regex ``run_sql`` checks is built alongside.
        """
        if self._hidden is None:
            hidden = set(self._excluded_columns)
            with self._connect() as conn:
                names = [
                    r["column_name"] for r in conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'daily_metrics' AND column_name NOT IN "
                        "('user_id', 'calendar_date', 'fetched_at')"
                    ).fetchall()
                ]
                counts = conn.execute(
                    "SELECT count(*) AS n, "
                    + ", ".join(f'count("{c}") AS "{c}"' for c in names)
                    + " FROM daily_metrics"
                ).fetchone() if names else {"n": 0}
            total = counts["n"]
            if total:
                hidden |= {
                    c for c in names
                    if counts[c] < _SPARSE_MAX_VALUES
                    and counts[c] < total * _SPARSE_MAX_SHARE
                }
            self._hidden = frozenset(hidden)
            if hidden:
                alternatives = "|".join(
                    sorted(map(re.escape, hidden), key=len, reverse=True)
                )
                self._hidden_re = re.compile(
                    rf"\b(?:{alternatives})\b", re.IGNORECASE
                )
        return self._hidden

    def _all_tables(self) -> list[str]:
        """Every public table name (listed once per handle)."""
        if self._table_names is None:
            with self._connect() as conn:
                self._table_names = [
                    r["table_name"] for r in conn.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' ORDER BY table_name"
                    ).fetchall()
                ]
        return self._table_names

    def tables(self) -> list[str]:
        return [n for n in self._all_tables() if n in _ALLOWED_TABLES]

    def _forbidden_table_regex(self) -> re.Pattern | None:
        """Regex matching any existing table the agent must not reference."""
        if self._forbidden is None:
            banned = [n for n in self._all_tables() if n not in _ALLOWED_TABLES]
            if not banned:
                return None
            alternatives = "|".join(sorted(map(re.escape, banned), key=len, reverse=True))
            self._forbidden = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)
        return self._forbidden

    def columns(self, table: str) -> list[dict[str, Any]]:
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"unknown table: {table!r}")
        with self._connect() as conn:
            cols = [
                dict(r) for r in conn.execute(
                    "SELECT column_name AS name, data_type AS type "
                    "FROM information_schema.columns WHERE table_name = %s "
                    "ORDER BY ordinal_position",
                    (table,),
                ).fetchall()
            ]
        if table != "daily_metrics":
            return cols
        hidden = self._hidden_columns()
        return [c for c in cols if c["name"] not in hidden]

    def data_span(self) -> tuple[date | None, date | None]:
        """(first, last) date with any stored data, or ``(None, None)``.

        ``last`` looks across daily metrics, derived metrics and activities so
        the agent can tell how stale the store is relative to today. Every date
        column is TEXT ``YYYY-MM-DD``, so MIN/MAX are lexicographic.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT (SELECT MIN(calendar_date) FROM daily_metrics) AS first,"
                    " GREATEST((SELECT MAX(calendar_date) FROM daily_metrics),"
                    " (SELECT MAX(calendar_date) FROM derived_metrics),"
                    " (SELECT MAX(start_date) FROM activity_summaries)) AS last"
                ).fetchone()
        except psycopg.Error:
            return None, None

        def parse(value: Any) -> date | None:
            try:
                return date.fromisoformat(str(value)[:10]) if value else None
            except ValueError:
                return None

        return parse(row["first"]), parse(row["last"])

    def day_summary(self, calendar_date: str) -> dict[str, Any]:
        """Every stored daily metric plus activities for one calendar date."""
        try:
            day = date.fromisoformat(calendar_date.strip()).isoformat()
        except (AttributeError, ValueError) as exc:
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
            running = conn.execute(
                "SELECT metric, value, qualifier FROM derived_metrics "
                "WHERE calendar_date = %s AND metric LIKE 'run_%%' ORDER BY metric",
                (day,),
            ).fetchall()
        metrics: dict[str, Any] = {}
        if row:
            metrics = {
                k: _jsonable(v)
                for k, v in dict(row).items()
                if v is not None
                and k not in ("user_id", "calendar_date", "fetched_at")
                and k not in self._hidden_columns()
            }
        return {
            "calendar_date": day,
            "metrics": metrics,
            "activities": _jsonify_rows([dict(r) for r in activities]),
            "running": _jsonify_rows([dict(r) for r in running]),
        }

    def trend_metrics(self) -> list[str]:
        """The metrics ``metric_trend`` supports for this account."""
        present = {c["name"] for c in self.columns("daily_metrics")}
        return list(_DERIVED_TREND_METRICS) + [
            m for m in _DAILY_TREND_METRICS if m in present
        ]

    def metric_trend(self, metric: str, days: int = 30) -> dict[str, Any]:
        """Query time series and summary stats for a core metric over N trailing days."""
        days = max(1, min(int(days), 365))
        m = metric.strip().lower()
        supported = self.trend_metrics()
        if m not in supported:
            raise ValueError(
                f"unsupported metric '{metric}'. Supported: {', '.join(supported)} "
                "(or write custom SQL with run_sql)"
            )
        start = (date.today() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            if m in _DERIVED_TREND_METRICS:
                rows = conn.execute(
                    "SELECT calendar_date, value, qualifier FROM derived_metrics "
                    "WHERE metric = %s AND calendar_date >= %s ORDER BY calendar_date ASC",
                    (m, start),
                ).fetchall()
                series = [
                    {"date": r["calendar_date"], "value": _jsonable(r["value"]), "qualifier": r["qualifier"]}
                    for r in rows
                ]
            else:
                # ``m`` is whitelisted above, so interpolating it is safe.
                rows = conn.execute(
                    f"SELECT calendar_date, {m} FROM daily_metrics "
                    f"WHERE calendar_date >= %s AND {m} IS NOT NULL ORDER BY calendar_date ASC",
                    (start,),
                ).fetchall()
                series = [
                    {"date": r["calendar_date"], "value": _jsonable(r[m])}
                    for r in rows
                ]
        vals = [s["value"] for s in series if isinstance(s.get("value"), (int, float))]
        summary: dict[str, Any] = {"count": len(series)}
        if vals:
            summary["min"] = min(vals)
            summary["max"] = max(vals)
            summary["avg"] = round(sum(vals) / len(vals), 2)
            summary["latest"] = vals[-1]
            summary["latest_date"] = series[-1]["date"]
            if len(vals) >= 2:
                summary["change_vs_start"] = round(vals[-1] - vals[0], 2)
        return {
            "metric": m,
            "days_requested": days,
            "summary": summary,
            "series": series,
        }

    def recent_activities(
        self,
        sport: str | None = None,
        days: int = 30,
        limit: int = 10,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return formatted recent activities with pace and cadence converted.

        If ``date_start`` and/or ``date_end`` (ISO YYYY-MM-DD) are given the
        window is scoped to that inclusive date range; otherwise it uses the
        trailing ``days`` window. ``limit`` caps the rows (max 50).
        """
        days = max(1, min(int(days), 180))
        limit = max(1, min(int(limit), 50))
        if date_start is not None:
            date_start = date.fromisoformat(date_start).isoformat()
        if date_end is not None:
            date_end = date.fromisoformat(date_end).isoformat()
        if date_start and date_end and date_start > date_end:
            date_start, date_end = date_end, date_start
        start = date_start or (date.today() - timedelta(days=days)).isoformat()
        params: list[Any] = [start]
        sql = f"SELECT {_ACTIVITY_COLUMNS} FROM activity_summaries WHERE start_date >= %s "
        if date_end:
            sql += "AND start_date <= %s "
            params.append(date_end)
        if sport and sport.strip():
            sql += "AND activity_type ILIKE %s "
            params.append(f"%{sport.strip()}%")
        sql += "ORDER BY start_date DESC, start_time_local DESC NULLS LAST LIMIT %s"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_format_activity(r) for r in rows]

    def activity_detail(
        self, activity_id: int | None = None, bucket_s: int = 60
    ) -> dict[str, Any]:
        """One activity in one call: its summary, its laps, and its time series
        averaged into ``bucket_s``-second buckets of elapsed time (widened so
        there are at most ``_MAX_BUCKETS``). ``activity_id=None`` is the most
        recent activity."""
        with self._connect() as conn:
            if activity_id is None:
                row = conn.execute(
                    f"SELECT {_ACTIVITY_COLUMNS} FROM activity_summaries "
                    "ORDER BY start_date DESC, start_time_local DESC NULLS LAST LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    f"SELECT {_ACTIVITY_COLUMNS} FROM activity_summaries "
                    "WHERE activity_id = %s",
                    (int(activity_id),),
                ).fetchone()
            if row is None:
                raise ValueError(f"activity {activity_id} not found")
            aid = row["activity_id"]
            span = conn.execute(
                "SELECT MAX(elapsed_s) AS span FROM activity_detail_series "
                "WHERE activity_id = %s",
                (aid,),
            ).fetchone()["span"] or 0
            bucket = max(5, int(bucket_s), math.ceil(span / _MAX_BUCKETS))
            series = conn.execute(
                "SELECT FLOOR(elapsed_s / %s) * %s AS t, AVG(heart_rate) AS hr, "
                "MAX(heart_rate) AS hr_max, AVG(speed_kmh) AS speed_kmh, "
                "AVG(cadence) AS cadence, AVG(power_w) AS power_w, "
                "AVG(elevation_m) AS elevation_m, MAX(distance_m) AS distance_m "
                "FROM activity_detail_series "
                "WHERE activity_id = %s AND elapsed_s IS NOT NULL "
                "GROUP BY 1 ORDER BY 1",
                (bucket, bucket, aid),
            ).fetchall()
            laps = conn.execute(
                "SELECT split_number, split_type, start_time_s, duration_s, "
                "distance_m, pace_sec_per_km, avg_hr, max_hr, avg_power, "
                "avg_cadence, elevation_gain_m FROM activity_splits "
                "WHERE activity_id = %s ORDER BY split_number",
                (aid,),
            ).fetchall()
        running = _is_running(row["activity_type"])
        points = [
            {
                "t": _fmt_duration(r["t"]),
                "hr": _round(r["hr"]),
                "hr_max": _round(r["hr_max"]),
                "pace": _fmt_pace(60.0 / r["speed_kmh"]) if running and r["speed_kmh"] else None,
                "speed_kmh": None if running else _round(r["speed_kmh"], 1),
                "cadence": _fmt_cadence(r["cadence"], running),
                "power_w": _round(r["power_w"]),
                "elevation_m": _round(r["elevation_m"], 1),
                "km": _round((r["distance_m"] or 0) / 1000.0, 2) if r["distance_m"] else None,
            }
            for r in series
        ]
        # Columnar and without all-NULL metrics: a ~200-row series stays small.
        columns = [c for c in (points[0] if points else {}) if any(p[c] is not None for p in points)]
        return {
            "activity": _format_activity(row),
            "laps": [
                {
                    "lap": lap["split_number"] + 1,
                    "type": lap["split_type"],
                    "start": _fmt_duration(lap["start_time_s"]),
                    "duration": _fmt_duration(lap["duration_s"]),
                    "km": _round((lap["distance_m"] or 0) / 1000.0, 2) if lap["distance_m"] else None,
                    "pace": _fmt_pace(lap["pace_sec_per_km"] / 60.0) if running and lap["pace_sec_per_km"] else None,
                    "avg_hr": _round(lap["avg_hr"]),
                    "max_hr": _round(lap["max_hr"]),
                    "cadence": _fmt_cadence(lap["avg_cadence"], running),
                    "power_w": _round(lap["avg_power"]),
                    "elevation_gain_m": _round(lap["elevation_gain_m"]),
                }
                for lap in laps
            ],
            "series": {
                "bucket_s": bucket,
                "columns": columns,
                "rows": [[p[c] for c in columns] for p in points],
            },
        }


    def run_sql(self, sql: str) -> dict[str, Any]:
        """Validate and execute a read-only query, returning rows as JSON-safe."""
        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise ValueError("empty statement")
        if ";" in statement:
            raise ValueError("only a single statement is allowed")
        if not _SELECT_PREFIX.match(statement):
            raise ValueError("only SELECT / WITH / EXPLAIN statements are allowed")
        write_word = _WRITE_WORDS.search(statement)
        if write_word:
            raise ValueError(
                f"statement contains a write keyword and was rejected: {write_word.group(0)}"
            )
        forbidden = self._forbidden_table_regex()
        if forbidden and forbidden.search(statement):
            raise ValueError(
                "statement references a table outside the allowed set "
                f"({', '.join(_ALLOWED_TABLES)})"
            )
        self._hidden_columns()
        hidden = self._hidden_re and self._hidden_re.search(statement)
        if hidden:
            raise ValueError(
                f"column {hidden.group(0)} holds no usable data for this account "
                "(its data type is disabled or it is almost never recorded)"
            )
        with self._connect() as conn:
            # Tuple rows: a dict row would merge same-named columns
            # (``SELECT AVG(a), AVG(b)`` -> one ``avg``).
            cur = conn.cursor(row_factory=psycopg.rows.tuple_row)
            try:
                cur.execute(statement)
            except psycopg.Error as exc:
                raise QueryError(f"{exc} | statement: {statement!r}") from exc
            columns = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(_MAX_ROWS + 1)
        truncated = len(rows) > _MAX_ROWS
        rows = rows[:_MAX_ROWS]
        return {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows],
            "truncated": truncated,
            "note": f"truncated to {_MAX_ROWS} rows" if truncated else f"{len(rows)} rows",
        }

    def close(self) -> None:
        """Release the pooled connections (no-op if never opened)."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None


class Weather:
    """Stateless Open-Meteo access (archive + short forecast).

    Every call is one small HTTP request; nothing is cached or stored here, so
    the agent's ``weather`` tool can never be a source of truth — only context
    for the stored data. The sync also uses it to refresh ``weather_forecast``.
    Coordinates come from a configured home location and can be overridden per
    request (both ``lat`` and ``lon`` together).
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    MAX_DAYS = 92
    FORECAST_DAYS = 16
    _DAILY_FIELDS = (
        "temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "wind_speed_10m_max,weather_code"
    )
    _FIELD_ALIASES = {
        "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c",
        "precipitation_sum": "precip_mm",
        "wind_speed_10m_max": "wind_max_kmh",
        "weather_code": "condition_code",
    }

    def __init__(
        self, *, default_lat: float | None = None, default_lon: float | None = None
    ) -> None:
        self._default_lat = default_lat
        self._default_lon = default_lon

    @classmethod
    def from_config(cls, cfg: dict[str, str]) -> "Weather":
        """Build from config strings (empty strings become no default)."""
        return cls(
            default_lat=_float_or_none(cfg.get("weather_home_lat")),
            default_lon=_float_or_none(cfg.get("weather_home_lon")),
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

        A range fully before today uses the historical archive; a range
        starting today or later is the forecast (up to ``FORECAST_DAYS`` days
        ahead). With no dates the forecast from today onward is returned. Raises
        ``ValueError`` with a model-facing message on invalid input.
        """
        import httpx

        lat, lon = self._resolve(lat, lon)
        today = date.today()
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
        if (end - start).days + 1 > self.MAX_DAYS:
            raise ValueError(f"request at most {self.MAX_DAYS} days at a time")

        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "daily": self._DAILY_FIELDS,
            "timezone": "auto",
        }
        if end < today:
            url = self.ARCHIVE_URL
            params["start_date"] = start.isoformat()
            params["end_date"] = end.isoformat()
        elif start >= today:
            url = self.FORECAST_URL
            forecast_end = today + timedelta(days=self.FORECAST_DAYS - 1)
            if end > forecast_end:
                raise ValueError(
                    f"the forecast only covers up to {forecast_end.isoformat()} "
                    f"({self.FORECAST_DAYS} days)"
                )
            # ``forecast_days`` is mutually exclusive with start/end dates, so
            # request from today and trim to the range below.
            params["forecast_days"] = (end - today).days + 1
        else:
            raise ValueError(
                "the range crosses today: pick range wholly before today "
                "(history) or wholly today-and-later (forecast)"
            )
        try:
            resp = httpx.get(url, params=params, timeout=15.0, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ValueError(f"weather service unavailable: {exc}") from exc
        daily = resp.json().get("daily") or {}
        days: list[dict[str, Any]] = []
        for i, day in enumerate(daily.get("time") or []):
            if start <= date.fromisoformat(day) <= end:
                row: dict[str, Any] = {"date": day}
                for field, alias in self._FIELD_ALIASES.items():
                    series = daily.get(field) or []
                    row[alias] = series[i] if i < len(series) else None
                days.append(row)
        return {
            "source": "forecast" if url == self.FORECAST_URL else "historical",
            "location": {"lat": lat, "lon": lon},
            "days": days,
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


#: Column-name patterns -> short auto-description, used when ``_COLUMN_DOCS``
#: has no explicit hint for a column. Ordered longest-first so the most
#: specific suffix wins.
_COLUMN_NAME_DOCS: tuple[tuple[str, str], ...] = (
    ("total_steps", "step count"),
    ("accumulated_power_w", "accumulated work (watts)"),
    ("sleep_time_hours", "time asleep (hours, 4dp)"),
    ("pace_min_km", "running pace (decimal min/km; 5.5 = 5:30)"),
    ("distance_km", "distance (km)"),
    ("speed_kmh", "speed (km/h)"),
    ("spo2_avg_sleep", "blood-oxygen saturation during sleep (%)"),
    ("spo2_avg", "blood-oxygen saturation (%)"),
    ("spo2_latest", "blood-oxygen saturation (%)"),
    ("spo2_lowest", "blood-oxygen saturation (%)"),
    ("spo2_last_7d_avg", "7-day average blood-oxygen saturation (%)"),
    ("weight_kg", "body weight (kg)"),
    ("bmi", "body mass index"),
    ("body_fat_pct", "body-fat percentage"),
    ("total_distance_m", "distance (metres)"),
    ("avg_speed_kmh", "average speed (km/h)"),
    ("max_speed_kmh", "max speed (km/h)"),
    ("_kmh", "value (km/h)"),
    ("_sec_per_km", "pace (seconds/km)"),
    ("_hours", "duration (hours)"),
    ("_watts", "power (watts)"),
    ("_kcal", "energy (kcal)"),
    ("_ml", "volume (ml)"),
    ("_pct", "percentage (0-100)"),
    ("_min", "minutes"),
    ("_spm", "steps/min"),
    ("_km", "distance (km)"),
    ("_m", "distance (metres)"),
    ("_w", "power (watts)"),
    ("_hr", "heart rate (bpm)"),
    ("_c", "temperature (degC)"),
    ("cadence", "steps/min PER LEG for running (total both-feet = 2x); rpm for cycling"),
    ("hr_time_zone", "% of activity duration spent in this HR zone"),
    ("power_time_zone", "% of activity duration spent in this power zone"),
    ("calories", "energy (kcal)"),
    ("latitude", "WGS84 latitude"),
    ("longitude", "WGS84 longitude"),
)

_TEXT_TYPE = frozenset({
    "text", "character varying", "varchar", "uuid", "json", "jsonb",
})
_NUM_TYPE = frozenset({
    "integer", "bigint", "smallint", "numeric", "real", "double precision",
    "decimal",
})
_DATE_TYPE_HINTS = ("timestamp", "date", "time")


def _auto_col_desc(name: str, type: str) -> str:
    """A short description for a column that has no curated hint in ``_COLUMN_DOCS``.

    Uses the column name's unit/semantic suffix first, then the data type, so
    even a lazily auto-created column gets a meaningful description in the
    ``table_schema`` tool output. Never raises.
    """
    n = name.lower()
    t = (type or "").lower()
    for suffix, desc in _COLUMN_NAME_DOCS:
        if n == suffix or n.endswith(suffix):
            return desc
    # Name hints beat the generic data-type fallback: several columns store a
    # date/timestamp as *text* (e.g. calendar_date, fetched_at, start_date).
    if "date" in n and t in _TEXT_TYPE:
        return "date (YYYY-MM-DD)"
    if any(h in t for h in _DATE_TYPE_HINTS):
        return "timestamp"
    if t == "boolean":
        return "boolean flag"
    if t in _TEXT_TYPE:
        return "text / category"
    if t in _NUM_TYPE:
        return "numeric value"
    return "value"


def _column_descriptions(db: ReadOnlyDB, table: str) -> list[dict[str, str]]:
    """Live columns for ``table``, each annotated with a short description.

    Uses ``_COLUMN_DOCS`` (curated) when available, otherwise ``_auto_col_desc``
    (generated from the name/unit + data type), so the description list stays
    complete even for columns created dynamically at sync time.
    """
    docs = _COLUMN_DOCS.get(table, {})
    out: list[dict[str, str]] = []
    for col in db.columns(table):
        name = col["name"]
        out.append({
            "name": name,
            "type": col["type"],
            "description": docs.get(name) or _auto_col_desc(name, col["type"]),
        })
    return out


def _schema_text(db: ReadOnlyDB) -> str:
    """The agent-facing *table overview*: one ``_TABLE_NOTES`` line per table.

    Columns are fetched on demand through ``table_schema``, so the prompt stays
    lean and never goes stale as columns are added dynamically.
    """
    return "\n".join(
        f"- {table}: {_TABLE_NOTES.get(table, 'see table_schema')}"
        for table in db.tables()
    ) or "(no tables available in the database)"


def _render_system_prompt(overview: str, *, has_training: bool) -> str:
    """Fill the static instructions for the tools this agent actually has, so
    the prompt never advertises a tool the model cannot call."""
    return _PROMPT_TEMPLATE.format(
        overview=overview,
        training=_TRAINING_BULLET if has_training else "",
    ).strip()


def _date_prompt(db: ReadOnlyDB) -> str:
    """Today's date (with weekday, which models otherwise miscompute), the
    stored data span and, when the data lags today, a staleness warning."""
    today = date.today()
    first, last = db.data_span()
    span = f"{first}..{last}" if first and last else "nothing yet (no data synced)"
    text = _DATE_PROMPT.format(today=today.strftime("%Y-%m-%d (%A)"), span=span)
    if last is not None and (gap := (today - last).days) > 0:
        text += _STALENESS_PROMPT.format(data_until=last.isoformat(), gap=gap)
    return text


def _gemini_thinking_level(effort: str | None) -> str:
    """Map a unified reasoning-effort level to a Gemini thinking level.

    ``gemini-3.8-flash`` (and newer Gemini models) accept explicit levels;
    fall back to ``MINIMAL`` for anything unrecognized so reasoning never
    silently turns off.
    """
    return {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
        "xhigh": "HIGH",
    }.get((effort or "").lower(), "MINIMAL")


def _build_model(
    provider: str,
    model_name: str,
    *,
    base_url: str | None,
    api_key: str | None,
    reasoning_effort: str | None,
) -> Any:
    """Construct the chat model for ``provider`` ('openai' or 'gemini')."""
    if provider == "gemini":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        settings = None
        if reasoning_effort:
            settings = {
                "google_thinking_config": {
                    "thinking_level": _gemini_thinking_level(reasoning_effort)
                }
            }
        return GoogleModel(
            model_name,
            provider=GoogleProvider(api_key=api_key),
            settings=settings,
        )
    # Default: any OpenAI-compatible base_url (DeepSeek, Ollama, OpenAI, ...).
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model_settings = (
        {"openai_reasoning_effort": reasoning_effort} if reasoning_effort else None
    )
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        settings=model_settings,
    )


#: Failures a tool reports back to the model as ``ERROR: ...`` text (so it can
#: correct itself) instead of aborting the run.
_TOOL_ERRORS: tuple[type[BaseException], ...] = (
    ValueError, QueryError, OSError, psycopg.Error,
)


def _tool_error(exc: BaseException) -> str:
    """Format a tool exception, avoiding a duplicated ``ERROR: `` prefix."""
    msg = str(exc)
    return msg if msg.startswith("ERROR: ") else f"ERROR: {msg}"


def _register_tools(
    agent: Any,
    db: ReadOnlyDB,
    *,
    weather: Weather | None,
    memory: _Memory | None,
    training: _Training | None,
    charts: dict[str, Any] | None,
    on_status: Callable[[str], None] | None,
) -> None:
    """Register every agent tool. Only ``memory`` and ``training`` write, each
    through its own app store — never through the read-only connection."""

    def tool(status: str, **options: Any) -> Callable[[Callable[..., Any]], Any]:
        """Register ``fn`` as a plain tool that reports ``status``, returns a
        dict/list as JSON and turns an expected failure into ``ERROR: ...``."""

        def register(fn: Callable[..., Any]) -> Any:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> str:
                if on_status is not None:
                    on_status(status)
                try:
                    out = fn(*args, **kwargs)
                except _TOOL_ERRORS as exc:
                    return _tool_error(exc)
                return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)

            return agent.tool_plain(**options)(wrapper)

        return register

    @tool("Inspecting table schema…")
    def table_schema(table: str) -> Any:
        """Return a table's live columns (name, type, description each).

        Call this before querying a column that is not in the prompt's common
        column list. Columns are read live, so newly-added ones appear.
        """
        return _column_descriptions(db, table)

    @tool("Running database query…")
    def run_sql(sql: str) -> Any:
        """Run one read-only SQL query (SELECT / WITH / EXPLAIN only) and return
        its columns and rows as JSON. Results are capped at 500 rows."""
        return db.run_sql(sql)

    @tool("Fetching day summary…")
    def get_day_summary(calendar_date: str) -> Any:
        """Return everything stored for one date (YYYY-MM-DD): the day's
        non-null daily_metrics values, its activities and its running-form
        scores. Use it for "how was my day on X?" instead of multi-table SQL."""
        return db.day_summary(calendar_date)

    @tool(
        "Analyzing metric trend…",
        description=(
            "Return the daily values plus min/max/avg/latest/change for one core "
            "metric over the trailing `days` (default 30, max 365). Metrics: "
            + ", ".join(db.trend_metrics())
            + ". Use it instead of SQL for trends or averages of these metrics."
        ),
    )
    def get_metric_trend(metric: str, days: int = 30) -> Any:
        return db.metric_trend(metric, days)

    @tool("Retrieving recent activities…")
    def get_recent_activities(
        sport: str | None = None,
        days: int = 30,
        limit: int = 10,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> Any:
        """Return activities with pace as MM:SS /km, total (2x) running
        cadence, duration, distance, HR, power, load and elevation.

        ``sport`` optionally filters by type (e.g. 'run', 'cycling'). The window
        is the trailing ``days`` (default 30), or the inclusive
        ``date_start``/``date_end`` range (YYYY-MM-DD; either may be open).
        ``limit`` caps the rows (default 10, max 50).
        """
        return db.recent_activities(sport, days, limit, date_start, date_end)

    @tool("Analyzing the session…")
    def get_activity_detail(activity_id: int | None = None, bucket_s: int = 60) -> Any:
        """Return one activity in full: its summary, its laps (auto-laps or
        manual lap presses, with pace/HR/cadence/power each) and its time
        series averaged into ``bucket_s``-second buckets of elapsed time
        (default 60; widened to at most 200 buckets). Omit ``activity_id`` for
        the most recent activity. Use it for any question about how a single
        session went — intervals, pacing, HR drift, fade — and use a smaller
        ``bucket_s`` (e.g. 15) to resolve short efforts.
        """
        return db.activity_detail(activity_id, bucket_s)

    @tool("Validating chart…")
    def chart(spec: str) -> Any:
        """Validate a chart spec and return it ready to embed.

        ``spec`` is a JSON object describing a Plotly figure:
          {
            "sql": "SELECT ...",             # read-only; returns the data
            "traces": [{
              "type": "line",                # line|scatter|area|bar|pie|histogram|box,
                                             # or "go": any plotly.graph_objects class
                                             # (Scattergl, Violin, Heatmap, ...)
              "x": "<result column>",        # x / y / z name result columns
              "y": "<numeric result column>",
              "marker": {"color": "red"}     # any other key goes to the Plotly trace
            }],
            "layout": {"title": {"text": "..."}}   # optional Plotly layout
          }
        Any other trace argument can take column data as {"column": "<name>"}
        (e.g. pie labels/values). Aggregate to at most ~200 points. The query is
        run and the figure built to check the spec; on success the reply is
        ``OK: <spec>`` — embed that spec verbatim in <chart> ... </chart>. The
        data itself is never returned; the UI reruns the query to draw it.
        """
        try:
            parsed = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise ValueError(f"spec is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sql"), str):
            raise ValueError("spec must be a JSON object with a string 'sql' key")
        result = db.run_sql(parsed["sql"])
        if result["truncated"]:
            raise ValueError(
                f"query returned more than {len(result['rows'])} rows and the "
                "chart would be cut off; aggregate (e.g. by day or week) or "
                "narrow the date range"
            )
        figure = _build_chart_figure(parsed, result)
        if charts is not None:
            charts[parsed["sql"].strip()] = json.loads(figure.to_json())
        return (
            "OK: " + json.dumps(parsed, ensure_ascii=False)
            + f" (query returned {len(result['rows'])} rows"
            + _chart_ranges(parsed, result) + ")"
        )

    if weather is not None:

        @tool("Checking the weather…", name="weather")
        def weather_tool(
            date_start: str | None = None,
            date_end: str | None = None,
            lat: float | None = None,
            lon: float | None = None,
        ) -> Any:
            """Return daily Open-Meteo weather (min/max degC, precip mm, max wind
            km/h, WMO condition code).

            ``date_start``/``date_end`` are inclusive YYYY-MM-DD bounds: a range
            wholly before today is observed history, one from today on is the
            forecast (up to 16 days); with neither, the forecast from today.
            Coordinates default to the athlete's home; override with both
            ``lat`` and ``lon``. Context only — never the source of an answer.
            """
            return weather.query(date_start, date_end, lat, lon)


    if memory is not None:

        @tool("Updating memory…", name="memory")
        def memory_tool(
            action: str = "set",
            facts: dict[str, Any] | None = None,
            keys: list[str] | None = None,
        ) -> Any:
            """Maintain durable facts about the athlete (long-term memory).

            HOLD A HIGH BAR. Store a fact only when it is BOTH durable (still
            true and useful months from now, in unrelated conversations) AND
            high-ROI (it changes how you coach): goals, preferences, habits,
            injuries, equipment, constraints. Never store transient details (a
            one-off question, a bad night, this week's soreness or schedule),
            and never database-queryable metrics (VO2max, FTP, LTHR, PRs, zones,
            resting HR/HRV) — query those fresh. When in doubt, don't. Existing
            facts are already in your instructions: update the key that covers
            a topic instead of adding a near-duplicate, and keep the profile small.

            This is a SILENT side effect: call it early and always finish with
            the full answer to the user's question — never end a turn with only
            a note that something was saved.

            - ``action="set"`` (default): upsert ``facts`` ({snake_case_key: value}).
            - ``action="forget"``: delete ``keys`` that are wrong, stale or superseded.
            - ``action="replace"``: overwrite the WHOLE profile with ``facts`` —
              how you consolidate overlapping keys.
            """
            action = (action or "set").strip().lower()
            if action == "forget":
                if not keys:
                    raise ValueError("keys is required for action=forget")
                removed = memory.forget(keys)
                if not removed:
                    raise ValueError("no such key(s): " + ", ".join(map(str, keys)))
                reply, total = "forgotten: " + ", ".join(removed), len(memory.get())
            elif action in ("set", "replace"):
                if not facts:
                    raise ValueError(f"facts is required for action={action}")
                if action == "set":
                    reply, total = "saved", memory.remember(facts)
                else:
                    reply, total = "replaced profile", memory.replace(facts)
            else:
                raise ValueError("action must be set, forget or replace")
            return reply + _memory_fullness(total, memory.max_facts)

    if training is not None:

        @tool("Consulting the training season…", name="training")
        def training_tool(
            action: str = "get",
            spec: str | None = None,
            day: str | None = None,
            weeks: int = 1,
            date_start: str | None = None,
            date_end: str | None = None,
            goal_id: int | None = None,
            full: bool = False,
            detail: bool = False,
        ) -> Any:
            """Read or edit the training season: goals -> periodized blocks ->
            weekly targets -> dated workouts.

            ``action="get"`` (default) returns an agenda for ``day`` (default
            today): each goal with the blocks overlapping the window, plus a
            ``weeks`` list with one entry per calendar week (Mon..Sun) holding
            that week's per-goal ``targets`` (week target and planned/actual
            ``coverage``), ``planned`` workouts and ``actual`` activities. The
            window is ``weeks`` weeks from Monday of ``day``'s week, or an
            explicit ``date_start``/``date_end``. ``goal_id`` focuses one goal;
            ``detail=true`` keeps full workout descriptions; ``full=true`` adds
            every block/week of each goal plus the resolved current block (use
            it before a season-wide reshape). ``can_undo`` says whether the last
            destructive edit can be rolled back.

            ``action="apply"`` makes one ATOMIC edit (all or nothing). ``spec``
            is JSON with a ``"workouts"`` and/or an ``"anchor"`` section, or
            ``{"undo": true}`` to roll back the last destructive edit. The
            result lists the changed ids, so no follow-up read is needed.

            "workouts": a list of entries. An entry with ``id`` is PATCHED (send
            only what changes); one without creates. ``delete_ids`` deletes. To
            move or copy workouts, read them first — never guess ids. Fields:
            planned_date, activity_type (run/cycle/swim/strength/rest/other),
            title, description (ONE short line, <= ~200 chars, about THIS
            session), duration_min, distance_km, intensity
            (easy/moderate/hard/race_pace), target_pace_min_km (decimal min/km,
            5.5 = 5:30), target_hr_zone, target_power_w, status
            (planned/completed/partial/skipped), optional goal_id override
            (workouts attach to the active block/goal by date). Add ``steps``
            whenever a session has more than one segment (warm-up/cool-down,
            intervals, strides, progressive runs): an ordered list of {kind
            (warmup/steady/work/recovery/cooldown/rest), duration
            ('15m'/'90s'/'1:30'), repeat?, label? (1-2 words, never digits or
            times), intensity?, target_pace_min_km?, target_hr_zone?,
            target_power_w?}.

            "anchor": ``goal_id`` patches that goal (omit to create one; a
            blocks-only spec targets the sole goal); fields title, sport
            (run/cycle/swim/strength/rest/other), start_date, target_date,
            target_time, target_distance_km. ``blocks`` upserts blocks (with
            ``id`` patches, without creates): name, focus, optional
            start_date/end_date, target_weekly_km (baseline for weeks without
            their own target — set it when weeks vary), and optional ``weeks``:
            a per-week patch keyed by ``week_start`` — {week_start, distance_km,
            duration_min, is_deload} writes only that week, a bare {week_start}
            removes that week's target, ``weeks: []`` clears them all. Mark
            recovery weeks ``is_deload: true``. ``delete_blocks`` /
            ``delete_goal_ids`` delete (restorable with undo).

            Titles, descriptions, step labels and block name/focus describe
            that row only — never rules, preferences or reminders (durable
            guidance belongs in ``memory``). Derive target paces/zones from
            recent data (volume, HR zones, race predictions, VO2max). When the
            user sets a new target, reshape the anchor AND re-derive the
            near-term workouts in the same ``apply``.
            """
            if action == "apply":
                if spec is None:
                    raise ValueError("action 'apply' needs a 'spec' JSON string")
                try:
                    parsed = json.loads(spec)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"spec is not valid JSON: {exc}") from exc
                return training.apply(parsed)
            if action != "get":
                raise ValueError("action must be 'get' or 'apply'")
            return training.agenda(
                day=day, weeks=weeks, date_start=date_start, date_end=date_end,
                goal_id=goal_id, full=full, detail=detail,
            )


def build_agent(
    db: ReadOnlyDB,
    *,
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    model: Any = None,
    memory: _Memory | None = None,
    principles: _Principles | None = None,
    weather: Weather | None = None,
    training: _Training | None = None,
    charts: dict[str, Any] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Any:
    """Build the Pydantic AI agent wired to ``db`` tools.

    Pass ``model`` (e.g. ``TestModel``) to override the transport. ``provider``
    selects ``openai`` (default, any OpenAI-compatible ``base_url``) or
    ``gemini``. ``memory`` adds the ``memory`` tool and injects the stored
    facts; ``principles`` injects the athlete's training principles (read-only);
    ``training`` adds the ``training`` tool. ``charts`` is a per-request dict
    the ``chart`` tool fills with ``{sql: plotly figure dict}`` so the caller
    can ship each validated figure with the answer.
    """
    from pydantic_ai import Agent

    if model is None:
        model = _build_model(
            provider or "openai",
            model_name,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    # Instructions (not system prompts): re-evaluated on every run and never
    # stored in the history, so date/memory/principles are always current.
    agent = Agent(
        model,
        instructions=_render_system_prompt(
            _schema_text(db), has_training=training is not None
        ),
    )
    agent.instructions(lambda: _date_prompt(db))
    if memory is not None:
        agent.instructions(lambda: _memory_prompt(memory))
    if principles is not None:
        agent.instructions(lambda: _principles_prompt(principles))

    _register_tools(
        agent, db, weather=weather, memory=memory, training=training,
        charts=charts, on_status=on_status,
    )
    return agent


def _build_agent(
    cfg: dict[str, str],
    db: ReadOnlyDB,
    *,
    memory: Any | None = None,
    principles: Any | None = None,
    training: Any | None = None,
    charts: dict[str, Any] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Any:
    api_key = cfg["llm_api_key"] or None
    base_url = cfg["llm_base_url"] or None
    if not api_key and not base_url:
        raise RuntimeError(
            "no LLM configured: set OPENAI_API_KEY / LLM_API_KEY for cloud, "
            "or LLM_BASE_URL (e.g. http://host.docker.internal:11434/v1) with LLM_MODEL for a local Ollama model"
        )
    return build_agent(
        db,
        model_name=cfg["llm_model"],
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=cfg.get("llm_reasoning_effort") or None,
        provider=cfg.get("llm_provider") or None,
        memory=memory,
        principles=principles,
        weather=Weather.from_config(cfg),
        training=training,
        charts=charts,
        on_status=on_status,
    )


def _load_excluded_types(db_url: str, user_id: int) -> str:
    """The account's disabled data types (comma list) from the users table.

    Reads as the writer role (``db_url``), which can SELECT ``users``; the
    read-only agent role cannot. Returns '' when there is no DSN, no row, or a
    query fails — the agent then simply sees every column.
    """
    if not db_url:
        return ""
    try:
        with psycopg.connect(db_url) as conn:
            row = conn.execute(
                "SELECT excluded_data_types FROM users WHERE id = %s", (user_id,)
            ).fetchone()
            return (row[0] or "") if row else ""
    except psycopg.Error:
        return ""


def _open_readonly(cfg: dict[str, Any]) -> ReadOnlyDB:
    """Read-only handle for the configured backend (read-only PG role)."""
    url = cfg.get("readonly_db_url") or cfg.get("db_url")
    if not url:
        raise RuntimeError(
            "GARMIN_DB_URL (or GARMIN_READONLY_DB_URL) must be set — "
            "Postgres is the only supported backend"
        )
    user_id = cfg.get("local_user_id") or 1
    excluded = _load_excluded_types(cfg.get("db_url", ""), user_id)
    return ReadOnlyDB(url, user_id=user_id, excluded_types=excluded)


def _result_usage(result: Any) -> Any:
    """Return a run result's usage object (``result.usage`` is a method)."""
    usage = getattr(result, "usage", None)
    if callable(usage):
        try:
            usage = usage()
        except Exception:
            pass
    return usage


#: A final reply shorter than this many characters is treated as a possible
#: bare acknowledgement rather than a real answer (see ``_final_answer``).
_TRIVIAL_ANSWER_CHARS = 200
#: An earlier reply must be at least this long — and this many times longer than
#: the final one — before it is salvaged over it.
_MIN_SALVAGED_CHARS = 400
_SALVAGE_RATIO = 3


def _final_answer(result: Any) -> str:
    """Return the user-facing answer for a completed run, salvaging a lost reply.

    The final model response is normally the answer. But a model can emit its
    real (long) reply in the *same* response as a side-effect tool call — most
    often ``memory`` — and then finish with a bare acknowledgement once that
    tool returns ("Noted in memory…"). The server suppresses text that
    accompanies a tool call as pre-tool narration, so trusting
    ``result.output`` verbatim would drop the actual answer. When the final
    reply is trivially short and an earlier reply from *this run* is
    substantially longer, prefer that earlier reply.

    Only the run's own new messages are considered, so a long answer from an
    earlier turn in the resumed history can never be resurrected.
    """
    from pydantic_ai.messages import ModelResponse, TextPart

    final = str(getattr(result, "output", "") or "")
    texts = [
        str(part.content)
        for message in result.new_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart) and str(part.content).strip()
    ]
    if not texts:
        return final
    best = max(texts, key=len)
    final_len = len(final.strip())
    if (
        final_len < _TRIVIAL_ANSWER_CHARS
        and len(best) >= _MIN_SALVAGED_CHARS
        and len(best) >= _SALVAGE_RATIO * max(final_len, 1)
    ):
        return best
    return final


def _record_turn(
    cfg: dict[str, str],
    question: str,
    result: Any,
    *,
    trace_writer: Any,
    answer: str | None = None,
) -> None:
    """Append a trace entry for this turn via ``trace_writer`` (which persists
    the record to the per-user ``user_state`` ``trace`` key)."""
    from .trace import build_trace_record

    if answer is None:
        answer = _final_answer(result)
    record = build_trace_record(
        question,
        result.new_messages(),
        answer=answer,
        model=cfg.get("llm_model", ""),
        usage=_result_usage(result),
    )
    trace_writer(record)



def compact(agent: Any, messages: list[Any]) -> list[Any]:
    """Fold a conversation into a one-message summary history.

    The instructions are not part of the history, so the summary (as a user
    message) is all a new session needs. Returns ``messages`` unchanged when
    there is nothing to compact or the model returns an empty summary; a model
    failure propagates to the caller.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    if not messages:
        return messages
    summary = str(
        agent.run_sync(_COMPACT_INSTRUCTION, message_history=messages).output
    ).strip()
    if not summary:
        return messages
    return [ModelRequest(parts=[UserPromptPart(content=f"{_SUMMARY_LABEL}\n\n{summary}")])]


def _messages_token_estimate(messages: Iterable[Any]) -> int:
    """Rough token count for a conversation (≈ chars/4)."""

    def text(part: Any) -> str:
        for attr in ("content", "args"):
            value = getattr(part, attr, None)
            if value is not None:
                return value if isinstance(value, str) else json.dumps(value, default=str)
        return str(part)

    total = sum(len(text(p)) for msg in messages for p in getattr(msg, "parts", []))
    return max(1, round(total / 4))


#: Stored tool results longer than this are cut to ``_PRUNED_CHARS``.
_PRUNE_OVER_CHARS = 350
_PRUNED_CHARS = 250


def _prune_session_messages(messages: list[Any]) -> list[Any]:
    """Cut bulky tool results (SQL rows, schemas, agendas) in stored history.

    User prompts, assistant text and tool-call identities are kept intact; only
    a long ``ToolReturnPart`` is replaced by its head plus a marker, so resumed
    sessions stay light. The model re-queries when it needs the data again.
    """
    from dataclasses import replace

    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    def prune(part: Any) -> Any:
        if not isinstance(part, ToolReturnPart):
            return part
        content = part.content if isinstance(part.content, str) else str(part.content)
        if len(content) <= _PRUNE_OVER_CHARS:
            return part
        return replace(part, content=content[:_PRUNED_CHARS] + "… [output pruned]")

    return [
        replace(msg, parts=[prune(p) for p in msg.parts])
        if isinstance(msg, ModelRequest) else msg
        for msg in messages
    ]


def _open_state(cfg: dict[str, str]) -> tuple[ReadOnlyDB, Any, Any, int]:
    """The CLI's read-only handle, user state, training store and user id."""
    from .server.state import TrainingStore, UserState

    return (
        _open_readonly(cfg),
        UserState(cfg["db_url"]),
        TrainingStore(cfg["db_url"]),
        cfg.get("local_user_id") or 1,
    )


def _cli_agent(cfg: dict[str, str], db: ReadOnlyDB, state: Any, training: Any, user_id: int) -> Any:
    from .server.state import PgMemory, PgPrinciples, TrainingSeason

    return _build_agent(
        cfg,
        db,
        memory=PgMemory(state, user_id),
        principles=PgPrinciples(state, user_id),
        training=TrainingSeason(training, user_id),
    )


def _ask(cfg: dict[str, str], question: str) -> str:
    db, state, training, user_id = _open_state(cfg)
    try:
        result = _cli_agent(cfg, db, state, training, user_id).run_sync(question)
        answer = _final_answer(result)
        _record_turn(
            cfg, question, result, answer=answer,
            trace_writer=lambda r: state.append_trace(user_id, r),
        )
        return answer
    finally:
        db.close()
        state.close()
        training.close()


def _ask_session(cfg: dict[str, str]) -> None:
    """Interactive multi-turn session; history persists across questions.

    The conversation lives in Postgres (``user_state``, scoped to
    ``GARMIN_LOCAL_USER_ID``), so a later ``garmin-ask`` resumes where this one
    left off. ``/clear`` drops the history; ``/new`` folds it into a summary.
    """
    db, state, training, user_id = _open_state(cfg)
    try:
        agent = _cli_agent(cfg, db, state, training, user_id)
        history = state.get_session_messages(user_id) or []
        if history:
            print(f"Resumed {len(history)} prior message(s) from the database (user {user_id})")
        print(
            "Ask about your Garmin data, one question per line "
            "(exit/quit or EOF to leave; /clear starts fresh; "
            "/new compacts the context into a new session)."
        )
        while True:
            try:
                prompt = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            low = prompt.lower()
            if not prompt:
                continue
            if low in ("exit", "quit"):
                return
            if low == "/clear" or (low == "/new" and not history):
                history = []
                state.clear_session(user_id)
                print("Cleared — new session with no prior context.")
                continue
            try:
                if low == "/new":
                    n_before = len(history)
                    history = compact(agent, history)
                    state.set_session_messages(user_id, history)
                    print(f"Compacted {n_before} message(s) into {len(history)}.")
                    continue
                result = agent.run_sync(prompt, message_history=history or None)
            except Exception as exc:  # noqa: BLE001 - keep the session alive
                print(f"error: {exc}")
                continue
            answer = _final_answer(result)
            _record_turn(
                cfg, prompt, result, answer=answer,
                trace_writer=lambda r: state.append_trace(user_id, r),
            )
            history = _prune_session_messages(result.all_messages())
            state.set_session_messages(user_id, history)
            print(answer)
    finally:
        db.close()
        state.close()
        training.close()


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
