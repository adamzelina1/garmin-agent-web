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
from typing import Annotated, Any, Callable, Iterable, Protocol

import psycopg
import psycopg.rows
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import ModelRetry, RunContext

from .chat_format import strip_leaked_tool_call
from .config import load_config
from .parser import TYPE_COLUMNS
from .tool_args import AsList, Loose, compact_schema


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


#: Guard a statement is a read-only query and not a write. (No REPLACE:
#: Postgres has no such statement, and the word blocked the harmless
#: ``replace()`` string function.)
_SELECT_PREFIX = re.compile(r"^\s*(?:SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)
_WRITE_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REINDEX|VACUUM|"
    r"TRIGGER)\b",
    re.IGNORECASE,
)

#: Functions a query may not call. ``set_config`` could rewrite ``app.user_id``
#: (the Row-Level Security identity) mid-statement; the ``*_to_xml`` family and
#: ``ts_stat``/``ts_rewrite`` execute a SQL *string*, which would slip past the
#: table/keyword checks below; the rest stall or probe the server. This is the
#: statement-gate layer only — the database enforces the same thing for the
#: agent role (``setup_db.ensure_roles`` revokes EXECUTE on ``set_config``).
_BLOCKED_FUNCTIONS = re.compile(
    r"\b(set_config|(?:query|cursor|table|schema|database)_to_xml\w*|"
    r"ts_stat|ts_rewrite|dblink\w*|pg_sleep\w*|pg_read_\w+|pg_ls_\w+|"
    r"pg_stat_file|pg_advisory_\w+|pg_terminate_backend|pg_cancel_backend|"
    r"pg_notify)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 500

#: Hard ceiling on one agent/chart statement, enforced by Postgres.
_STATEMENT_TIMEOUT_MS = 20_000

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

#: Derived metrics ``get_metric_trend`` reads (``derived_metrics`` rows).
_DERIVED_TREND_METRICS = (
    "acwr", "acwr_acute_load", "acwr_chronic_load", "acwr_daily_load",
    "run_acwr", "run_acute_km", "run_chronic_km",
    "run_cadence_drift", "run_cadence", "run_gct_ms",
)
#: Daily columns the tool's description names. Any numeric ``daily_metrics``
#: column visible to the account works; these are the ones worth suggesting.
_DAILY_TREND_METRICS = (
    "resting_hr", "hrv_last_night_avg", "sleep_score", "sleep_time_hours",
    "avg_stress", "body_battery_highest", "body_battery_lowest", "vo2max",
    "total_steps", "total_distance_m", "weight_kg",
)
#: Most metrics one ``get_metric_trend`` call returns side by side.
_MAX_TREND_METRICS = 8

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
    "gear": "one row per equipment item (type, name, cumulative distance, activity count, retired); current snapshot",
    "devices": "one row per Garmin device (model name); current snapshot",
    "derived_metrics": (
        "TALL key/value table, one row per (calendar_date, metric), recomputed "
        "each sync. Its ONLY data columns are value and qualifier: every name "
        "below is a VALUE of the metric column, never a column of its own. "
        "'acwr' (ratio, qualifier Sweet Spot/Elevated/Danger/Detraining) plus "
        "its component rows 'acwr_acute_load' / 'acwr_chronic_load' / "
        "'acwr_daily_load'. RUNNING-SPECIFIC (running activities only, so "
        "cycling/swimming cannot inflate it): 'run_acwr' (foot-strike volume "
        "ratio) plus 'run_acute_km' / 'run_chronic_km' (km/day, 7d/28d EMA); "
        "'run_cadence_drift' (pace-normalised cadence z-score over the "
        "trailing 90d fit — negative = overstriding / worse mechanics, "
        "qualifier Form Up/Stable/Form Dropping/Breaking Down) plus "
        "'run_cadence' (spm PER LEG; total both-feet = 2x) and 'run_gct_ms'. "
        "'zone_pace_z1'..'zone_pace_z5' (dated today): learned running pace "
        "per HR zone in decimal min/km, qualifier JSON with sd_s / minutes / "
        "ratio to lactate-threshold pace. Put several metrics side by side "
        "with one row per day: SELECT calendar_date, MAX(value) FILTER (WHERE "
        "metric = 'run_acwr') AS run_acwr, MAX(value) FILTER (WHERE metric = "
        "'run_acute_km') AS acute_km FROM derived_metrics WHERE metric IN "
        "('run_acwr', 'run_acute_km') GROUP BY 1 ORDER BY 1"
    ),
    "training_workout": (
        "planned workouts; READ-ONLY via SQL — write only through "
        "training_apply. planned_date and activity_type associate workouts to the active "
        "block and goal (goal_id is an optional manual override). status is "
        "planned/completed/partial/skipped; target_* columns hold the prescribed pace/HR-zone/power"
    ),
    "training_goal": (
        "one row per long-term goal/target (an event such as a marathon or a "
        "half-marathon) over [start_date, target_date]; an account may hold many. read-only via "
        "SQL — write only through training_apply. sport is "
        "run/cycle/swim/strength/rest/other"
    ),
    "training_block": (
        "one row per periodized phase of a goal: name/focus are free text, "
        "start_date/end_date define the span (ordered chronologically by start_date), "
        "target_weekly_km provides an optional baseline weekly volume for weeks in this phase; "
        "read-only via SQL — write only through training_apply"
    ),
    "training_week": (
        "one row per (block_id, week_start): explicit targets for one calendar "
        "week (Mon..Sun) of a block — distance_km and/or duration_min, optional "
        "is_deload flag; falls back to block's target_weekly_km when distance_km is NULL. "
        "read-only via SQL — write only through training_apply"
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
        "vo2max": "precise VO2max estimate, forward-filled from the last one (see vo2max_date)",
        "vo2max_date": "date of the estimate vo2max is carried from",
        "weight_kg": "kg, forward-filled from the last weigh-in (see weight_date)",
        "weight_date": "date of the weigh-in weight_kg/bmi/body_fat_pct come from",
        "body_battery_change": "Body Battery charged DURING SLEEP (the night ending this day)",
        "body_battery_net_change": "net Body Battery change over the WHOLE day",
        "avg_stress": "0-100; NULL when Garmin had too little data",
        "day_of_goal_met": "weekday the weekly intensity-minutes goal was reached (e.g. 'Tue'), NULL if not yet",
        "resting_hr": "bpm; no 7-day avg stored — compute rolling averages yourself",
        "lactate_threshold_hr": "bpm, forward-filled",
        "lactate_threshold_speed_kmh": "km/h, forward-filled",
        "running_ftp_watts": "running FTP (watts), forward-filled",
        "cycling_ftp_watts": "cycling FTP (watts), forward-filled; absent without a power meter",
        "sweat_loss_ml": "estimated sweat loss (ml)",
    },
    "activity_summaries": {
        "activity_type": "Garmin typeKey (e.g. running, road_biking)",
        "distance_km": "km",
        "duration_hours": "hours",
        "elapsed_hours": "hours",
        "moving_hours": "hours; 0 for indoor activities without GPS — use duration_hours there",
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
        "weather_apparent_c": "feels-like degC; Garmin often just repeats weather_temp_c",
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
        "metric": "which derived metric the row holds (a name such as 'run_acwr' or 'run_acute_km' — see the table note); never a column",
        "value": "the metric's value; units differ per metric (acwr ratio, loads arbitrary, run_*_km km/day)",
        "qualifier": "category label: acwr = Sweet Spot/Elevated/Danger/Detraining; run_cadence_drift = Form Up/Stable/Form Dropping/Breaking Down",
    },
    "training_workout": {
        "activity_type": "run/cycle/swim/strength/rest/other",
        "duration_min": "planned minutes (computed from steps)",
        "distance_km": "planned km",
        "target_pace_min_km": "prescribed running pace as DECIMAL min/km (5.5 = 5:30)",
        "target_hr_zone": "prescribed HR zone/target, free text (e.g. 'Z2' or '140-150')",
        "target_power_w": "prescribed power in watts (int)",
        "steps": (
            "TEXT holding an ordered JSON array (cast `steps::jsonb` before any "
            "JSON function) of the session's segments (warm-up/work/"
            "recovery/cool-down, intervals, strides, ...); at least one for "
            "every non-rest workout (NULL only on rest days); stored as "
            "{kind: warmup/steady/work/recovery/cooldown/rest, duration_sec, "
            "repeat?, label?, intensity?, target_pace_min_km?, target_hr_zone?, "
            "target_power_w?}; label is a short segment name (no duration/rep). "
            "When WRITING via training_apply pass a friendly `duration` "
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

Common columns (exact names — write SQL with these directly and call
`table_schema` before using any column not listed; never guess a name):
{common}

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

- `get_day_summary` for one day; `get_metric_trend` for one or several daily
  metrics over time and against the athlete's normal level (readiness: sleep,
  HRV, resting HR, stress, load); `get_recent_activities` for a list of
  activities; `get_activity_detail` for how one session went (HR zones, laps,
  or one stretch of it). Otherwise write SQL with `run_sql`.
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
- `training` to read the training season (goals, blocks, weekly targets,
  planned workouts) — before building or reshaping a week and to answer "how
  is my week going?" — and `training_apply` to change it. To show a plan in chat, embed
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


class ChartTrace(BaseModel):
    """One trace of the ``chart`` tool. Any other key (marker, line, mode,
    orientation, labels, ...) passes to the Plotly trace as-is."""

    model_config = ConfigDict(extra="allow")

    type: str | None = Field(None, description="line|scatter|area|bar|pie|histogram|box")
    go: str | None = Field(
        None, description="instead of type: any plotly.graph_objects class (Scattergl, Violin, Heatmap, ...)"
    )
    x: str | None = Field(None, description="result column for x")
    y: str | None = Field(None, description="numeric result column for y")
    z: str | None = Field(None, description="result column for z (heatmaps)")


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
    ``SET LOCAL app.user_id`` so Row-Level Security scopes the transaction
    (plus a statement timeout), and executes as the read-only PG role (SELECT
    on the agent tables only). ``SET LOCAL`` rather than ``set_config`` because
    that role has no EXECUTE on ``set_config`` — a query must not be able to
    switch accounts. Connections are per-call because Pydantic AI runs tools
    from a worker thread.

    ``pool`` is an already-open pool to draw from (the server shares one across
    requests; safe because the identity is transaction-scoped). Without it the
    handle opens — and on ``close`` releases — its own.

    ``excluded_types`` is the account's disabled daily data types (a comma list
    string or an iterable): ``daily_metrics`` columns that no *enabled* type can
    write — plus columns that are (almost) empty for this account — are hidden
    from the schema and day summaries and rejected in SQL, so the agent never
    sees metrics the account turned off (even before the next sync prunes their
    stored values) or that its devices don't record.

    ``utc_offset`` is the account's UTC offset (``fetcher.conn_utc_offset``):
    the server runs in UTC, so ``today()`` — the agent's "today" in its
    instructions and trailing tool windows — is the user's calendar date,
    computed live so a long-lived handle never goes stale. None falls back to
    the machine's date.
    """

    def __init__(
        self,
        url: str,
        *,
        user_id: int | None = None,
        excluded_types: Iterable[str] | str = (),
        pool: Any = None,
        utc_offset: timedelta | None = None,
    ) -> None:
        self.url = url
        self.user_id = user_id
        self.utc_offset = utc_offset
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
        self._pool: Any = pool
        self._owns_pool = pool is None
        self._table_names: list[str] | None = None
        self._forbidden: re.Pattern | None = None
        self._hidden: frozenset[str] | None = None
        self._hidden_re: re.Pattern | None = None

    def today(self) -> date:
        """The account's current calendar date (see ``utc_offset``)."""
        from .fetcher import user_today

        return user_today(self.utc_offset)

    @contextmanager
    def _connect(self) -> Any:
        if self._pool is None:
            from .db import open_pg_pool

            self._pool = open_pg_pool(self.url, min_size=1, max_size=8)
        with self._pool.connection() as conn:
            # SET takes no bind parameters; both values are ints formatted here.
            conn.execute(f"SET LOCAL statement_timeout = {_STATEMENT_TIMEOUT_MS}")
            if self.user_id is not None:
                conn.execute(f"SET LOCAL app.user_id = '{int(self.user_id)}'")
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

    def zone_paces(self) -> dict[str, Any] | None:
        """The learned run pace per HR zone (see :mod:`.zone_pace`), or None."""
        from .zone_pace import read_zone_paces

        try:
            with self._connect() as conn:
                return read_zone_paces(conn)
        except psycopg.Error:
            return None

    def trend_metrics(self) -> list[str]:
        """Every metric ``metric_trend`` accepts for this account: the derived
        scores plus each numeric ``daily_metrics`` column it can see."""
        daily = [
            c["name"] for c in self.columns("daily_metrics")
            if c["type"] in _NUM_TYPE
        ]
        return list(_DERIVED_TREND_METRICS) + daily

    def suggested_trend_metrics(self) -> list[str]:
        """The trend metrics worth naming in the tool description."""
        present = set(self.trend_metrics())
        return [
            m for m in (*_DAILY_TREND_METRICS, *_DERIVED_TREND_METRICS)
            if m in present
        ]

    def metric_trend(
        self, metrics: list[str], days: int = 30, baseline_days: int = 60
    ) -> dict[str, Any]:
        """Daily values of up to ``_MAX_TREND_METRICS`` metrics side by side
        over the trailing ``days`` (today included), plus per-metric stats:
        min/max/avg/latest/change over the window, the last-7-day average and
        how it compares with the trailing ``baseline_days`` average (the
        athlete's normal level; 0 skips it)."""
        names = list(dict.fromkeys(m.strip().lower() for m in metrics if m and m.strip()))
        if not names:
            raise ValueError("pass at least one metric")
        if len(names) > _MAX_TREND_METRICS:
            raise ValueError(f"at most {_MAX_TREND_METRICS} metrics per call")
        supported = set(self.trend_metrics())
        unknown = [m for m in names if m not in supported]
        if unknown:
            raise ValueError(
                f"unsupported metric(s): {', '.join(unknown)}. Use a derived metric "
                f"({', '.join(_DERIVED_TREND_METRICS)}) or a numeric daily_metrics "
                "column (see table_schema), or write SQL with run_sql"
            )
        days = max(1, min(int(days), 365))
        baseline_days = max(0, min(int(baseline_days), 365))
        today = self.today()

        def since(n: int) -> str:
            return (today - timedelta(days=n - 1)).isoformat()

        window_from, week_from = since(days), since(7)
        read_from = since(max(days, baseline_days, 7))
        values: dict[str, dict[str, float]] = {}
        qualifiers: dict[str, str | None] = {}
        daily = [m for m in names if m not in _DERIVED_TREND_METRICS]
        derived = [m for m in names if m in _DERIVED_TREND_METRICS]
        with self._connect() as conn:
            if daily:
                # The names are information_schema column names (checked above).
                cols = ", ".join(f'"{m}"' for m in daily)
                for r in conn.execute(
                    f"SELECT calendar_date, {cols} FROM daily_metrics "
                    "WHERE calendar_date >= %s ORDER BY calendar_date",
                    (read_from,),
                ).fetchall():
                    for m in daily:
                        if isinstance(v := _jsonable(r[m]), (int, float)):
                            values.setdefault(r["calendar_date"], {})[m] = v
            if derived:
                for r in conn.execute(
                    "SELECT calendar_date, metric, value, qualifier FROM derived_metrics "
                    "WHERE metric = ANY(%s) AND calendar_date >= %s ORDER BY calendar_date",
                    (derived, read_from),
                ).fetchall():
                    if isinstance(v := _jsonable(r["value"]), (int, float)):
                        values.setdefault(r["calendar_date"], {})[r["metric"]] = v
                        qualifiers[r["metric"]] = r["qualifier"]

        def avg(xs: list[float]) -> float | None:
            return round(sum(xs) / len(xs), 2) if xs else None

        dates = sorted(d for d in values if d >= window_from)
        stats: dict[str, Any] = {}
        for m in names:
            window = [(d, values[d][m]) for d in dates if m in values[d]]
            if not window:
                stats[m] = {"count": 0}
                continue
            vals = [v for _, v in window]
            entry: dict[str, Any] = {
                "count": len(vals),
                "latest": vals[-1],
                "latest_date": window[-1][0],
                "min": min(vals),
                "max": max(vals),
                "avg": avg(vals),
                "avg_7d": avg([values[d][m] for d in values if d >= week_from and m in values[d]]),
            }
            if len(vals) >= 2:
                entry["change_vs_start"] = round(vals[-1] - vals[0], 2)
            if baseline_days:
                base = avg([values[d][m] for d in values if d >= since(baseline_days) and m in values[d]])
                entry[f"baseline_avg_{baseline_days}d"] = base
                if base is not None and entry["avg_7d"] is not None:
                    entry["avg_7d_vs_baseline"] = round(entry["avg_7d"] - base, 2)
            if qualifiers.get(m):
                entry["latest_qualifier"] = qualifiers[m]
            stats[m] = entry
        return {
            "days": days,
            "from": window_from,
            "stats": stats,
            "series": {
                "columns": ["date", *names],
                "rows": [[d, *(values[d].get(m) for m in names)] for d in dates],
            },
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
        start = date_start or (self.today() - timedelta(days=days)).isoformat()
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

    def _hr_zones(self, conn: Any, activity_type: str | None) -> list[tuple[int, float]]:
        """``[(zone, lower bpm), ...]`` for the activity's sport (falling back
        to Garmin's DEFAULT zones), or ``[]`` when none are configured."""
        kind = (activity_type or "").lower()
        sport = "RUNNING" if "run" in kind else (
            "CYCLING" if any(k in kind for k in ("cycl", "bik", "ride")) else "DEFAULT"
        )
        rows = {
            r["sport"]: r for r in conn.execute(
                "SELECT sport, zone1_min, zone2_min, zone3_min, zone4_min, zone5_min "
                "FROM hr_zones WHERE sport IN (%s, 'DEFAULT')",
                (sport,),
            ).fetchall()
        }
        row = rows.get(sport) or rows.get("DEFAULT")
        if not row:
            return []
        return [(z, row[f"zone{z}_min"]) for z in range(1, 6) if row[f"zone{z}_min"] is not None]

    def activity_detail(
        self,
        activity_id: int | None = None,
        bucket_s: int = 60,
        from_s: int | None = None,
        to_s: int | None = None,
    ) -> dict[str, Any]:
        """One activity in one call: its summary, time in HR zones, its laps,
        and its time series averaged into ``bucket_s``-second buckets of
        elapsed time (widened so there are at most ``_MAX_BUCKETS``).
        ``activity_id=None`` is the most recent activity. ``from_s``/``to_s``
        (seconds of elapsed time) narrow the series to one stretch and add a
        ``window`` summary of it, time in each HR zone included."""
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
            zone_pct = conn.execute(
                "SELECT " + ", ".join(f"hr_time_zone_{z}_pct AS z{z}" for z in range(1, 6))
                + " FROM activity_summaries WHERE activity_id = %s",
                (aid,),
            ).fetchone()
            zones = self._hr_zones(conn, row["activity_type"])
            span = conn.execute(
                "SELECT MAX(elapsed_s) AS span FROM activity_detail_series "
                "WHERE activity_id = %s",
                (aid,),
            ).fetchone()["span"] or 0
            lo = max(0, int(from_s)) if from_s is not None else 0
            hi = int(to_s) if to_s is not None else int(math.ceil(span))
            if (from_s is not None or to_s is not None) and hi <= lo:
                raise ValueError(
                    f"to_s must be after from_s (the activity spans 0..{span:g} s)"
                )
            bucket = max(5, int(bucket_s), math.ceil(max(hi - lo, 0) / _MAX_BUCKETS))
            series = conn.execute(
                "SELECT FLOOR(elapsed_s / %s) * %s AS t, AVG(heart_rate) AS hr, "
                "MAX(heart_rate) AS hr_max, AVG(speed_kmh) AS speed_kmh, "
                "AVG(cadence) AS cadence, AVG(power_w) AS power_w, "
                "AVG(elevation_m) AS elevation_m, MAX(distance_m) AS distance_m "
                "FROM activity_detail_series "
                "WHERE activity_id = %s AND elapsed_s BETWEEN %s AND %s "
                "GROUP BY 1 ORDER BY 1",
                (bucket, bucket, aid, lo, hi),
            ).fetchall()
            laps = conn.execute(
                "SELECT split_number, split_type, start_time_s, duration_s, "
                "distance_m, pace_sec_per_km, avg_hr, max_hr, avg_power, "
                "avg_cadence, elevation_gain_m FROM activity_splits "
                "WHERE activity_id = %s ORDER BY split_number",
                (aid,),
            ).fetchall()
            window = None
            if from_s is not None or to_s is not None:
                window = self._window_summary(conn, aid, lo, hi, zones)
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
        out: dict[str, Any] = {"activity": _format_activity(row)}
        if any(zone_pct[f"z{z}"] for z in range(1, 6)):
            # Garmin's own split of the whole activity, as Connect shows it.
            minutes = (row["duration_hours"] or 0) * 60
            bounds = dict(zones)
            out["hr_zones"] = [
                {
                    "zone": f"Z{z}",
                    **({"from_bpm": _round(bounds[z])} if z in bounds else {}),
                    "pct": _round(zone_pct[f"z{z}"] or 0, 1),
                    "min": _round((zone_pct[f"z{z}"] or 0) * minutes / 100, 1),
                }
                for z in range(1, 6)
            ]
        if window is not None:
            out["window"] = window
        out["laps"] = [
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
        ]
        out["series"] = {
            "bucket_s": bucket,
            "columns": columns,
            "rows": [[p[c] for c in columns] for p in points],
        }
        return out

    @staticmethod
    def _window_summary(
        conn: Any, aid: int, lo: int, hi: int, zones: list[tuple[int, float]],
    ) -> dict[str, Any]:
        """Totals and averages of the samples in ``[lo, hi]`` seconds, with
        minutes per HR zone (each sample weighted by the gap to the next one,
        capped so a pause doesn't count as time in a zone)."""
        zone_sql, params = "", []
        for i, (z, lower) in enumerate(zones):
            upper = zones[i + 1][1] if i + 1 < len(zones) else None
            cond = "heart_rate >= %s" + (" AND heart_rate < %s" if upper is not None else "")
            params += [lower] + ([upper] if upper is not None else [])
            zone_sql += f", SUM(dt) FILTER (WHERE {cond}) AS z{z}"
        if zones:
            zone_sql += ", SUM(dt) FILTER (WHERE heart_rate < %s) AS z0"
            params.append(zones[0][1])
        r = conn.execute(
            "WITH s AS (SELECT elapsed_s, heart_rate, cadence, power_w, distance_m, "
            "LEAST(LEAD(elapsed_s) OVER (ORDER BY elapsed_s) - elapsed_s, 10) AS dt "
            "FROM activity_detail_series WHERE activity_id = %s AND elapsed_s IS NOT NULL) "
            "SELECT MIN(elapsed_s) AS lo, MAX(elapsed_s) AS hi, "
            "MAX(distance_m) - MIN(distance_m) AS dist_m, AVG(heart_rate) AS hr, "
            "MAX(heart_rate) AS hr_max, AVG(NULLIF(cadence, 0)) AS cadence, "
            "AVG(power_w) AS power_w"
            + zone_sql
            + " FROM s WHERE elapsed_s BETWEEN %s AND %s",
            (aid, *params, lo, hi),
        ).fetchone()
        if r["lo"] is None:
            return {"from": _fmt_duration(lo), "to": _fmt_duration(hi), "samples": 0}
        seconds = float(r["hi"] - r["lo"])
        km = (r["dist_m"] or 0) / 1000.0
        out: dict[str, Any] = {
            "from": _fmt_duration(r["lo"]),
            "to": _fmt_duration(r["hi"]),
            "duration": _fmt_duration(seconds),
            "km": _round(km, 2) if km else None,
            "pace": _fmt_pace(seconds / 60.0 / km) if km else None,
            "avg_hr": _round(r["hr"]),
            "max_hr": _round(r["hr_max"]),
            "cadence_per_leg": _round(r["cadence"]),
            "power_w": _round(r["power_w"]),
        }
        if zones:
            out["hr_zone_min"] = {
                ("below Z1" if z == 0 else f"Z{z}"): _round((r[f"z{z}"] or 0) / 60.0, 1)
                for z in (0, *(z for z, _ in zones))
            }
        return out


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
        blocked = _BLOCKED_FUNCTIONS.search(statement)
        if blocked:
            raise ValueError(
                f"statement calls a function that is not allowed: {blocked.group(0)}"
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
                # Not echoing the statement: the model has it in its own call.
                raise QueryError(str(exc).strip()) from exc
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
        """Release the handle's own pool (no-op if never opened or shared)."""
        if self._pool is not None and self._owns_pool:
            self._pool.close()
            self._pool = None


class Weather:
    """Stateless Open-Meteo access (forecast endpoint + historical archive).

    The forecast endpoint also serves the last ``PAST_DAYS`` days, so any range
    from three months back to ``FORECAST_DAYS`` ahead is one request, today
    included; only older ranges go to the archive (which also lags a few days
    behind, so recent days come back empty there).

    Every call is one small HTTP request; nothing is cached or stored here, so
    the agent's ``weather`` tool can never be a source of truth — only context
    for the stored data. The sync also uses it to refresh ``weather_forecast``.
    Coordinates come from a configured home location and can be overridden per
    request (both ``lat`` and ``lon`` together).
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    MAX_DAYS = 92
    PAST_DAYS = 92
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
        today: date | None = None,
    ) -> dict[str, Any]:
        """Return per-day weather for ``date_start``..``date_end`` (inclusive).

        Up to ``PAST_DAYS`` back and ``FORECAST_DAYS`` ahead comes from the
        forecast endpoint (one request, a range may span today); older ranges
        from the historical archive. With no dates the forecast from today
        onward is returned. Raises
        ``ValueError`` with a model-facing message on invalid input. ``today``
        is the user's date (default: the machine's).
        """
        import httpx

        lat, lon = self._resolve(lat, lon)
        today = today or date.today()
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
        forecast_end = today + timedelta(days=self.FORECAST_DAYS - 1)
        if end > forecast_end:
            raise ValueError(
                f"the forecast only covers up to {forecast_end.isoformat()} "
                f"({self.FORECAST_DAYS} days)"
            )
        if start >= today - timedelta(days=self.PAST_DAYS):
            url = self.FORECAST_URL
            # past/forecast_days are exclusive with start/end dates, so ask for
            # the span around today and trim to the range below.
            params["past_days"] = max(0, (today - start).days)
            params["forecast_days"] = max(1, (end - today).days + 1)
        elif end < today:
            url = self.ARCHIVE_URL
            params["start_date"] = start.isoformat()
            params["end_date"] = end.isoformat()
        else:
            raise ValueError(
                f"a range starting over {self.PAST_DAYS} days ago must end "
                "before today; split it at today"
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
            "source": "historical" if url == self.ARCHIVE_URL else "forecast",
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


#: The prompt's common columns per table, each with an optional short hint.
#: Rendered against the live schema (``_common_columns_text``), so a column
#: that is hidden for the account or doesn't exist yet is never advertised —
#: the model otherwise writes SQL against names it was told exist.
_COMMON_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "daily_metrics": (
        ("calendar_date", ""), ("resting_hr", "bpm"),
        ("hrv_last_night_avg", "ms"), ("sleep_score", "0-100"),
        ("sleep_time_hours", ""), ("vo2max", ""), ("total_steps", ""),
        ("total_distance_m", "METRES"), ("body_battery_highest", ""),
        ("body_battery_lowest", ""), ("avg_stress", ""), ("weight_kg", ""),
    ),
    "activity_summaries": (
        ("activity_id", ""), ("activity_name", ""),
        ("activity_type", "running, cycling, ..."), ("start_date", ""),
        ("duration_hours", ""), ("distance_km", "KM"), ("avg_hr", ""),
        ("max_hr", ""), ("pace_min_km", "decimal min/km; 5.5 = 5:30"),
        ("avg_cadence", "PER LEG"), ("avg_power_w", ""), ("training_load", ""),
        ("elevation_gain_m", ""), ("weather_temp_c", ""),
    ),
    "activity_splits": (
        ("activity_id", ""), ("split_number", "0-based"), ("split_type", ""),
        ("distance_m", "METRES"), ("duration_s", "SECONDS"),
        ("pace_sec_per_km", "seconds/km"), ("avg_hr", ""), ("avg_power", "W"),
        ("avg_cadence", "PER LEG"), ("elevation_gain_m", ""),
    ),
    "derived_metrics": (
        ("calendar_date", ""),
        ("metric", "a NAME such as 'run_acwr' — no per-metric columns"),
        ("value", ""), ("qualifier", ""),
    ),
    "weather_forecast": (
        ("calendar_date", ""), ("temp_max_c", ""), ("temp_min_c", ""),
        ("precip_mm", ""), ("wind_max_kmh", ""), ("condition_code", ""),
    ),
}


def _common_columns_text(db: ReadOnlyDB) -> str:
    """The prompt's common-column bullets, limited to columns that exist and
    are visible for this account."""
    lines: list[str] = []
    tables = set(db.tables())
    for table, cols in _COMMON_COLUMNS.items():
        if table not in tables:
            continue
        live = {c["name"] for c in db.columns(table)}
        names = [
            f"`{name}` ({hint})" if hint else f"`{name}`"
            for name, hint in cols if name in live
        ]
        if names:
            lines.append(f"- **{table}**: " + ", ".join(names))
    return "\n".join(lines)


def _schema_text(db: ReadOnlyDB) -> str:
    """The agent-facing *table overview*: one ``_TABLE_NOTES`` line per table.

    Columns are fetched on demand through ``table_schema``, so the prompt stays
    lean and never goes stale as columns are added dynamically.
    """
    return "\n".join(
        f"- {table}: {_TABLE_NOTES.get(table, 'see table_schema')}"
        for table in db.tables()
    ) or "(no tables available in the database)"


def _render_system_prompt(
    overview: str, *, common: str = "", has_training: bool
) -> str:
    """Fill the static instructions for the tools this agent actually has, so
    the prompt never advertises a tool the model cannot call."""
    return _PROMPT_TEMPLATE.format(
        overview=overview,
        common=common,
        training=_TRAINING_BULLET if has_training else "",
    ).strip()


def _date_prompt(db: ReadOnlyDB) -> str:
    """Today's date (with weekday, which models otherwise miscompute), the
    stored data span and, when the data lags today, a staleness warning."""
    today = db.today()
    first, last = db.data_span()
    span = f"{first}..{last}" if first and last else "nothing yet (no data synced)"
    text = _DATE_PROMPT.format(today=today.strftime("%Y-%m-%d (%A)"), span=span)
    if last is not None and (gap := (today - last).days) > 0:
        text += _STALENESS_PROMPT.format(data_until=last.isoformat(), gap=gap)
    return text


def _zone_pace_prompt(db: ReadOnlyDB) -> str | None:
    """The user's learned run pace per HR zone, one line, re-read every run so
    it follows each sync. None until a sync has produced zone paces."""
    paces = db.zone_paces()
    if not paces:
        return None
    parts = []
    for z, info in sorted(paces["zones"].items()):
        lo, hi = info.get("hr_min"), info.get("hr_max")
        bpm = f"{lo:g}-{hi:g}" if lo and hi else (f"<{hi + 1:g}" if hi else f">={lo:g}" if lo else "")
        m, s = divmod(round(info["pace_min_km"] * 60), 60)
        parts.append(f"Z{z} {bpm} bpm ~{m}:{s:02d}/km")
    return (
        f"The user's run pace by HR zone (learned from their recent runs, as of "
        f"{paces['as_of']}): {' · '.join(parts)}. Use these for any pace, time or "
        "distance reasoning about their running."
    )


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


#: Most characters of JSON one ``run_sql`` result may put in the context
#: (~7.5k tokens); 500 wide rows would otherwise flood a single turn.
_MAX_RESULT_CHARS = 30_000


def _fit_result(result: dict[str, Any]) -> dict[str, Any]:
    """Cut a ``run_sql`` result's rows to fit ``_MAX_RESULT_CHARS``, saying so.

    Only for what the model reads: the chart tool needs every row.
    """
    size = len(json.dumps(result, ensure_ascii=False))
    rows = result["rows"]
    if size <= _MAX_RESULT_CHARS or not rows:
        return result
    keep = max(1, len(rows) * _MAX_RESULT_CHARS // size)
    while keep > 1 and len(json.dumps(rows[:keep], ensure_ascii=False)) > _MAX_RESULT_CHARS:
        keep = max(1, keep * 9 // 10)
    return {
        **result,
        "rows": rows[:keep],
        "truncated": True,
        "note": (
            f"cut to the first {keep} of {len(rows)} rows to fit the output "
            "limit — aggregate, select fewer columns or narrow the range"
        ),
    }


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
    """Register every agent tool. Only ``memory`` and ``training_apply`` write, each
    through its own app store — never through the read-only connection."""

    async def compact(ctx: RunContext[None], tool_def: Any) -> Any:
        """Send the tool's schema without pydantic's optional-field noise."""
        from dataclasses import replace

        return replace(
            tool_def, parameters_json_schema=compact_schema(tool_def.parameters_json_schema)
        )

    def tool(status: str, **options: Any) -> Callable[[Callable[..., Any]], Any]:
        """Register ``fn`` as a plain tool that reports ``status``, returns a
        dict/list as JSON and turns an expected failure into ``ERROR: ...``.
        Typed arguments that fail validation go back to the model as a retry
        (``retries`` per tool, as pydantic-ai counts them)."""

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

            return agent.tool_plain(prepare=compact, **options)(wrapper)

        return register

    @tool("Inspecting table schema…")
    def table_schema(tables: Annotated[list[str], AsList]) -> Any:
        """Return the live columns (name, type, description each) of one or
        more tables, keyed by table.

        Call this before querying a column that is not in the prompt's common
        column list. Columns are read live, so newly-added ones appear.

        Args:
            tables: table names, e.g. ["activity_splits", "hr_zones"].
        """
        out: dict[str, Any] = {}
        for table in dict.fromkeys(t.strip() for t in tables if t.strip()):
            try:
                out[table] = _column_descriptions(db, table)
            except ValueError as exc:
                out[table] = _tool_error(exc)
        return out

    @tool("Running database query…")
    def run_sql(sql: str) -> Any:
        """Run one read-only SQL query (SELECT / WITH / EXPLAIN only) and return
        its columns and rows as JSON. Results are capped at 500 rows (and at
        an output size limit: aggregate or pick fewer columns when cut)."""
        return _fit_result(db.run_sql(sql))

    @tool("Fetching day summary…")
    def get_day_summary(calendar_date: str) -> Any:
        """Return everything stored for one date (YYYY-MM-DD): the day's
        non-null daily_metrics values, its activities and its running-form
        scores. Use it for "how was my day on X?" instead of multi-table SQL."""
        return db.day_summary(calendar_date)

    @tool(
        "Analyzing metric trends…",
        description=(
            f"Return up to {_MAX_TREND_METRICS} metrics side by side, one row per "
            "day over the trailing `days` (default 30, max 365), plus per-metric "
            "stats: latest, min/max/avg, change over the window, the last-7-day "
            "average and its difference from the trailing `baseline_days` "
            "average (default 60 — the athlete's normal level; 0 skips it). Use "
            "it for trends and for readiness (sleep, HRV, resting HR, stress, "
            "load vs normal) instead of SQL. Metrics: "
            + ", ".join(db.suggested_trend_metrics())
            + ", or any other numeric daily_metrics column."
        ),
    )
    def get_metric_trend(
        metrics: Annotated[list[str], AsList], days: int = 30, baseline_days: int = 60
    ) -> Any:
        return db.metric_trend(metrics, days, baseline_days)

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
    def get_activity_detail(
        activity_id: int | None = None,
        bucket_s: int = 60,
        from_s: int | None = None,
        to_s: int | None = None,
    ) -> Any:
        """Return one activity in full: its summary, Garmin's time in each HR
        zone (with the zone's lower bpm), its laps (auto-laps or manual lap
        presses, with pace/HR/cadence/power each) and its time series averaged
        into buckets of elapsed time. Use it for any question about how a
        single session went — intervals, pacing, HR drift, fade, zones — before
        reaching for activity_detail_series in SQL.

        Args:
            activity_id: the activity; omit for the most recent one.
            bucket_s: series bucket in seconds (default 60, widened to at most
                200 buckets); smaller (e.g. 15) resolves short efforts.
            from_s: start of a stretch to analyse, in seconds of elapsed time.
                With from_s and/or to_s the series covers only that stretch and
                a `window` summary is added (duration, km, pace, HR, cadence,
                power, minutes per HR zone) — e.g. from_s=120, to_s=1800 for
                minutes 2-30.
            to_s: end of that stretch (default the end of the activity).
        """
        return db.activity_detail(activity_id, bucket_s, from_s, to_s)

    @tool("Validating chart…", retries=3)
    def chart(
        sql: str,
        traces: Annotated[list[ChartTrace], Loose],
        layout: Annotated[dict[str, Any] | None, Loose] = None,
    ) -> Any:
        """Validate a chart and return its spec ready to embed.

        The query is run and the Plotly figure built to check it; on success
        the reply is ``OK: <spec>`` — embed that spec verbatim in
        <chart> ... </chart>. The data itself is never returned; the UI reruns
        the query to draw it. Aggregate to at most ~200 points.

        Args:
            sql: read-only query returning the data.
            traces: one entry per trace, naming result columns in x / y / z.
                Any other Plotly trace argument can take column data as
                {"column": "<name>"} (e.g. pie labels/values).
            layout: optional Plotly layout, e.g. {"title": {"text": "..."}}.
        """
        parsed: dict[str, Any] = {
            "sql": sql,
            "traces": [t.model_dump(exclude_none=True) for t in traces],
        }
        if layout:
            parsed["layout"] = layout
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

            ``date_start``/``date_end`` are inclusive YYYY-MM-DD bounds, any
            range up to 92 days long from the past to 16 days ahead (it may
            span today); with neither, the forecast from today. Coordinates
            default to the athlete's home; override with both ``lat`` and
            ``lon``. Context only — never the source of an answer.
            """
            return weather.query(date_start, date_end, lat, lon, today=db.today())


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
        # Imported here: ``server`` imports this module, so it cannot be
        # imported at load time (see ``training_tools``).
        from .server.state.training_tools import training_tools

        read, apply = training_tools(training)
        tool("Consulting the training season…", name="training")(read)
        tool("Updating the training plan…", name="training_apply", retries=3)(apply)


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
    ``training`` adds the ``training`` / ``training_apply`` tools. ``charts`` is a per-request dict
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
    # stored in the history, so date/zone paces/memory/principles are always
    # current.
    agent = Agent(
        model,
        instructions=_render_system_prompt(
            _schema_text(db),
            common=_common_columns_text(db),
            has_training=training is not None,
        ),
    )
    agent.instructions(lambda: _date_prompt(db))
    agent.instructions(lambda: _zone_pace_prompt(db))
    if memory is not None:
        agent.instructions(lambda: _memory_prompt(memory))
    if principles is not None:
        agent.instructions(lambda: _principles_prompt(principles))

    _register_tools(
        agent, db, weather=weather, memory=memory, training=training,
        charts=charts, on_status=on_status,
    )

    @agent.output_validator
    def _reject_leaked_tool_call(ctx: RunContext[None], output: str) -> str:
        """Send a reply that spelled a tool call out as text back to the model.

        Such a call never ran (typically a ``memory`` write tacked onto the
        answer), so the model gets one chance to make it properly. If it leaks
        again the markup is simply cut — a cosmetic slip must not fail the turn.
        """
        if ctx.partial_output or not isinstance(output, str):
            return output
        clean, leaked = strip_leaked_tool_call(output)
        if not leaked:
            return output
        if ctx.retry < ctx.max_retries:
            raise ModelRetry(
                "Your reply ended with a tool call written out as text (markup "
                "such as `<memory …>` / DSML tags), so that call was NOT "
                "executed. If you still want it, make it now through the "
                "function-calling interface. "
                + (
                    "The text before the markup has already been shown to the "
                    "user: do not repeat it — after the call, finish with one "
                    "short sentence."
                    if clean.strip()
                    else "Then write your reply to the user."
                )
                + " Never write tool calls in the reply text."
            )
        return clean

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


def _load_account(db_url: str, user_id: int) -> tuple[str, timedelta | None]:
    """The account's disabled data types (comma list) and UTC offset.

    Reads as the writer role (``db_url``), which can SELECT ``users`` and the
    raw payloads the offset comes from; the read-only agent role cannot.
    Returns ``("", None)`` when there is no DSN, no row, or a query fails — the
    agent then simply sees every column and the machine's date.
    """
    if not db_url:
        return "", None
    from .fetcher import conn_utc_offset

    try:
        with psycopg.connect(db_url) as conn:
            row = conn.execute(
                "SELECT excluded_data_types FROM users WHERE id = %s", (user_id,)
            ).fetchone()
            conn.execute("SELECT set_config('app.user_id', %s, true)", (str(user_id),))
            return ((row[0] or "") if row else ""), conn_utc_offset(conn)
    except psycopg.Error:
        return "", None


def _open_readonly(cfg: dict[str, Any]) -> ReadOnlyDB:
    """Read-only handle for the configured backend (read-only PG role)."""
    url = cfg.get("readonly_db_url") or cfg.get("db_url")
    if not url:
        raise RuntimeError(
            "GARMIN_DB_URL (or GARMIN_READONLY_DB_URL) must be set — "
            "Postgres is the only supported backend"
        )
    user_id = cfg.get("local_user_id") or 1
    excluded, offset = _load_account(cfg.get("db_url", ""), user_id)
    return ReadOnlyDB(url, user_id=user_id, excluded_types=excluded, utc_offset=offset)


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

    A reply that ended with a tool call written out as text (see
    ``strip_leaked_tool_call``) is a second way to lose the answer: the model
    is sent back to make the call properly and then finishes with a short
    confirmation. The prose of that first reply is the answer, unless the model
    rewrote it at greater length. The leaked markup itself is never returned.

    Only the run's own new messages are considered, so a long answer from an
    earlier turn in the resumed history can never be resurrected.
    """
    from pydantic_ai.messages import ModelResponse, TextPart

    final, _ = strip_leaked_tool_call(str(getattr(result, "output", "") or ""))
    texts: list[str] = []
    rescued: list[str] = []
    for message in result.new_messages():
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if not isinstance(part, TextPart):
                continue
            clean, leaked = strip_leaked_tool_call(str(part.content))
            if clean.strip():
                texts.append(clean)
                if leaked:
                    rescued.append(clean)
    if rescued:
        best_rescued = max(rescued, key=len)
        if len(best_rescued) > len(final.strip()):
            return best_rescued
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

#: Rough token budget (see ``_messages_token_estimate``) for the stored
#: conversation. Every model request re-sends it, so past this the oldest
#: whole turns are dropped.
_HISTORY_TOKEN_BUDGET = 30_000


def _trim_to_budget(messages: list[Any], budget: int) -> list[Any]:
    """Drop the oldest whole turns until ``messages`` fits ``budget`` tokens.

    A turn starts at a request carrying the user's prompt, so the kept history
    still opens with a user message and never splits a tool call from its
    result. The newest turn is always kept, even when it alone is over budget.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    starts = [
        i for i, msg in enumerate(messages)
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in msg.parts)
    ]
    if not starts:
        return messages
    sizes = [_messages_token_estimate([msg]) for msg in messages]
    tail = sum(sizes[starts[-1]:])
    keep_from = starts[-1]
    for start, nxt in zip(reversed(starts[:-1]), reversed(starts[1:])):
        tail += sum(sizes[start:nxt])
        if tail > budget:
            break
        keep_from = start
    return messages[keep_from:] if keep_from else messages


def _prune_session_messages(messages: list[Any]) -> list[Any]:
    """Slim the stored history that every later model request re-sends.

    User prompts, assistant text and tool-call identities are kept intact; a
    long ``ToolReturnPart`` (SQL rows, schemas, agendas) is replaced by its
    head plus a marker, so resumed sessions stay light. The model re-queries
    when it needs the data again.

    The model's reasoning (``ThinkingPart``) is dropped: it only matters while
    its own turn is still calling tools, yet pydantic-ai would send every past
    turn's reasoning back to the provider on every request — most of the
    history's tokens. A tool call the model wrote out as text is cut from
    assistant replies too (see ``strip_leaked_tool_call``): left in the history
    it would be shown again on reload and would teach the model to repeat the
    mistake. Finally the oldest turns go once the whole is over
    ``_HISTORY_TOKEN_BUDGET``.

    Idempotent, so it is also applied to a stored session when it is loaded.
    """
    from dataclasses import replace

    from pydantic_ai.messages import (
        ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolReturnPart,
    )

    def prune(part: Any) -> Any:
        if not isinstance(part, ToolReturnPart):
            return part
        content = part.content if isinstance(part.content, str) else str(part.content)
        if len(content) <= _PRUNE_OVER_CHARS:
            return part
        return replace(part, content=content[:_PRUNED_CHARS] + "… [output pruned]")

    def unleak(part: Any) -> Any:
        if not isinstance(part, TextPart):
            return part
        clean, leaked = strip_leaked_tool_call(part.content)
        return replace(part, content=clean) if leaked else part

    def clean_message(msg: Any) -> Any:
        if isinstance(msg, ModelRequest):
            return replace(msg, parts=[prune(p) for p in msg.parts])
        if isinstance(msg, ModelResponse):
            parts = [unleak(p) for p in msg.parts if not isinstance(p, ThinkingPart)]
            if len(parts) != len(msg.parts) or any(
                new is not old for new, old in zip(parts, msg.parts)
            ):
                return replace(msg, parts=parts)
        return msg

    return _trim_to_budget(
        [clean_message(msg) for msg in messages], _HISTORY_TOKEN_BUDGET
    )


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
        training=TrainingSeason(training, user_id, today=db.today),
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
