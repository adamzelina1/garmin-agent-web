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
from argparse import ArgumentParser
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Iterable, Protocol

import psycopg

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
    ``server.state.TrainingSeason``).

    One tree — goal → block → week → workout — with one write surface. The
    agent talks to it through a single ``training`` tool (``action=get`` /
    ``action=apply``) instead of separate plan/anchor/week tools.
    ``resolve_goal`` is the single composition of the goal/block/week
    resolution; ``apply`` takes one season spec (an ``anchor`` and/or ``plan``
    section, or ``undo``)."""

    def list_goals(self) -> list[dict[str, Any]]: ...

    def resolve_goal(
        self, goal_id: int, day: str | None = None
    ) -> dict[str, Any] | None: ...

    def blocks(self, goal_id: int) -> list[dict[str, Any]]: ...

    def coverage_range(
        self, date_start: str, date_end: str, goal_id: int | None = None
    ) -> list[dict[str, Any]]: ...

    def activities(
        self, date_start: str | None = None, date_end: str | None = None
    ) -> list[dict[str, Any]]: ...

    def list_workouts(
        self, date_start: str | None = None, date_end: str | None = None
    ) -> list[dict[str, Any]]: ...

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]: ...

    def can_undo(self) -> bool: ...


#: Guard a statement is a read-only query and not a write.
_SELECT_PREFIX = re.compile(r"^\s*(?:SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)
_WRITE_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REINDEX|VACUUM|"
    r"REPLACE|TRIGGER)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 500

#: Training-tool guardrail: the workout list is capped at this safety limit.
_MAX_WORKOUTS = 200

#: Preview length for a workout description when ``detail=False``: the text is
#: kept (so no ``has_description`` + second call round-trip) but capped so a long
#: description never bloats the context.
_DESC_PREVIEW_CHARS = 160

#: Approximate token budget for a stored conversation. When a session's history
#: (system prompt + every prior turn, including raw tool results) grows past this
#: it is folded into a compact summary so the next request stays cheap. The
#: static system prompt is re-sent on *every* request regardless (chat APIs are
#: stateless), so this is the real lever for long conversations. A rough 4
#: chars/token estimate is used — the budget is a trigger, not a precise count.
_AUTO_COMPACT_MAX_TOKENS = 200000

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

#: Data-driven schema annotations: table-level overviews and per-column
#: unit/semantic hints. ``_schema_text`` renders the agent-facing *table
#: overview* from ``_TABLE_NOTES`` (the prompt deliberately lists NO columns —
#: those are fetched dynamically via ``table_schema``, so auto-created columns
#: are always picked up); ``_COLUMN_DOCS`` feeds ``table_schema``'s per-column
#: descriptions. A schema change needs at most one dict entry here — never a
#: prose edit that can drift out of sync.
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
        "one row per intra-activity tick (activity_id + tick): HR, cadence, "
        "power, speed, elevation, cumulative distance, respiration. Prefer "
        "aggregate/interval queries; a metric may be NULL on every tick if the "
        "device didn't record it"
    ),
    "activity_splits": (
        "one row per split/lap (activity_id + split_number): distance, duration, "
        "pace, start time, HR/power/cadence, elevation gain (work vs rest)"
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
        "ts_ms": "epoch ms",
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
        "split_type": "'distance' (work) or 'rest'",
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

#: The agent's system prompt, rendered as one structured Markdown document so the
#: routing and formatting guidance reads clearly. Two values are injected:
#:   * ``{overview_text}`` — the per-user *table overviews* from ``_schema_text``
#:     (table-level descriptions only, NO columns), filled ONCE when the agent is
#:     built. Columns are introspected on demand via the ``table_schema`` tool.
#:   * today's date + data freshness — NOT this template: it comes from
#:     ``_current_date_prompt`` (below), a *dynamic* prompt re-evaluated on every
#:     turn so a cached agent or a resumed session can never keep a stale date
#:     or answer "today" questions from days-old rows.
#: The template contains literal JSON braces (``{"sql": ...}``) in the chart
#: guidance, so it is filled with ``str.replace``, never ``str.format``.
_PROMPT_TEMPLATE = """\
# Role & Environment

You are an expert sports scientist and data analyst with read-only access to the
user's personal Garmin health and fitness database (PostgreSQL).

- **Access Level:** Read-only SQL (`run_sql`). Writes are limited to long-term
  memory (`memory`) and the planning tools when they are available.
- **Be genuinely helpful:** give thorough, accurate, insight-driven answers the
  user can act on. Connect metrics, flag trends, anomalies and relationships, and
  add concise context rather than just returning raw numbers.

---

# Database & SQL Guidelines

Only a high-level overview of each table is listed below — no columns, so the
prompt stays lean. Call `table_schema` before querying a table to see its live
columns and a short description of each, then SELECT only what you need.

{overview_text}

### Core Columns Quick Reference (common columns — write SQL immediately without calling table_schema):
- **daily_metrics**: `calendar_date`, `resting_hr` (bpm), `hrv_last_night_avg` (ms), `sleep_score` (0-100), `sleep_time_hours` (hours), `vo2max`, `total_steps`, `total_distance_m` (METRES, not km!), `body_battery_max/min`, `stress_avg`, `weight_kg`
- **activity_summaries**: `activity_id`, `activity_name`, `activity_type` (running, cycling, ...), `start_date`, `duration_hours`, `distance_km` (KM), `avg_hr`, `max_hr`, `pace_min_km` (decimal min/km; 5.5 = 5:30), `avg_cadence` (PER LEG; 2x for total spm), `avg_power_w`, `training_load`, `elevation_gain_m`, `weather_temp_c`
- **derived_metrics**: `calendar_date`, `metric` ('acwr', 'run_acwr', 'run_cadence_drift'), `value`, `qualifier`
- **weather_forecast**: `calendar_date`, `temp_max_c`, `temp_min_c`, `precip_mm`, `wind_max_kmh`, `condition_code`
Call `table_schema` only when you need other columns or want to inspect a table's full schema.

### Query rules
1. PostgreSQL dialect: use `DATE_TRUNC`, `INTERVAL '7 days'`, etc.
2. Never `SELECT *`; pick only the columns you need.
3. Results are capped at 500 rows — aggregate (GROUP BY week/month/sport) and
   `ORDER BY` time columns chronologically; `LIMIT 10` to probe first.
4. Almost any column may be NULL when a value is N/A; aggregate over the rows you
   have and wrap denominators in `NULLIF(col, 0)`.
5. For a single day ("how was my day on X?"), call `get_day_summary(YYYY-MM-DD)`
   instead of multi-table SQL — it returns the whole day in one call. Otherwise
   write the SQL yourself.
6. Date-like columns (`calendar_date`, `start_date`, `start_time_local`,
   `sleep_start_local`, `sleep_end_local`, …) are stored as TEXT in
   `'YYYY-MM-DD'` / `'HH:MM'` form — you cannot pass them straight to
   `DATE_TRUNC`/`EXTRACT` (that throws "function date_trunc(unknown, text) does
   not exist"). Cast with `::date` first (`calendar_date::date`) or compare
   ranges lexicographically (`'2026-08-01' <= calendar_date <= '2026-08-31'`).
   Check a column's exact name via `table_schema` before writing SQL if unsure —
   a wrong name fails the whole query.
7. Explore freely. Start with a small `LIMIT` to check a table's shape, then
   refine. Running a corrective or additional query is fine — accuracy matters
   more than query count.

---

# Tool usage

- **`get_metric_trend(metric, days=30)`** — high-level shortcut for core metric
  trends (acwr, run_acwr, run_cadence_drift, sleep_score, resting_hr,
  hrv_last_night_avg, vo2max, total_steps, etc.). Computes min, max, avg, and
  trend change in one call without writing SQL.
- **`get_recent_activities(sport=None, days=30, limit=10, date_start=None, date_end=None)`** —
  high-level
  shortcut for recent activities. Automatically formats pace (MM:SS /km), 2x
  total running cadence, duration, elevation, and load so you don't have to write
  multi-column SQL or do pace math. Pass `date_start`/`date_end` (Y-M-D) to scope
  to an explicit date range instead of a trailing window.
{query_guidance}
- **`chart`** — when asked for a chart, build a spec
  `{"sql": "...", "traces": [...], "layout": {"title": {"text": "..."}}}` whose
  `sql` returns the data; the `chart` tool validates it. On `"OK: <spec>"`, embed
  the spec verbatim in `<chart> ... </chart>` with one descriptive sentence (do
  NOT paste query data or Python). Trace columns are referenced by name in
  x/y/z or `{"column": "<name>"}`; aggregate to ≤200 points.
- **`weather`** — prefer the historical weather columns on `activity_summaries`
  when it's about a stored activity. Use `weather` only for a forecast, a day
  with no activity, or explicit lat/lon. Never a substitute for a stored value.
- **`memory`** — durable facts the user volunteers (goals, preferences, habits,
  injuries, equipment, constraints). HOLD A HIGH BAR: only record a fact that is
  BOTH durable (still true and useful months from now, in unrelated future
  conversations) AND high-ROI (it changes how you coach, not just colour for one
  reply). If it is transient or situational — a one-off question, a bad night's
  sleep, this week's soreness or schedule, a passing mood, anything about today's
  plan — do NOT store it. When in doubt, don't: an empty slot costs nothing,
  a stale fact misleads every future turn. `action="set"` upserts (overwrite the
  same key on change — never add a near-duplicate); `action="forget"` deletes
  facts that are wrong, stale or superseded; `action="replace"` rewrites the
  whole profile, which is how you consolidate it. Keep the profile small: before
  adding, check the existing keys in the system prompt and update the one that
  already covers the topic. NEVER store database-queryable metrics (VO2max, FTP,
  LTHR, PRs, zones, resting HR/HRV) — query those fresh. Existing facts are
  already in the system prompt, so there is no need to fetch them. Recording a
  fact is a SILENT side effect, never an answer: save it as an early step,
  before you write your reply, and NEVER end a turn with only a one-line
  acknowledgement such as "Noted in memory…". Your final message must always
  contain the complete, data-backed answer to the question the user actually
  asked.
{training_tools}

---

{output_conventions}
"""

#: Output conventions, injected as ``{output_conventions}`` into the main
#: agent's prompt so numbers and units are always phrased the same way.
_OUTPUT_CONVENTIONS = """\
# Output conventions

1. Always include units (`7.6 hours`, `154 bpm`, `245 W`).
2. Give running as pace (`MM:SS /km`, from `60 / speed_kmh`), not km/h; lead with it.
3. Durations as `HH:MM:SS` or `Xh Ym`.
4. Cadence: the stored value is running steps/min PER LEG — total both-feet
   cadence is 2x (quote the 2x); cycling/rowing is rpm.
5. ACWR vs running form: general `acwr` uses training load across ALL activities
   and can't see tissue stress — prefer `run_acwr` (foot-strike volume) for
   running load, and `run_cadence_drift` (negative = overstriding) when the user
   mentions leg/joint/tendon aches or form breaking down. Read both fresh from
   `derived_metrics`.
6. Present numbers, units and ranges as plain text, not LaTeX. Write
   `20 km/wk → 75–78 km/wk`, `154 bpm`, `MM:SS /km` directly. Do NOT emit
   math markup such as `$...$`, LaTeX text commands or fraction notation —
   those render as raw text in the chat. Only use `$...$` for an actual
   formula, and keep it short.
"""

#: The "how freely to query" bullet, injected via ``{query_guidance}``. The main
#: agent does all the querying itself.
_QUERY_GUIDANCE = """\
- **Query freely.** It's cheap and safe to run multiple queries — probe,
  iterate, and dig into the data before answering. Don't over-optimize for a few
  tool calls; run whichever queries help you get an accurate, well-grounded
  answer. `get_day_summary` is just a convenient one-call shortcut for a single
  day, not a restriction — you're welcome to write your own SQL.
"""

#: The single training bullet of the system prompt, injected via
#: ``{training_tools}`` (omitted when the entry point does not pass a training
#: season). One tool covers the whole season — goals, periodized blocks, weekly
#: targets and the dated workouts.
_TRAINING_TOOLS_GUIDE = """\
- **`training`** — the ONE tool for the whole training season (goals → blocks →
  weeks → workouts); read and write it only here, never via a plan/anchor split.
  * `action="get"` (default): a **windowed agenda** for `day` (default today).
    It returns each goal's header + the blocks overlapping the window, and a
    `weeks` array with one entry per calendar week (Mon..Sun): that week's
    per-goal `targets` (block, week target, planned-vs-target and
    actual-vs-target `coverage`), its `planned` workouts and its `actual` synced
    activities. Scope the window with `weeks=N` (N calendar weeks from Monday of
    `day`'s week; default 1) or an explicit `date_start`/`date_end` range. Pass
    `goal_id` to focus one goal; `detail=true` for full descriptions. Use
    `full=true` only when you need the whole anchor (every goal/block/week + the
    resolved current block), e.g. before a season-wide reshape. `can_undo`
    reports whether a destructive edit can still be rolled back. Call this
    BEFORE building or reshaping a week, and to answer "how is my week going?"
    or "what are the next N weeks?".
  * `action="apply"`: one **atomic** season edit (anchor and workouts run in one
    transaction, so a bad part writes nothing). `spec` is JSON with an optional
    `"workouts"` section (creates/patches workouts + `delete_ids`) and/or an
    `"anchor"` section (goals/blocks/weeks), or `{"undo": true}` to roll back
    the last destructive edit. A destructive edit snapshots the whole season
    first, and the result reports what changed (`added_ids`/`updated_ids` in
    `"workouts"`, the anchor's new `blocks`/`weeks` ids) so you never need a
    follow-up read.
  * WORKOUT FIELDS (each entry of a `"workouts"` list): planned_date,
    activity_type (run/cycle/swim/strength/rest/other), title, description (ONE
    short line, at most ~200 chars, summarising THIS session), duration_min, distance_km, intensity
    (easy/moderate/hard/race_pace), target_pace_min_km (decimal min/km, 5.5 =
    5:30), target_hr_zone, target_power_w, status
    (planned/completed/partial/skipped). Workouts automatically attach to the
    active block and goal by planned_date (goal_id is an optional manual
    override). Add `steps` whenever a session has more than one segment — any
    warm-up/work/recovery/cool-down structure, intervals, strides, a progressive
    long run, etc. — as an ordered list of {kind
    (warmup/steady/work/recovery/cooldown/rest), duration ('15m'/'90s'/'1:30'),
    repeat?, label? (a 1-2 word segment name, NEVER digits/time/rep), intensity?,
    target_pace_min_km?, target_hr_zone?, target_power_w?}; omit `steps` only for
    a single continuous effort. An entry with an `id` is PATCHED (send only what
    changes); one without creates. `delete_ids` deletes workouts. To move or copy
    workouts, read them first and PATCH/create the ids you mean — never guess ids.
    `title`, `description` and a step `label` name the session or its segments;
    they are never a place for rules, preferences or reminders — any guidance
    that outlives this one workout belongs in `memory`.
  * ANCHOR FIELDS (inside an `"anchor"` section): `goal_id` updates that goal
    (PATCH; omitted creates a new one) — a blocks-only spec targets the sole
    goal rather than inventing a blank one. title/sport/start_date/target_date/
    target_time/target_distance_km are the target; sport is
    run/cycle/swim/strength/rest/other. `blocks` upserts blocks (an entry WITH
    `id` patches it, one WITHOUT creates) — each: name, focus (short phase
    labels, never rules or reminders), start_date/
    end_date (optional), target_weekly_km (optional baseline weekly volume), and optional `weeks`: a
    per-week PATCH keyed by `week_start` — each entry {week_start, distance_km,
    duration_min, is_deload} writes ONLY that week, every week you do not send is
    left untouched (so changing one week only requires sending that week); a bare
    {week_start} entry removes that week's target, and `weeks: []` clears
    all of them. Give each block a `target_weekly_km` when its weeks vary, so the
    weeks you leave out still carry a baseline target.
    `delete_blocks`/`delete_goal_ids` delete — destructive but restorable with
    `undo`. Mark recovery weeks `is_deload: true`.
  * Derive target paces/zones from recent metrics (volume, HR zones, race
    predictions, VO2max) and store them ON the workouts. When the user sets a new
    target, reshape the anchor and re-derive the near-term workouts in the SAME
    `apply` by sending both `anchor` and `workouts` sections. When showing a plan
    in chat, embed `<plan_table />` (or `<plan_table from="..." to="..." />`);
    never format JSON or a markdown table.
"""

#: The date note, emitted by the module-level *dynamic* system prompt. Kept as its
#: own (date-only) part so it is the sole thing re-evaluated each turn — the static
#: template above is built once and never needs touching again.
_TODAY_PROMPT = """\
Today's date is {today}. This line is refreshed on every turn, but for any question
that depends on NOW — "today", "tomorrow", "this week", "last Monday" — call the
`today` tool and trust its result: it is the authoritative current date and its
weekday is already computed for you.
"""

#: Appended to the date note when the store is behind today. Kept as its own
#: template (not merged into ``_TODAY_PROMPT``) so the freshness report can be
#: dropped entirely when the data is current or there is none yet — the model
#: is never told a precise gap that does not exist.
_STALENESS_PROMPT = """\
# Data freshness

The most recent date with stored data is {data_until}, which is {gap} day(s) behind
today. For any question that is about NOW — "today", "last night", "this week",
"current" or "most recent" form — never present a value from {data_until} or earlier
as if it were today's. If the user asks about a window that extends beyond
{data_until}, say explicitly that the data is only up to {data_until} ({gap} day(s)
stale) and reason about the trend instead of inventing a value for the missing days.
"""

#: Stable identity of the date+freshness dynamic system prompt. Pydantic AI keys
#: a dynamic prompt by ``func.__qualname__`` and re-evaluates a resumed
#: ``SystemPromptPart`` only when its ``dynamic_ref`` equals that key, so this is
#: pinned to a constant (rather than a ``build_agent.<locals>`` name that would
#: drift) to keep ``_refresh_resumed_prompt`` able to re-stamp persisted history.
_DATE_PROMPT_REF = "garmin_fetch.ask.dynamic_date_prompt"

#: The long-term-memory note, also a *dynamic* system prompt (registered as a
#: closure in ``build_agent`` so it can read the per-agent ``memory`` and be
#: re-evaluated every turn, like the date). Because it is dynamic, a fact the
#: model stores mid-session is visible on the next turn and across resumed
#: sessions without the static template ever being touched.
_MEMORY_PROMPT_TITLE = "## Long-term memory about this athlete (persistent facts)"


def _memory_prompt(memory: _Memory) -> str | None:
    """Render the user's durable facts as a system-prompt section, or ``None``
    when there is nothing to say (so no empty block is injected)."""
    facts = memory.get()
    if not facts:
        return None
    lines = "\n".join(f"- **{key}**: {value}" for key, value in facts.items())
    prompt = (
        _MEMORY_PROMPT_TITLE
        + "\n\nThe facts below were recorded in earlier conversations and stay "
        "available every turn. Rely on them for personalisation, but NEVER "
        "duplicate database-queryable metrics (VO2max, FTP, PRs, zones, HR) "
        "here — always query those fresh. Every entry must be durable (still "
        "true and useful months from now) and high-ROI (it changes how you "
        "coach); forget anything transient, situational or single-chat. Keep "
        "this profile small: before adding a fact, update the existing key that "
        "already covers the topic; forget facts that are wrong, stale or "
        "superseded rather than leaving duplicates.\n\n"
        + lines
    )
    cap = memory.max_facts
    total = len(facts)
    if cap and total >= cap:
        prompt += (
            f"\n\nMemory is FULL ({total}/{cap} facts). Do not add more: use the "
            "`memory` tool to consolidate overlapping keys and forget stale "
            'facts; `action="replace"` rewrites the whole profile.'
        )
    elif cap and total >= cap - 5:
        prompt += (
            f"\n\nMemory is nearly full ({total}/{cap} facts). Prefer updating "
            "or merging existing keys over adding new ones."
        )
    return prompt


def _memory_cap_note(total: int, cap: int) -> str:
    """Feedback appended to a memory write so the model can self-throttle.

    Reports how full the profile is and, near/at the cap, pushes the model to
    consolidate instead of accumulating more keys.
    """
    if cap and total >= cap:
        return (
            f" ({total}/{cap} facts — full; consolidate overlapping keys or "
            "forget stale ones before adding)"
        )
    if cap and total >= cap - 5:
        return (
            f" ({total}/{cap} facts — near the limit; prefer updating or "
            "merging over adding new keys)"
        )
    return f" ({total} facts)"


#: The athlete-authored training principles, injected as a *dynamic* system
#: prompt like memory. Unlike memory these are directives the athlete controls
#: (from Settings), so they are framed as authoritative guidance rather than as
#: facts to personalise with.
_PRINCIPLES_PROMPT_TITLE = (
    "## Training principles (authoritative guidance from the athlete)"
)


def _principles_prompt(principles: _Principles) -> str | None:
    """Render the athlete's training principles as a system-prompt section, or
    ``None`` when there are none (so no empty block is injected)."""
    text = principles.get().strip()
    if not text:
        return None
    return (
        _PRINCIPLES_PROMPT_TITLE
        + "\n\nThe athlete set the following principles. Treat them as "
        "authoritative guidance: follow them when planning, adjusting or "
        "evaluating training, even when they differ from your defaults. They "
        "are instructions, not data — do not edit them or repeat them back "
        "unless asked.\n\n"
        + text
    )


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

    def __init__(
        self,
        url: str,
        *,
        user_id: int | None = None,
        excluded_types: Iterable[str] | str = (),
    ) -> None:
        self.path = url
        self.user_id = user_id
        self._backend = _PgReadOnlyBackend(url)
        self._excluded = self._normalize_excluded(excluded_types)
        self._enabled_cols: set[str] | None = None
        self._claimed_cols: set[str] | None = None

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        user_id: int | None = None,
        excluded_types: Iterable[str] | str = (),
    ) -> "ReadOnlyDB":
        """A Postgres-backed read-only handle from a ``postgres://`` DSN.

        ``excluded_types`` is the account's disabled daily data types (a comma
        list string or an iterable). Columns that no *enabled* type can write
        (e.g. a SpO2 column when spo2 is excluded) are hidden from the agent's
        schema so it never pays tokens to introspect metrics it has no data for.
        """
        return cls(url, user_id=user_id, excluded_types=excluded_types)

    @staticmethod
    def _normalize_excluded(excluded_types: Iterable[str] | str) -> set[str]:
        if isinstance(excluded_types, str):
            return {s.strip() for s in excluded_types.split(",") if s.strip()}
        return {s.strip() for s in excluded_types if s.strip()}

    def _daily_enabled_cols(self) -> tuple[set[str], set[str]]:
        """(enabled, claimed) column sets for ``daily_metrics``.

        ``claimed`` is every column some daily type can write; ``enabled`` is the
        subset still written by a non-excluded type (mirrors
        ``db.prune_excluded_types``). A column a disabled type exclusively writes
        is dropped so the agent only sees columns it can actually have data for.
        """
        if self._enabled_cols is None:
            enabled: set[str] = set()
            claimed: set[str] = set()
            for name, cols in TYPE_COLUMNS.items():
                claimed |= cols
                if name not in self._excluded:
                    enabled |= cols
            self._enabled_cols = enabled
            self._claimed_cols = claimed
        return self._enabled_cols, self._claimed_cols

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
            cols = self._backend.columns(conn, table)
        if table != "daily_metrics":
            return cols
        # Only the day-metric columns are gated by the per-user data-type
        # toggles: drop columns no *enabled* type can write (keeps system
        # columns like user_id/calendar_date/fetched_at, which no type claims).
        enabled, claimed = self._daily_enabled_cols()
        return [c for c in cols if c["name"] not in claimed or c["name"] in enabled]

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

    def latest_date(self) -> date | None:
        """The most recent calendar_date with any stored data (or ``None``).

        Looks across ``daily_metrics``, ``derived_metrics`` and
        ``activity_summaries`` so the agent can tell how stale the store is
        relative to today (a user who has not synced lately would otherwise be
        served "today" answers from days-old rows). Every date column is TEXT
        in ``YYYY-MM-DD`` form (db.py), so ``MAX`` is lexicographic and the
        values are parsed back to real ``date`` objects here.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT"
                    " (SELECT MAX(calendar_date) FROM daily_metrics) AS d,"
                    " (SELECT MAX(calendar_date) FROM derived_metrics) AS m,"
                    " (SELECT MAX(start_date) FROM activity_summaries) AS a"
                ).fetchone()
        except _DB_ERROR_TYPES:
            return None
        candidates = [row[k] for k in ("d", "m", "a") if row and row[k]]
        if not candidates:
            return None
        parsed: list[date] = []
        for value in candidates:
            if isinstance(value, date):
                parsed.append(value)
                continue
            try:
                parsed.append(date.fromisoformat(str(value).strip()[:10]))
            except ValueError:
                continue
        return max(parsed) if parsed else None

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
            running = conn.execute(
                "SELECT metric, value, qualifier FROM derived_metrics "
                "WHERE calendar_date = %s AND metric LIKE 'run_%' ORDER BY metric",
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
            "running": _jsonify_rows([dict(r) for r in running]),
        }

    def metric_trend(self, metric: str, days: int = 30) -> dict[str, Any]:
        """Query time series and summary stats for a core metric over N trailing days."""
        days = max(1, min(int(days), 365))
        m = metric.strip().lower()
        derived_names = {"acwr", "run_acwr", "run_cadence_drift"}
        daily_cols = {
            "sleep_score", "sleep_time_hours", "resting_hr", "hrv_last_night_avg", "vo2max",
            "total_steps", "total_distance_m", "stress_avg", "body_battery_max",
            "body_battery_min", "weight_kg", "sweat_loss_ml"
        }
        if m not in derived_names and m not in daily_cols:
            raise ValueError(
                f"unsupported metric '{metric}'. Supported: "
                + ", ".join(sorted(derived_names | daily_cols))
                + " (or write custom SQL with run_sql)"
            )
        start = (date.today() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            if m in derived_names:
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
        sql = (
            "SELECT activity_id, activity_name, activity_type, start_date, start_time_local, "
            "duration_hours, distance_km, avg_hr, max_hr, pace_min_km, avg_speed_kmh, "
            "avg_cadence, avg_power_w, training_load, elevation_gain_m, weather_temp_c, is_pr "
            "FROM activity_summaries WHERE start_date >= %s "
        )
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

        out: list[dict[str, Any]] = []
        for r in rows:
            dur = r["duration_hours"]
            dur_fmt = None
            if dur is not None and dur > 0:
                h = int(dur)
                m = int(round((dur - h) * 60))
                dur_fmt = f"{h}h {m}m" if h else f"{m}m"

            pace_val = r["pace_min_km"]
            if pace_val is None and r["avg_speed_kmh"] and r["avg_speed_kmh"] > 0:
                pace_val = 60.0 / r["avg_speed_kmh"]
            pace_fmt = None
            if pace_val is not None and pace_val > 0:
                p_m = int(pace_val)
                p_s = int(round((pace_val - p_m) * 60))
                if p_s >= 60:
                    p_m += 1
                    p_s = 0
                pace_fmt = f"{p_m}:{p_s:02d} /km"

            cad = r["avg_cadence"]
            cad_total = None
            cad_unit = "rpm"
            act_type = (r["activity_type"] or "").lower()
            if cad is not None:
                if "run" in act_type:
                    cad_total = round(cad * 2)
                    cad_unit = "total spm (both feet)"
                else:
                    cad_total = round(cad)
                    cad_unit = "rpm"

            out.append({
                "activity_id": r["activity_id"],
                "activity_name": r["activity_name"],
                "activity_type": r["activity_type"],
                "start_date": r["start_date"],
                "duration": dur_fmt,
                "distance_km": round(r["distance_km"], 2) if r["distance_km"] else None,
                "pace": pace_fmt,
                "cadence": f"{cad_total} {cad_unit}" if cad_total else None,
                "avg_hr": round(r["avg_hr"]) if r["avg_hr"] else None,
                "max_hr": round(r["max_hr"]) if r["max_hr"] else None,
                "avg_power_w": round(r["avg_power_w"]) if r["avg_power_w"] else None,
                "training_load": round(r["training_load"]) if r["training_load"] else None,
                "elevation_gain_m": round(r["elevation_gain_m"]) if r["elevation_gain_m"] else None,
                "weather_temp_c": r["weather_temp_c"],
                "is_pr": bool(r["is_pr"]),
            })
        return out

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
        self,
        *,
        default_lat: float | None = None,
        default_lon: float | None = None,
        http_get: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        today: date | None = None,
        db: ReadOnlyDB | None = None,
    ) -> None:
        self._default_lat = default_lat
        self._default_lon = default_lon
        self._http_get = http_get
        self._today = today or date.today()
        self._db = db

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._http_get is not None:
            return self._http_get(url, params)
        import httpx

        resp = httpx.get(url, params=params, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def from_config(
        cls, cfg: dict[str, str], *, db: ReadOnlyDB | None = None, **kwargs: Any
    ) -> "Weather":
        """Build from config strings (empty strings become no default)."""
        return cls(
            default_lat=_float_or_none(cfg.get("weather_home_lat")),
            default_lon=_float_or_none(cfg.get("weather_home_lon")),
            db=db,
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
            # Try stored forecast from database first when querying home location
            if self._db is not None and lat == self._default_lat and lon == self._default_lon:
                try:
                    sql = (
                        "SELECT calendar_date AS date, temp_max_c, temp_min_c, "
                        "precip_mm, wind_max_kmh, condition_code "
                        "FROM weather_forecast "
                        f"WHERE calendar_date >= '{start.isoformat()}' "
                        f"AND calendar_date <= '{end.isoformat()}' "
                        "ORDER BY calendar_date"
                    )
                    res = self._db.run_sql(sql)
                    cols = res.get("columns") or []
                    rows = res.get("rows") or []
                    if rows:
                        days_list = [dict(zip(cols, r)) for r in rows]
                        return {
                            "source": "forecast",
                            "location": {"lat": lat, "lon": lon},
                            "days": days_list,
                            "note": (
                                "Stored Open-Meteo daily forecast from local database. "
                                f"Covered {start.isoformat()}..{end.isoformat()}."
                            ),
                        }
                except Exception:
                    pass
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


#: Bookkeeping columns the agent never needs (they only cost tokens).
_ANCHOR_DROP = ("created_at", "updated_at")


def _slim(d: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop bookkeeping keys from one anchor row (None passes through)."""
    if d is None:
        return None
    return {k: v for k, v in d.items() if k not in _ANCHOR_DROP}


def _anchor_goal_dump(
    anchor: Any, goal_id: int, day: str
) -> dict[str, Any] | None:
    """Slim one resolved goal for the agent (None when it does not exist).

    ``anchor.resolve_goal`` is the single composition of the goal/block/week
    resolution; this only drops bookkeeping columns and flattens the weekly
    target so the model never re-derives the block/week arithmetic.
    """
    resolved = anchor.resolve_goal(goal_id, day)
    if resolved is None:
        return None
    out = _slim(resolved["goal"]) or {}
    out["blocks"] = [
        {**(_slim(block) or {}),
         "weeks": [_slim(w) or {} for w in block.get("weeks", [])]}
        for block in resolved["blocks"]
    ]
    out["current_block"] = _slim(resolved["current_block"])
    out["week_number"] = resolved["week_number"]
    target = resolved["weekly_target"]
    out["weekly_target"] = (
        None
        if target is None
        else {
            "block_id": target["block"]["id"],
            "block": target["block"]["name"],
            "week_number": target.get("week_number"),
            "week_count": target.get("week_count"),
            "week": _slim(target["week"]),
            "coverage": target.get("coverage"),
        }
    )
    return out


def _monday_iso(value: str) -> str:
    """Monday of the calendar week containing a YYYY-MM-DD date."""
    d = date.fromisoformat(value)
    return (d - timedelta(days=d.weekday())).isoformat()


def _block_overlaps(
    block: dict[str, Any], win_start: str, win_end: str
) -> bool:
    """True when a block's dated range intersects the window.

    An undated block (or one with a single open bound) always counts, so a
    windowed view never hides a phase that has not been dated yet.
    """
    start, end = block.get("start_date"), block.get("end_date")
    if start and end:
        return start <= win_end and end >= win_start
    if start:
        return start <= win_end
    if end:
        return end >= win_start
    return True


def _goal_header(
    season: Any, goal: dict[str, Any], win_start: str, win_end: str
) -> dict[str, Any]:
    """One goal's header plus only the blocks that overlap the window."""
    out = _slim(goal) or {}
    out["blocks"] = [
        {
            "id": b["id"],
            "name": b["name"],
            "focus": b["focus"],
            "start_date": b["start_date"],
            "end_date": b["end_date"],
            "target_weekly_km": b.get("target_weekly_km"),
        }
        for b in season.blocks(goal["id"])
        if _block_overlaps(b, win_start, win_end)
    ]
    return out


def _build_weeks(
    win_start: str,
    win_end: str,
    planned: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One entry per calendar week (Mon..Sun) in the window.

    Each entry groups that week's planned workouts, actual activities and
    per-goal phase targets (with scheduled-vs-target coverage), so the model
    never has to cross-reference goals -> blocks -> weeks to know what a given
    week's plan is.
    """
    planned_by_week: dict[str, list[dict[str, Any]]] = {}
    for w in planned:
        planned_by_week.setdefault(_monday_iso(w["planned_date"]), []).append(w)
    actual_by_week: dict[str, list[dict[str, Any]]] = {}
    for a in actual:
        actual_by_week.setdefault(_monday_iso(a["start_date"]), []).append(a)
    targets_by_week: dict[str, list[dict[str, Any]]] = {}
    for t in targets:
        targets_by_week.setdefault(t["week_start"], []).append(t)
    out: list[dict[str, Any]] = []
    cur = date.fromisoformat(win_start)
    end = date.fromisoformat(win_end)
    while cur <= end:
        ws = cur.isoformat()
        out.append({
            "week_start": ws,
            "week_end": (cur + timedelta(days=6)).isoformat(),
            "targets": targets_by_week.get(ws, []),
            "planned": planned_by_week.get(ws, []),
            "actual": actual_by_week.get(ws, []),
        })
        cur += timedelta(days=7)
    return out


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
    """Render the agent-facing *table overview* (no columns).

    Each table gets its semantic description from ``_TABLE_NOTES`` only — the
    prompt deliberately omits the column list. Columns (and their short
    descriptions) are fetched on demand through ``table_schema``, so the prompt
    stays lean and never goes stale as columns are added dynamically.
    """
    lines: list[str] = []
    for table in db.tables():
        note = _TABLE_NOTES.get(table)
        lines.append(f"{table}: " + (note if note else "see table_schema"))
    return "\n".join(lines)


def _render_system_prompt(overview: str, *, has_training: bool) -> str:
    """Fill the static system prompt for the tools this agent actually has.

    ``str.replace`` (not ``.format``) because the template's chart guidance
    contains literal JSON braces (``{"sql": ...}``). The single training bullet
    is emitted only when the training season is wired, so the prompt never
    advertises a tool the model cannot call.
    """
    prompt = (
        _PROMPT_TEMPLATE
        .replace("{overview_text}", overview or "(no tables available in the database)")
        .replace(
            "{training_tools}",
            _TRAINING_TOOLS_GUIDE if has_training else "",
        )
        .replace("{query_guidance}", _QUERY_GUIDANCE)
        .replace("{output_conventions}", _OUTPUT_CONVENTIONS)
    )
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()


def _current_date_prompt(db: ReadOnlyDB | None = None) -> str:
    """The date + data-freshness note, re-evaluated as a dynamic system prompt.

    Includes the weekday name (e.g. ``2026-08-16 (Sunday)``) so the model never
    has to compute the day of week from the date itself — models reliably get
    that arithmetic wrong by a day. When a ``db`` is given it also reports the
    most recent date with stored data and how far behind today it is, so the
    model never presents stale rows as current. Everything else is the (static)
    ``_PROMPT_TEMPLATE`` built once per agent.
    """
    text = _TODAY_PROMPT.format(today=date.today().strftime("%Y-%m-%d (%A)"))
    if db is None:
        return text
    try:
        data_until = db.latest_date()
    except Exception:
        return text
    if data_until is None:
        return text + "\n\n# Data freshness\n\nNo data has been synced yet."
    gap = (date.today() - data_until).days
    if gap <= 0:
        return text
    return text + "\n\n" + _STALENESS_PROMPT.format(
        data_until=data_until.isoformat(), gap=gap
    )


def _refresh_resumed_prompt(
    messages: list[Any], db: ReadOnlyDB | None = None, *,
    has_training: bool = True,
) -> list[Any]:
    """Re-stamp a resumed session's system prompts for the current build.

    Two parts need refreshing when a stored session is resumed:

    * the *date* part, so the model never thinks "today" is the day the session
      started (legacy sessions carry it with no usable ``dynamic_ref``); and
    * the *static role/tool* part, when the tool surface changed since the
      session was persisted — e.g. a tool was renamed, so the stored prompt
      tells the model to call a tool that no longer exists and Pydantic AI
      answers ``Unknown tool name``. Re-rendering it from the current template
      keeps a resumed chat working across code changes instead of requiring the
      user to clear the conversation.

    The dynamic memory / training-principles parts are re-evaluated by Pydantic
    AI and are left alone.
    """
    from pydantic_ai.messages import SystemPromptPart

    fresh = _current_date_prompt(db)
    ref = _DATE_PROMPT_REF
    old_prefix = "Today's date is"
    static_prefix = "# Role & Environment"
    fresh_static = (
        _render_system_prompt(_schema_text(db), has_training=has_training)
        if db is not None else None
    )
    for message in messages:
        for part in getattr(message, "parts", []):
            if not isinstance(part, SystemPromptPart):
                continue
            if part.content.startswith(static_prefix):
                if fresh_static is not None and part.content != fresh_static:
                    part.content = fresh_static
                    part.dynamic_ref = None
                continue
            if part.dynamic_ref != ref and not part.content.startswith(old_prefix):
                continue
            if part.content == fresh and part.dynamic_ref == ref:
                continue
            part.content = fresh
            part.dynamic_ref = ref
    return messages


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


def _register_query_tools(
    agent: Any,
    db: ReadOnlyDB,
    *,
    weather: "Weather | None" = None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Register the read-only schema/query/stats tools on ``agent``.

    The read-only surface: schema introspection, date helpers, free SQL, and the
    day/trend/activity shortcuts (plus ``weather`` when configured). Nothing
    here writes.
    """

    @agent.tool_plain
    def table_schema(table: str) -> str:
        """Return a table's live columns (name, type, description each).

        Call this before writing a query against a table — the system prompt
        only lists table overviews. Columns are queried live, so newly-added
        columns appear automatically.
        """
        if on_status is not None:
            on_status(f"Inspecting schema for {table}…")
        try:
            return json.dumps(_column_descriptions(db, table), ensure_ascii=False)
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
        small limits — results are capped at 500 rows.
        """
        if on_status is not None:
            on_status("Running database query…")
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
        you need without selecting the wide daily_metrics row.
        """
        if on_status is not None:
            on_status(f"Fetching summary for {calendar_date}…")
        try:
            return json.dumps(db.day_summary(calendar_date), ensure_ascii=False)
        except (ValueError, QueryError, psycopg.Error) as exc:
            return _tool_error(exc)

    @agent.tool_plain
    def get_metric_trend(metric: str, days: int = 30) -> str:
        """Return daily values and summary statistics (min/max/avg/trend) for a core metric.

        ``metric`` is one of:
        - Load & running form: 'acwr' (acute/chronic load ratio),
          'run_acwr' (running distance ACWR), 'run_cadence_drift' (gait z-score)
        - Daily health: 'sleep_score' (0-100), 'resting_hr' (bpm),
          'hrv_last_night_avg' (ms), 'vo2max', 'total_steps', 'total_distance_m',
          'stress_avg', 'weight_kg'
        ``days`` is the trailing window (default 30, up to 365).
        Use this instead of writing SQL when asked for trends or averages of these core metrics.
        """
        if on_status is not None:
            on_status(f"Analyzing {metric} trend…")
        try:
            return json.dumps(db.metric_trend(metric, days), ensure_ascii=False)
        except (ValueError, psycopg.Error) as exc:
            return _tool_error(exc)

    @agent.tool_plain
    def get_recent_activities(
        sport: str | None = None,
        days: int = 30,
        limit: int = 10,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> str:
        """Return recent activities with formatted pace (MM:SS /km) and 2x total cadence.

        ``sport`` is an optional filter (e.g. 'run', 'cycling', 'swimming').
        ``days`` is the trailing window (default 30) used when ``date_start`` and
        ``date_end`` are omitted. Pass ``date_start`` / ``date_end`` (ISO YYYY-MM-DD)
        to scope to an explicit inclusive date range instead — useful for comparing
        epochs without writing multi-column SQL; if only one is given the other is
        open-ended.
        ``limit`` is max activities to return (default 10, max 50).
        Returns human-friendly summaries with converted running paces, duration,
        and total both-feet cadence. Use this instead of multi-column SQL for
        'what were my last runs' or 'show recent workouts'.
        """
        if on_status is not None:
            on_status("Retrieving recent activities…")
        try:
            return json.dumps(
                db.recent_activities(sport, days, limit, date_start, date_end),
                ensure_ascii=False,
            )
        except (ValueError, psycopg.Error) as exc:
            return _tool_error(exc)

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
            if on_status is not None:
                on_status("Checking weather forecast…")
            try:
                return json.dumps(
                    weather.query(date_start, date_end, lat, lon),
                    ensure_ascii=False,
                )
            except ValueError as exc:
                return f"ERROR: {exc}"


def _register_chart_tool(
    agent: Any,
    db: ReadOnlyDB,
    *,
    chart_cache: dict[str, Any] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Register the chart tool on ``agent``.

    Validates a spec and caches the built figure under its SQL, so the UI can
    rerun the query to draw the chart when the agent embeds the returned
    ``<chart>`` spec.
    """

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
        if on_status is not None:
            on_status("Validating chart…")
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
            fig = _build_chart_figure(parsed, result)
            if chart_cache is not None:
                try:
                    fig_dict = _json.loads(fig.to_json())
                    chart_cache[sql.strip()] = fig_dict
                    chart_cache[_json.dumps(parsed, sort_keys=True)] = fig_dict
                except Exception:
                    pass
        except ValueError as exc:
            return _tool_error(exc)
        rows = result.get("rows") or []
        return (
            "OK: " + _json.dumps(parsed, ensure_ascii=False)
            + f" (query returned {len(rows)} rows)"
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
    weather: "Weather | None" = None,
    training: _Training | None = None,
    chart_cache: dict[str, Any] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Any:
    """Build the Pydantic AI agent wired to ``db`` tools.

    Pass ``model`` (e.g. ``TestModel``) to override the transport for tests.
    Pass ``reasoning_effort`` to request a reasoning effort level from the
    underlying model (``low``/``medium``/``high``). ``provider`` selects the
    transport: ``openai`` (default, any OpenAI-compatible ``base_url``) or
    ``gemini`` (native google-genai). Pass ``memory`` to give the agent the
    ``memory`` tool over a durable user profile (long-term memory between
    sessions). Pass ``principles`` (a ``server.state.PgPrinciples``) to
    inject the athlete-authored training principles as authoritative guidance
    (read-only: they are edited in Settings, never by the agent). Pass
    ``training`` (a ``server.state.TrainingSeason``) to give the agent the single
    ``training`` tool over the whole season (goals/blocks/weeks/workouts).
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
    overview = _schema_text(db)
    # Columns are NOT in the prompt — the model introspects them via the
    # ``table_schema`` tool on demand; the date is injected separately as a
    # dynamic prompt below.
    system_prompt = _render_system_prompt(
        overview,
        has_training=training is not None,
    )
    agent = Agent(model, system_prompt=system_prompt)

    # The date must be a dynamic system prompt: static prompts are evaluated
    # once and stored in the message history, so a cached agent (or a resumed
    # session) would keep "Today is <yesterday>" forever. A dynamic prompt is
    # re-evaluated on every model request, so the date can never go stale. The
    # closure pins ``__qualname__`` to ``_DATE_PROMPT_REF`` so resumed history
    # (matched by ``_refresh_resumed_prompt``) keeps refreshing on later turns.
    def _date_dynamic_prompt() -> str:
        return _current_date_prompt(db)

    _date_dynamic_prompt.__qualname__ = _DATE_PROMPT_REF
    agent.system_prompt(dynamic=True)(_date_dynamic_prompt)

    # The long-term-memory profile is injected the same way: a *dynamic* system
    # prompt, so it is re-evaluated every turn (a fact the model stores via
    # ``memory`` tool mid-session shows up on the next turn) and stays fresh
    # across resumed sessions. It is a closure, so its ``dynamic_ref`` is stable
    # (``build_agent.<locals>._memory_dynamic_prompt``) and resumed history can
    # match it. When there are no facts it renders nothing, so no empty block is
    # sent.
    if memory is not None:

        def _memory_dynamic_prompt() -> str | None:
            return _memory_prompt(memory)

        agent.system_prompt(dynamic=True)(_memory_dynamic_prompt)

    # The athlete's training principles are injected the same way: a *dynamic*
    # system prompt so an edit made in Settings is picked up on the next turn
    # and across resumed sessions, and nothing is sent when there are none.
    if principles is not None:

        def _principles_dynamic_prompt() -> str | None:
            return _principles_prompt(principles)

        agent.system_prompt(dynamic=True)(_principles_dynamic_prompt)

    _register_query_tools(agent, db, weather=weather, on_status=on_status)

    _register_chart_tool(agent, db, chart_cache=chart_cache, on_status=on_status)

    if memory is not None:

        @agent.tool_plain(name="memory")
        def memory_tool(
            action: str = "set",
            facts: dict[str, Any] | None = None,
            keys: list[str] | None = None,
        ) -> str:
            """Maintain durable facts about the user (long-term memory).

            Existing facts are already in the system prompt, so never re-add
            one. Use short snake_case keys.

            Only store facts that are durable (still true and useful months
            from now, across future conversations) AND high-ROI (they change
            how you coach). Do NOT store transient or situational details — a
            one-off question, a bad night, this week's soreness or schedule, a
            passing mood. When in doubt, do not store it.

            This is a SILENT side effect, not an answer: call it as an early
            step and always finish with the full answer to the user's question —
            never end a turn with only a confirmation that a fact was saved.

            - ``action="set"`` (default): upsert one or more ``facts``
              ({key: value}). An existing key is overwritten on change — never
              create a near-duplicate for a fact you already have.
            - ``action="forget"``: delete the given ``keys`` — use it for facts
              that are wrong, stale or superseded.
            - ``action="replace"``: overwrite the WHOLE profile with ``facts``
              — the way to consolidate overlapping keys and drop facts that are
              no longer true or are database-queryable.
            """
            action = (action or "set").strip().lower()
            if action == "set":
                if not facts:
                    return "ERROR: facts is required for action=set"
                if on_status is not None:
                    on_status("Updating memory…")
                try:
                    total = memory.remember(facts)
                except ValueError as exc:
                    return f"ERROR: {exc}"
                return f"saved{_memory_cap_note(total, memory.max_facts)}"
            if action == "forget":
                if not keys:
                    return "ERROR: keys is required for action=forget"
                removed = memory.forget(keys)
                if not removed:
                    return "ERROR: no such key(s): " + ", ".join(
                        str(k) for k in keys
                    )
                total = len(memory.get())
                return (
                    "forgotten: "
                    + ", ".join(removed)
                    + _memory_cap_note(total, memory.max_facts)
                )
            if action == "replace":
                if not facts:
                    return "ERROR: facts is required for action=replace"
                if on_status is not None:
                    on_status("Updating memory…")
                try:
                    total = memory.replace(facts)
                except ValueError as exc:
                    return f"ERROR: {exc}"
                return f"replaced profile{_memory_cap_note(total, memory.max_facts)}"
            return "ERROR: action must be set, forget or replace"

    if training is not None:
        season = training

        @agent.tool_plain
        def training(
            action: str = "get",
            spec: str | None = None,
            day: str | None = None,
            goal_id: int | None = None,
            date_start: str | None = None,
            date_end: str | None = None,
            weeks: int = 1,
            full: bool = False,
            detail: bool = False,
        ) -> str:
            """The ONE tool for the whole training season (goals -> blocks ->
            weeks -> workouts): read and write it here, never via a split.

            ``action="get"`` (default) returns a windowed agenda for ``day``
            (default today): each goal's header plus the blocks overlapping the
            window, and a ``weeks`` array with one entry per calendar week
            (Mon..Sun) — that week's per-goal ``targets`` (block, week target,
            planned-vs-target and actual-vs-target ``coverage``), its
            ``planned`` workouts and its ``actual`` synced activities. The
            window is ``weeks`` calendar weeks from Monday of ``day``'s week
            (default 1), or an explicit ``date_start``/``date_end`` range. Pass
            ``goal_id`` to focus one goal. ``detail=true`` keeps full workout
            descriptions. Set ``full=true`` only when you need the entire anchor
            (every goal/block/week of the season plus the resolved current
            block) — e.g. before a season-wide reshape. ``can_undo`` reports
            whether a destructive edit can still be rolled back.

            ``action="apply"`` writes one **atomic** season edit (anchor and
            workouts run in one transaction, so a bad part writes nothing):
            ``spec`` is JSON with a ``"workouts"`` section (creates/patches +
            ``delete_ids``) and/or an ``"anchor"`` section (goals/blocks/weeks),
            or ``{"undo": true}``. A destructive edit snapshots the whole season
            first, so one undo restores everything. The result reports the
            changed ids (``added_ids``/``updated_ids`` under ``"workouts"`` and
            the anchor's ``blocks``/``weeks``).

            Workout fields: planned_date, activity_type
            (run/cycle/swim/strength/rest/other), title, description (ONE short
            line, ~200 chars, summarising THIS session), duration_min, distance_km, intensity
            (easy/moderate/hard/race_pace), target_pace_min_km (decimal min/km,
            5.5 = 5:30), target_hr_zone, target_power_w, status
            (planned/completed/partial/skipped). Workouts automatically attach to
            the active block and goal by planned_date (goal_id is an optional manual
            override). Add ``steps`` whenever the session has more than one segment —
            any warm-up/work/recovery/cool-down structure, intervals, strides, a
            progressive long run, etc. (omit only for a single continuous
            effort): [{kind (warmup/steady/work/recovery/cooldown/rest), duration
            ('15m'/'90s'/'1:30'), repeat?, label? (1-2 words, no digits/time),
            intensity?, target_pace_min_km?, target_hr_zone?, target_power_w?}].
            An entry with an ``id`` PATCHes that workout; one without creates.
            ``delete_ids`` deletes workouts. To move or copy workouts, get them
            first and PATCH/create the ids you mean. ``title``/``description`` and a
            step ``label`` describe this session only — never rules, preferences or
            reminders; durable guidance belongs in ``memory``.

            ANCHOR section: ``goal_id`` updates that goal (omit to create; a
            blocks-only spec targets the sole goal rather than inventing a blank
            one); title/sport/start_date/target_date/target_time/
            target_distance_km are the target. ``blocks`` UPSERT blocks
            (with ``id`` patches, without creates): name, focus (short labels, not
            rule storage), optional
            start_date/end_date, target_weekly_km (optional baseline weekly volume),
            and optional ``weeks``: a per-week PATCH keyed by ``week_start`` — each
            entry {week_start, distance_km, duration_min, is_deload} writes only
            that week and every week you do not send is left untouched (changing one
            week only requires sending that week); a bare {week_start} entry removes
            that week and ``weeks: []`` clears them all. Set a block's
            target_weekly_km when its weeks vary so omitted weeks keep a baseline.
            ``delete_blocks``/``delete_goal_ids`` delete -- destructive but
            restorable with ``undo``.
            """
            if on_status is not None:
                on_status("Consulting the training season…")
            try:
                if action == "apply":
                    if spec is None:
                        return "ERROR: action 'apply' needs a 'spec' JSON string"
                    try:
                        parsed = json.loads(spec)
                    except json.JSONDecodeError as exc:
                        return f"ERROR: spec is not valid JSON: {exc}"
                    try:
                        result = season.apply(parsed)
                    except (ValueError, OSError, psycopg.Error) as exc:
                        return _tool_error(exc)
                    if not parsed.get("undo"):
                        result["guidance"] = (
                            "description, a step label and block name/focus describe "
                            "a row only — never rules, preferences or reminders; "
                            "durable guidance belongs in the memory tool"
                        )
                    return json.dumps(result, ensure_ascii=False)

                if action != "get":
                    return "ERROR: action must be 'get' or 'apply'"
                day_iso = day or date.today().isoformat()
                date.fromisoformat(day_iso)
                if (date_start is None) != (date_end is None):
                    return "ERROR: pass both date_start and date_end, or neither"
                if date_start is not None and weeks not in (None, 1):
                    return (
                        "ERROR: pass either 'weeks' or a date_start/date_end "
                        "range, not both"
                    )
                if date_start is not None:
                    try:
                        date.fromisoformat(date_start)
                        date.fromisoformat(date_end)
                    except ValueError:
                        return "ERROR: date_start/date_end must be YYYY-MM-DD"
                    if date_start > date_end:
                        date_start, date_end = date_end, date_start
                    win_start, win_end = date_start, date_end
                else:
                    span = max(1, min(int(weeks or 1), 26))
                    win_start = _monday_iso(day_iso)
                    win_end = (
                        date.fromisoformat(win_start)
                        + timedelta(days=7 * span - 1)
                    ).isoformat()
                span_weeks = (
                    date.fromisoformat(win_end) - date.fromisoformat(win_start)
                ).days // 7 + 1

                goals = season.list_goals()
                if goal_id is not None:
                    goals = [g for g in goals if g["id"] == int(goal_id)]
                    if not goals:
                        return f"ERROR: goal {goal_id} not found"

                if full:
                    out_goals = [
                        dumped for dumped in
                        (_anchor_goal_dump(season, g["id"], day_iso)
                         for g in goals)
                        if dumped is not None
                    ]
                else:
                    out_goals = [
                        _goal_header(season, g, win_start, win_end)
                        for g in goals
                    ]

                workouts = season.list_workouts(win_start, win_end)
                # ``goals`` is already narrowed to the requested goal, so the
                # same shared attribution rule the coverage numbers use applies.
                target_goal = goals[0] if goal_id is not None else None
                if target_goal is not None:
                    from .server.state import goal_activities, goal_workouts

                    workouts = goal_workouts(
                        target_goal, season.blocks(target_goal["id"]), workouts
                    )
                total = len(workouts)
                workouts = workouts[:_MAX_WORKOUTS]
                if not detail:
                    for w in workouts:
                        desc = w.get("description")
                        if desc and len(desc) > _DESC_PREVIEW_CHARS:
                            w["description"] = (
                                desc[:_DESC_PREVIEW_CHARS].rstrip() + "…"
                            )
                actual = season.activities(win_start, win_end)
                if target_goal is not None:
                    actual = goal_activities(target_goal, actual)
                targets = season.coverage_range(
                    win_start, win_end, goal_id
                )
                payload: dict[str, Any] = {
                    "day": day_iso,
                    "window": {
                        "from_date": win_start,
                        "to_date": win_end,
                        "weeks": span_weeks,
                    },
                    "can_undo": season.can_undo(),
                    "goals": out_goals,
                    "weeks": _build_weeks(
                        win_start, win_end, workouts, actual, targets
                    ),
                }
                if total > _MAX_WORKOUTS:
                    payload["workouts_truncated"] = {
                        "total": total, "shown": _MAX_WORKOUTS
                    }
                return json.dumps(payload, ensure_ascii=False)
            except (ValueError, OSError, psycopg.Error) as exc:
                return _tool_error(exc)

    return agent


def _build_agent(
    cfg: dict[str, str],
    db: ReadOnlyDB,
    *,
    memory: Any | None = None,
    principles: Any | None = None,
    training: Any | None = None,
    chart_cache: dict[str, Any] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Any:
    api_key = cfg["llm_api_key"] or None
    base_url = cfg["llm_base_url"] or None
    if not api_key and not base_url:
        raise RuntimeError(
            "no LLM configured: set OPENAI_API_KEY / LLM_API_KEY for cloud, "
            "or LLM_BASE_URL (e.g. http://host.docker.internal:11434/v1) with LLM_MODEL for a local Ollama model"
        )
    weather = Weather.from_config(cfg, db=db)
    return build_agent(
        db,
        model_name=cfg["llm_model"],
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=cfg.get("llm_reasoning_effort") or None,
        provider=cfg.get("llm_provider") or None,
        memory=memory,
        principles=principles,
        weather=weather,
        training=training,
        chart_cache=chart_cache,
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
    return ReadOnlyDB.from_url(url, user_id=user_id, excluded_types=excluded)


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


def _ask(cfg: dict[str, str], question: str) -> str:
    db = _open_readonly(cfg)
    from .server.state import (
        PgMemory, PgPrinciples, TrainingSeason, TrainingStore, UserState,
    )

    state = UserState(cfg["db_url"])
    training = TrainingStore(cfg["db_url"])
    user_id = cfg.get("local_user_id") or 1
    try:
        result = _build_agent(
            cfg,
            db,
            memory=PgMemory(state, user_id),
            principles=PgPrinciples(state, user_id),
            training=TrainingSeason(training, user_id),
        ).run_sync(question)
        _record_turn(
            cfg,
            question,
            result,
            trace_writer=lambda r: state.append_trace(user_id, r),
        )
        return _final_answer(result)
    finally:
        db.close()
        state.close()
        training.close()


def _seed_history_from_summary(summary: str) -> list[Any]:
    """Wrap a compacted summary as a one-message history for a new session."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    return [
        ModelRequest(
            parts=[UserPromptPart(content=f"{_SUMMARY_LABEL}\n\n{summary}")]
        )
    ]


def _system_prompt_head(messages: Iterable[Any]) -> Any | None:
    """Collect every ``SystemPromptPart`` from a history into one ``ModelRequest``.

    A resumed request only receives base system prompts when its history is empty
    (Pydantic AI injects ``_sys_parts`` only then), so compacting a session must
    carry the role/schema/date/memory parts forward or the model loses its
    instructions. Returns ``None`` when there are none.
    """
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    parts = [
        SystemPromptPart(content=p.content, dynamic_ref=p.dynamic_ref)
        for msg in messages
        for p in getattr(msg, "parts", [])
        if isinstance(p, SystemPromptPart)
    ]
    return ModelRequest(parts=parts) if parts else None


def _seed_after_system_prompts(
    messages: Iterable[Any], summary: str
) -> list[Any]:
    """Seed a fresh session with a summary, preserving the system-prompt head.

    The compacted transcript is dropped; the system-prompt parts (which carry
    the role, schema overview, date+freshness and memory prompts) are kept so a
    resumed session still has its instructions, followed by the summary as a
    user message.
    """
    head = _system_prompt_head(messages)
    seed = _seed_history_from_summary(summary)
    return ([head] if head is not None else []) + seed


def _part_text(part: Any) -> str:
    """Best-effort text of one message part, for a token estimate."""
    content = getattr(part, "content", None)
    if content is None:
        content = getattr(part, "args_json", None)
    if content is None:
        content = getattr(part, "args", None)
    if content is None:
        return str(part)
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _messages_token_estimate(messages: Iterable[Any]) -> int:
    """Rough token count for a conversation (≈ chars/4)."""
    total = sum(len(_part_text(p)) for msg in messages for p in getattr(msg, "parts", []))
    return max(1, round(total / 4))


def _auto_compact(
    agent: Any, messages: list[Any], *, max_tokens: int = _AUTO_COMPACT_MAX_TOKENS
) -> list[Any]:
    """Fold an over-budget conversation into a summary, or return it unchanged.

    Triggers once per session once the stored history (including raw tool
    results) exceeds ``max_tokens``. The one extra LLM call to produce the
    summary is paid only occasionally; the cheaper resumed sessions then start
    from the summary and the system-prompt head instead of the whole transcript.
    A compaction failure never loses the session — the history is kept as-is.
    """
    if not messages or _messages_token_estimate(messages) <= max_tokens:
        return messages
    try:
        summary = str(agent.run_sync(_COMPACT_INSTRUCTION, message_history=messages).output).strip()
    except Exception:
        return messages
    if not summary:
        return messages
    return _seed_after_system_prompts(messages, summary)


def _prune_session_messages(messages: list[Any]) -> list[Any]:
    """Sanitize ephemeral tool return payloads in stored message history.

    Replaces large raw SQL row dumps, schema listings, and bulky tool return
    payloads with concise markers (e.g. '[Query executed: N rows returned]'),
    while preserving the full conversational transcript (user prompts,
    system prompts, assistant final text, and tool-call identities). This
    keeps resumed sessions light and prevents context-window bloat over multi-turn
    conversations.
    """
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    pruned: list[Any] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            pruned.append(msg)
            continue
        new_parts = []
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart):
                new_parts.append(part)
                continue
            content = part.content
            content_str = str(content) if not isinstance(content, str) else content
            if len(content_str) > 350:
                tool = part.tool_name
                summary_content = None
                try:
                    parsed = json.loads(content_str)
                    if isinstance(parsed, dict):
                        if "rows" in parsed and "columns" in parsed:
                            n_rows = len(parsed.get("rows") or [])
                            cols = ", ".join(parsed.get("columns") or [])
                            summary_content = f"[Query executed: {n_rows} rows returned with columns ({cols})]"
                        elif "series" in parsed and "summary" in parsed:
                            pts = len(parsed.get("series") or [])
                            m = parsed.get("metric", "")
                            summary_content = f"[{m} trend: {pts} points, summary: {json.dumps(parsed.get('summary'))}]"
                        elif "workouts" in parsed:
                            wkts = len(parsed.get("workouts") or [])
                            summary_content = f"[Training plan: {wkts} workouts retrieved]"
                    elif isinstance(parsed, list):
                        if tool == "table_schema":
                            summary_content = f"[Table schema: {len(parsed)} columns inspected]"
                        elif tool == "get_recent_activities":
                            summary_content = f"[Recent activities: {len(parsed)} workouts retrieved]"
                except Exception:
                    pass
                if not summary_content:
                    summary_content = content_str[:250] + "… [output pruned for memory]"
                new_part = ToolReturnPart(
                    tool_name=part.tool_name,
                    content=summary_content,
                    tool_call_id=part.tool_call_id,
                    outcome=part.outcome,
                )
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        pruned.append(ModelRequest(parts=new_parts))
    return pruned


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
    from .server.state import (
        PgMemory, PgPrinciples, TrainingSeason, TrainingStore, UserState,
    )

    db = _open_readonly(cfg)
    state = UserState(cfg["db_url"])
    training = TrainingStore(cfg["db_url"])
    user_id = cfg.get("local_user_id") or 1
    try:
        agent = _build_agent(
            cfg,
            db,
            memory=PgMemory(state, user_id),
            principles=PgPrinciples(state, user_id),
            training=TrainingSeason(training, user_id),
        )
        history = state.get_session_messages(user_id) or []
        if history:
            _refresh_resumed_prompt(history, db)
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
                history = _seed_after_system_prompts(history, summary)
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
            print(_final_answer(result))
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