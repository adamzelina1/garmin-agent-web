# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Postgres-backed, multi-user Garmin Connect fetcher + FastAPI server (JWT auth, background sync worker, read-only Pydantic AI agent, same-origin JS frontend in `src/garmin_fetch/server/static/index.html`). Python 3.14, `uv` only. Developed on **Windows**.

## Commands

```sh
uv sync                              # install deps
docker compose up -d --build         # Postgres (garmin-db) + server image; .env must exist first (see .env.example)
uv run garmin-server --port 8000     # run the API/UI locally against the containerised DB
uv run garmin-fetch --parse --full   # single-user CLI sync + full re-parse for GARMIN_LOCAL_USER_ID
uv run garmin-ask "avg sleep last week?"   # ask the agent locally (also: garmin-ask-web, garmin-trace)
uv add <pkg>                         # add a dependency
```

There is no test suite and no linter configured. The project is deliberately test-free — use ephemeral one-off scripts in the scratchpad if you need to verify something. There is also deliberately **no migration/backfill logic in the codebase**; `create_schema` is idempotent and that is all. When a change leaves old data behind, clean it up with a temporary one-time script (scratchpad, run once, never committed).

## No Legacy Guards or Backward-Compatibility Shims

- Always write code that reflects the current, canonical state of the codebase.
- Assume a clean cutover for all changes: update all callers, types, and references directly, and delete superseded code cleanly.
- Never add defensive fallbacks, adapters, dual-format checks, or migration guards for previous code behavior or old data shapes (e.g., avoid `newProp ?? oldProp`, deprecated parameter overloads, or runtime checks for legacy formats).
- Only write migration or backward-compatibility logic into the codebase if I explicitly ask for it. A temporary one-time script that migrates or cleans up existing data is fine and expected: propose it with the change, then run it once outside the codebase.

## Reading the database (important)

Every data table is RLS-scoped by `user_id` via `current_setting('app.user_id')`, so a plain connection sees **zero rows**. Set the session user first:

```python
import psycopg
c = psycopg.connect("postgresql://garmin_app:garmin_app@localhost:5432/garmin")
c.execute("SELECT set_config('app.user_id','1',false)")   # id passed as text
```

An empty result almost always means a wrong/unset `app.user_id`, not an empty DB. Roles: `garmin_app` (owner/writer, used at runtime), `garmin_readonly` (SELECT-only agent role, real REVOKEs), `garmin` (superuser, bypasses RLS — only for role bootstrap via `GARMIN_ADMIN_DB_URL`; never use at runtime). DSNs live in `.env` (gitignored; never print secrets).

## Architecture

Data flow: Garmin Connect → sync worker → raw JSON in `metrics`/`activities` (source of truth) → `parser.py` → typed projection tables (`daily_metrics`, `activity_summaries`, `activity_detail_series`, `activity_splits`, …) → `derived.py` computes `derived_metrics` (ACWR, run ACWR, cadence drift, learned pace per HR zone from `zone_pace.py`) once per sync, replaced wholesale. Planned workouts' `duration_min` is computed from their steps and a run's `distance_km` is always an estimate from those steps at the zone paces (never typed in), priced on save and re-priced for still-planned runs on every sync when the paces are rebuilt (`zone_pace.reprice_planned_runs`). Volume is planned in minutes only: a week's target is `training_week.duration_min` and km is shown as an estimate. UI and agent read the same derived values.

- `db.py` — `Database(url, user_id=...)` is bound to one user; `merge_daily` auto-creates columns on demand; `prune_dates_before` / `prune_excluded_types` apply config changes retroactively on the next config-driven full sync (explicit `--range`/`--type` syncs never prune). `parser.TYPE_COLUMNS` maps each data type to the `daily_metrics` columns it may write.
- `fetcher.py` — `sync_data(...)`; never prompts for MFA; the `widget+cffi` login strategy is always skipped (falsely reports "MFA required" during Cloudflare/429 windows). Also refreshes the stored weather forecast (`refresh_weather_forecast`); `/weather` only reads the table. The container runs in UTC, so "today" and "is this day finished" use the user's offset read from their own data (`user_utc_offset`), never `date.today()`. Profile snapshots (zones, race predictions, gear, devices) refresh on the first sync of the user's day only (`profile_fetched_on` in `sync_state`). Activities in the rolling window are re-compared with Garmin's list each sync: changed ones are re-stored, ones deleted on Garmin are removed. Parse passes write in one `db.batch()` transaction.
- `ask.py` — the agent. Read-only via `garmin_readonly`; a statement gate (SELECT/WITH/EXPLAIN, `_BLOCKED_FUNCTIONS`) plus an `_ALLOWED_TABLES` regex as a second layer. `ReadOnlyDB` binds the user with `SET LOCAL app.user_id` (never `set_config`) and a 20 s `statement_timeout`; the server passes it one shared pool. Tools include `run_sql`, fast helpers (`get_day_summary`, `get_metric_trend`, `get_recent_activities`), `chart`, `weather`, `memory`, and training tools. Agent writes to plan/memory/session go through app stores, never the read-only connection. Prompt bullets for plan/goal tools are only emitted when those stores are passed to `build_agent`. Tool outputs are sanitized before storing to avoid context bloat; `/ask` streams over SSE. DeepSeek sometimes writes a tool call into its reply text (`<memory facts='…'</｜｜DSML｜｜ parameter>…`) instead of calling it: `chat_format.strip_leaked_tool_call` cuts that markup from streamed, final and stored text, and an output validator in `build_agent` sends the model back once to make the call for real.
- Training model: ONE goal per account → dated `training_block` phases → `training_week` per-week minute targets (addressed by date) → `training_workout` rows (with `status` and target pace/HR zone/power; steps are one entry per rep). Resolution of current block/week/target lives in one place (`TrainingAnchorStore.resolve_goal`), shared by the API and the agent's `training` tool. Every plan write — agent and UI routes alike — goes through `TrainingStore.apply`. Undo is a single season-wide snapshot stored in `user_state`. Workout updates are id-keyed partial PATCHes.
- `server/` — `app.py` (`create_app(cfg)`, all routes), `auth.py` (UserStore, bcrypt, JWT; signup logs into Garmin once to verify creds and handles the two-step MFA challenge via `/auth/register/mfa`), `crypto.py` (AES-GCM for creds/tokens), `sync_worker.py` (thread pool + APScheduler cron; per-account exponential backoff via `rate_limit_until`, 30 min ×2^n capped at 8 h, ban detection jumps to 8 h), `state/` (package: `user_state.py` agent memory/session/trace in `user_state`; `training_*.py` the season — `training_store.TrainingStore.apply` is the single transactional writer, `training_rules` holds the pure week/goal-matching rules), `geocode.py`, `setup_db.py` (role bootstrap).
- Config: `config.py` + `.env` (see `.env.example`); LLM provider is server-wide (`LLM_*`), per-account settings live in the `users` table.

## Security invariants

RLS is the security boundary: every data table has `user_id` as PK prefix, `FORCE ROW LEVEL SECURITY`, and a `user_isolation` policy; `users` is exempt. Preserve this when adding tables (new agent-visible tables also need a `garmin_readonly` GRANT and an `_ALLOWED_TABLES` entry).

RLS keys off the `app.user_id` setting, so `garmin_readonly` must never be able to change it: `setup_db.ensure_roles` revokes `EXECUTE` on `set_config` from PUBLIC (granting it back to `garmin_app` only) and the server refuses to start if the agent role can still call it. Any code that runs SQL as the agent role must set the user with `SET LOCAL`, not `set_config`.
