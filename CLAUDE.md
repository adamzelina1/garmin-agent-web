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

There is no test suite and no linter configured. The project is deliberately test-free — use ephemeral one-off scripts in the scratchpad if you need to verify something. There is also deliberately **no migration/backfill logic**; `create_schema` is idempotent and that is all.

## Reading the database (important)

Every data table is RLS-scoped by `user_id` via `current_setting('app.user_id')`, so a plain connection sees **zero rows**. Set the session user first:

```python
import psycopg
c = psycopg.connect("postgresql://garmin_app:garmin_app@localhost:5432/garmin")
c.execute("SELECT set_config('app.user_id','1',false)")   # id passed as text
```

An empty result almost always means a wrong/unset `app.user_id`, not an empty DB. Roles: `garmin_app` (owner/writer, used at runtime), `garmin_readonly` (SELECT-only agent role, real REVOKEs), `garmin` (superuser, bypasses RLS — only for role bootstrap via `GARMIN_ADMIN_DB_URL`; never use at runtime). DSNs live in `.env` (gitignored; never print secrets).

## Architecture

Data flow: Garmin Connect → sync worker → raw JSON in `metrics`/`activities` (source of truth) → `parser.py` → typed projection tables (`daily_metrics`, `activity_summaries`, `activity_detail_series`, `activity_splits`, …) → `derived.py` computes `derived_metrics` (ACWR, run ACWR, cadence drift) once per sync, replaced wholesale. UI and agent read the same derived values.

- `db.py` — `Database(url, user_id=...)` is bound to one user; `merge_daily` auto-creates columns on demand; `prune_dates_before` / `prune_excluded_types` apply config changes retroactively on the next config-driven full sync (explicit `--range`/`--type` syncs never prune). `parser.TYPE_COLUMNS` maps each data type to the `daily_metrics` columns it may write.
- `fetcher.py` — `sync_data(...)`; never prompts for MFA; the `widget+cffi` login strategy is always skipped (falsely reports "MFA required" during Cloudflare/429 windows). Also refreshes the stored weather forecast (`refresh_weather_forecast`); `/weather` only reads the table.
- `ask.py` — the agent. Read-only via `garmin_readonly`; a statement gate (SELECT/WITH/EXPLAIN) plus an `_ALLOWED_TABLES` regex as a second layer. Tools include `run_sql`, fast helpers (`get_day_summary`, `get_metric_trend`, `get_recent_activities`), `chart`, `weather`, `memory`, and training tools. Agent writes to plan/memory/session go through app stores, never the read-only connection. Prompt bullets for plan/goal tools are only emitted when those stores are passed to `build_agent`. Tool outputs are sanitized before storing to avoid context bloat; `/ask` streams over SSE.
- Training model: goals → `training_block` phases → `training_week` per-week targets → `training_workout` rows (with `status` and target pace/HR zone/power). Resolution of current block/week/target lives in one place (`TrainingGoalStore.resolve_goal`), shared by the API and the agent's `training` tool. Undo is a single season-wide snapshot stored in `user_state`. Workout updates are id-keyed partial PATCHes.
- `server/` — `app.py` (`create_app(cfg)`, all routes), `auth.py` (UserStore, bcrypt, JWT; signup logs into Garmin once to verify creds and handles the two-step MFA challenge via `/auth/register/mfa`), `crypto.py` (AES-GCM for creds/tokens), `sync_worker.py` (thread pool + APScheduler cron; per-account exponential backoff via `rate_limit_until`, 30 min ×2^n capped at 8 h, ban detection jumps to 8 h), `state.py` (per-user agent memory/session/trace in `user_state`), `geocode.py`, `setup_db.py` (role bootstrap).
- Config: `config.py` + `.env` (see `.env.example`); LLM provider is server-wide (`LLM_*`), per-account settings live in the `users` table.

## Security invariants

RLS is the security boundary: every data table has `user_id` as PK prefix, `FORCE ROW LEVEL SECURITY`, and a `user_isolation` policy; `users` is exempt. Preserve this when adding tables (new agent-visible tables also need a `garmin_readonly` GRANT and an `_ALLOWED_TABLES` entry).
