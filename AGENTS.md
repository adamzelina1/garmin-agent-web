# AGENTS.md

## Project

Postgres-backed multi-user Garmin Connect fetcher + FastAPI server (JWT auth, background sync, read-only AI agent). Python 3.14, `uv` only (no system python). **Windows.**

## Commands

```sh
uv run garmin-server --port 8000   # FastAPI server (auth + sync + ask + JS frontend)
uv run garmin-fetch --parse --full # single-user CLI sync/parse for GARMIN_LOCAL_USER_ID
uv run garmin-ask "avg sleep last week?"
uv add <pkg>                       # install deps into the managed venv
```

## Reading the database (important)

The live store is Postgres in the `garmin-db` container (roles `garmin_app`,
`garmin_readonly`, admin `garmin`). **Every data table is RLS-scoped by
`user_id` via `current_setting('app.user_id')`**, so a plain connection sees
**zero rows**. To inspect data you MUST set the session user first, e.g.:

```python
import psycopg
c = psycopg.connect("postgresql://garmin_app:garmin_app@localhost:5432/garmin")
c.execute("SELECT set_config('app.user_id','1',false)")   # user_id 1 = adam.zelina20@gmail.com
# now SELECT works and RLS scopes rows to that account
```

Notes:
- `garmin_app` = writer (syncs + auth); `garmin_readonly` = SELECT-only agent
  role (real REVOKEs, only the agent-facing tables); `garmin` = superuser (bypasses
  RLS; never use at runtime). DSNs in `.env`.
- `SET app.user_id` without the id string quoted as text (or a `'0'`/unset)
  also yields 0 rows — an empty result usually means the session id is wrong,
  not that the DB is empty.
- Raw payloads live in `metrics` (daily, keyed `data_type+calendar_date`) and
  `activities` (summary + details_json + weather_json + splits_json). Parsed
  projections are in `daily_metrics`, `activity_summaries`,
  `activity_detail_series`, `activity_splits`, `hr_zones`, `power_zones`,
  `race_predictions`, `gear`, `devices`. Computed daily metrics (readiness,
  ACWR, ...) live in `derived_metrics` (one row per `calendar_date`+`metric`,
  replaced wholesale each sync). The daily weather forecast lives in
  `weather_forecast` (one row per `calendar_date`), replaced wholesale each
  sync by `fetcher.refresh_weather_forecast` — `/weather` just reads it, so a
  page view never calls Open-Meteo.
- `GARMIN_ADMIN_DB_URL` (`garmin:garmin`) is used only for role bootstrap.

API (all JSON, `Authorization: Bearer <JWT>`): `POST /auth/register|login`, `GET /auth/me`, `POST /sync`, `GET /sync/status`, `POST /cron/sync` (daemon token), `GET /readiness`, `GET /acwr`, `GET|POST /training-plan`, `PUT|DELETE /training-plan/{workout_id}`, `GET /weather`, `POST /ask`, `POST /ask/chart`, `GET /`.

## Architecture

- Fetcher stores raw JSON in `metrics` (source of truth); parser projects to typed tables.
- **RLS is the security boundary** — every data table has `user_id` PK-leading; `FORCE ROW LEVEL SECURITY` + `user_isolation` policy reading `current_setting('app.user_id')`. `users` is exempt. Runtime connections use `garmin_app` (owner) / `garmin_readonly` (SELECT-only agent role, an actual REVOKE). Superusers bypass RLS → never run as superuser.
- `db.py`: `Database(url, user_id=...)` bound to one user; `create_schema` idempotent; `merge_daily` auto-creates columns.
- `fetcher.py`: `sync_data(...)`, never prompts MFA; **`widget+cffi` login strategy is always skipped** (falsely reports "MFA required" during Cloudflare/429 windows).
- `ask.py`: read-only Pydantic AI agent via `garmin_readonly`; statement gate (SELECT/WITH/EXPLAIN) + `_ALLOWED_TABLES` regex as a second layer. Tools: list_tables, table_schema, date_range, today, run_sql, get_day_summary (one call returns a full day's metrics + activities — a token-saver for "how was my day X"), chart (spec only), weather, memory (get/remember/forget), training-plan (get/update). Every other data question is answered by the model writing its own SQL through `run_sql`.
- `server/`:
  - `auth.py`: UserStore, bcrypt, JWT; signup verifies the Garmin credentials by logging in once — invalid creds → 400, rate-limit/Cloudflare → 502 (both delete the account). If Garmin requires a verification code (two-step), `register` returns `mfa_required` + a `challenge` and keeps the account unconfirmed; `POST /auth/register/mfa` completes it via `confirm_mfa` (in-memory, TTL'd). Creds/tokens AES-GCM encrypted (`crypto.py`).
  - `sync_worker.py`: thread pool + APScheduler cron; exponential backoff per account (`rate_limit_until`, 30min ×2^n cap 8h; ban detection jumps straight to 8h); marks `confirmed` on first success.
  - `app.py`: `create_app(cfg)`; agent memory/session/trace live per user in the `user_state` table.
- Config via `config.py` + `.env` (gitignored): `GARMIN_DB_URL`, `GARMIN_READONLY_DB_URL`, `GARMIN_ADMIN_DB_URL`, `GARMIN_ENC_KEY`, `GARMIN_JWT_SECRET`, `GARMIN_CRON_TOKEN`, `GARMIN_SYNC_INTERVAL_MIN`, `GARMIN_SYNC_MAX_WORKERS`, `GARMIN_LOCAL_USER_ID`. Never print secrets.

## Constraints

- **No migration/backfill logic** — user explicitly wants none.
- **No persisted tests** — consider the project test-free; only ephemeral one-off scripts in temp space when needed.
- Sync is idempotent but not byte-identical (parsing is marker-tracked: `metrics.parsed_at` etc.; `--parse --full` forces rebuild).
- Config changes take effect retroactively on the next config-driven full sync (`sync_data` with no `types`/`start`): `db.prune_dates_before(start)` deletes raw+projected rows before the account's start date, and `db.prune_excluded_types(excluded, enabled)` deletes raw rows of newly-excluded types and NULLs their exclusive `daily_metrics` columns (shared columns are kept — `TYPE_COLUMNS` in `parser.py` maps each type to the columns it can write). Explicit `--range` or `--type` syncs never prune.
- MFA supported at signup (two-step); Garmin rate-limit/Cloudflare blocks fail the bind with a clear message.
- Token refresh: garminconnect auto-refreshes; worker re-encrypts changed tokens after sync.