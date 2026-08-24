# Garmin Health-Data Agent

Fetch Garmin Connect data into Postgres, then ask a read-only AI agent
questions about it — sleep, resting HR, HRV, training load, activity zones —
and get answers with charts and weather context. A self-hosted web app
(Chat · Readiness · ACWR · Training Plan · Settings) keeps the derived
metrics and plan in the same database the agent queries, so the UI and the AI
always agree.

```
Garmin Connect ──▶ sync worker ──▶ Postgres (raw JSON, source of truth) ──▶ parser ──▶ typed tables
                                       ▲  (user_id + RLS on every row)        │
    JS frontend ─▶ FastAPI (JWT) ──▶ garmin-ask agent (read-only role) ◀──────┘
                                           │
                                     derived_metrics (readiness, ACWR)
```

Accounts authenticate with email+password (JWT), bind their own Garmin account
(credentials encrypted AES-GCM), and sync in background worker threads. Every
data row is scoped per account by Postgres Row-Level Security; the LLM agent
runs as a SELECT-only PG role, so it is provably confined to one user's rows
even if its statement gate is ever removed.

## What it does

- **Multi-user server.** Register/login with email+password; each account
  supplies its own Garmin credentials at signup and logs in once (two-step
  verification is handled as an extra code step), with the resulting OAuth
  tokens stored encrypted (AES-GCM, key from env). `POST /sync` triggers the
  caller's own sync; a cron endpoint iterates all active users. Syncs run on a
  bounded worker pool so one account never blocks another.
- **Web dashboard.** A same-origin JS frontend with tabs: **Chat** (the agent),
  **Readiness** and **ACWR** (custom derived scores with history charts),
  **Training Plan** (a weekly workout calendar you and the agent can edit,
  showing a stored daily weather forecast), and
  **Settings** (sync + per-account LLM/weather/data-type config).
- **Custom derived metrics.** Computed once per sync and stored in
  `derived_metrics`, visible to both the UI and the agent:
  - **Training readiness** (0-100): a weighted composite of nightly HRV
    (rmssd), resting HR and sleep-score z-scores against a rolling 28-day
    baseline (`min_samples=7`, HRV clamped at +2.0), auto-scaled through a
    rolling 90-day calibration so the score always reflects recent context.
    Categories: Prime / Moderate / Low / Depleted.
  - **ACWR** (acute-to-chronic workload ratio): `EMA₇(daily load) /
    EMA₂₈(daily load)`, where daily load is the sum of each day's activity
    training load. Categories: Sweet Spot (0.8-1.3) / Elevated (1.3-1.5) /
    Danger (>1.5) / Detraining (<0.8).
- **RLS-isolated data.** `user_id` is the leading key on every data table and
  Row-Level Security filters every read/write by
  `current_setting('app.user_id')`. The agent connects as a `garmin_readonly`
  role with SELECT only on the agent-facing tables — the raw stores are a real
  REVOKE.
- **Read-only AI agent.** A Pydantic AI agent over the same per-user tables.
  Safety is by construction: read-only PG role + RLS scoping, plus a statement
  gate (SELECT/WITH/EXPLAIN only) as a second layer.
- **Tool-calling agent.** The model inspects the schema, runs read-only SQL
  (including the `derived_metrics` table), validates and embeds Plotly chart
  specs (the UI re-runs the query to draw them), consults a stateless weather
  tool (Open-Meteo), keeps long-term memory, and can read/edit the training
  plan. Every data question is answered by the model writing its own SQL
  through `run_sql`, except per-day summaries which use the single
  `get_day_summary` shortcut (one call returns a day's metrics + activities
  instead of a ~60-column dump). Conversation and memory are per-user, in
  Postgres.
- **Weather forecast for the plan.** The Open-Meteo daily forecast is fetched
  only when a sync runs (via the worker) and stored per account in
  `weather_forecast`; the Training Plan calendar reads those rows, so no
  external API is hit on a page view. The agent's stateless `weather` tool still
  serves ad-hoc history and other locations.
- **Operation hardening.** Per-account exponential backoff on the sync worker
  protects against Garmin's informal API and ban risk; tokens are
  auto-refreshed and re-encrypted; signup fails fast and clearly whenever
  Garmin blocks or rate-limits the login.

## Data model

Every data table is keyed by `(user_id, …)` and RLS-scoped. `users` is the
separate identity table (email, password hash, encrypted Garmin creds/tokens,
sync status).

| Table | Purpose |
| --- | --- |
| `users` | Accounts: email + bcrypt hash + encrypted Garmin credentials/tokens, active/confirmed, sync status + backoff |
| `metrics` | Raw Garmin JSON payloads, keyed by `(user_id, data_type, calendar_date)` — the single source of truth |
| `daily_metrics` | Parsed daily projection, one wide row per date; columns auto-created on demand |
| `activities` | Raw activity summaries + intra-activity detail/weather/splits payloads, keyed by `(user_id, activityId)` |
| `activity_summaries` | Parsed activity projection (durations, HR, zones, elevation, power, cadence, weather) |
| `activity_detail_series` | Parsed intra-activity time series (HR / cadence / power / speed / elevation / GPS per tick) |
| `activity_splits` | Parsed per-lap splits (work/rest chunks: distance, duration, pace, HR, power, cadence) |
| `hr_zones` | Derived per-sport heart-rate zone ranges from the device profile |
| `power_zones` | Derived per-sport power-zone ranges (watts) + functional threshold power |
| `race_predictions` | Current race-prediction snapshot (5k/10k/half/full finish times) |
| `user_profile` | Raw profile snapshots (HR zones, power zones, race predictions, gear, devices) |
| `gear` | Current gear snapshot (bikes, shoes, ...) with cumulative stats — replaced each sync, no history |
| `devices` | Current Garmin devices (model + which is primary) — replaced each sync, no history |
| `derived_metrics` | Computed daily metrics — one row per `(calendar_date, metric)`; `readiness` (+ `readiness_*` components) and `acwr` (+ acute/chronic/daily load). Replaced wholesale each sync |
| `weather_forecast` | Stored daily Open-Meteo forecast (min/max °C, precip mm, max wind), refreshed once per sync — the Training Plan calendar renders it from here, RLS-scoped per account |
| `training_plan` | Per-user planned workouts (editable in the UI and by the agent) |
| `user_state` | Per-user agent state: long-term memory, conversation history, tool-call trace |

## Setup

Requirements: Python 3.14, [`uv`](https://docs.astral.sh/uv/) and Docker.

```sh
git clone https://github.com/adamzelina1/garmin-agent-web.git
cd garmin-agent-web
uv sync

# 1. Start Postgres + server (roles garmin_app / garmin_readonly are
#    bootstrapped on a fresh volume by docker/initdb/01-roles.sh)
docker compose up -d

# 2. Configure secrets in .env (see .env.example): GARMIN_ENC_KEY,
#    GARMIN_JWT_SECRET, GARMIN_CRON_TOKEN and the three DSNs.
cp .env.example .env
```

For an already-running database, create the roles once (as superuser):

```sh
uv run python -c "from garmin_fetch.server.setup_db import ensure_roles; import os
ensure_roles(os.environ['GARMIN_ADMIN_DB_URL'], os.environ['GARMIN_DB_URL'], os.environ['GARMIN_READONLY_DB_URL'])"
```

## Usage

```sh
uv run garmin-server --port 8000        # open http://127.0.0.1:8000
```

The page handles register/login (a single-step Garmin bind: signup logs into
Garmin once to verify the credentials — MFA, wrong password, or a
rate-limit/Cloudflare block all fail signup with a clear message), a
"Sync now" button with status, a chat box that renders charts, and the
derived-metric tabs. Readiness and ACWR are recomputed automatically on every
sync once enough data has landed, so the tabs need no manual refresh beyond a
sync. Every request carries the JWT; the agent only ever sees the
authenticated user's rows.

Direct API examples:

```sh
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"password123"}' | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')

curl -s -X POST http://127.0.0.1:8000/sync -H "Authorization: Bearer $TOKEN"
curl -s http://127.0.0.1:8000/readiness -H "Authorization: Bearer $TOKEN"   # readiness series + today + scale
curl -s http://127.0.0.1:8000/acwr -H "Authorization: Bearer $TOKEN"        # ACWR series + today
curl -s "http://127.0.0.1:8000/weather?from_date=2026-08-24&to_date=2026-08-30" -H "Authorization: Bearer $TOKEN"   # stored daily forecast
curl -s -X POST http://127.0.0.1:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"question":"avg sleep last week?"}'

# daemon cron (configure GARMIN_CRON_TOKEN):
curl -s -X POST http://127.0.0.1:8000/cron/sync -H "Authorization: Bearer $GARMIN_CRON_TOKEN"
```

Single-user CLI (optional, legacy): `uv run garmin-fetch` syncs
`GARMIN_LOCAL_USER_ID`; `uv run garmin-ask "…"` asks the same agent locally.

## Configuration

`.env` holds only server infrastructure (see `.env.example`). Everything a
user can set lives in the website (Settings → Config) per account, stored in
the `users` table — Garmin credentials/tokens (encrypted), LLM API key / base
URL / model, home lat/lon for the weather tool, excluded data types, and the
sync start date.

Server-level `.env` settings:

- `GARMIN_DB_URL` / `GARMIN_READONLY_DB_URL` / `GARMIN_ADMIN_DB_URL` — app,
  read-only-agent, and superuser DSNs
- `GARMIN_ENC_KEY` — base64 of a 32-byte AES-GCM key (generate with
  `uv run python -c "import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())"`)
- `GARMIN_JWT_SECRET`, `GARMIN_CRON_TOKEN`, `GARMIN_JWT_TTL_HOURS`
- `GARMIN_SYNC_INTERVAL_MIN`, `GARMIN_SYNC_MAX_WORKERS` — worker tuning
- `GARMIN_ACTIVITY_FREEZE_DAYS` — trailing days of activities re-scanned for
  late uploads
- `GARMIN_LOCAL_USER_ID` — which account the single-user CLI (`garmin-ask` /
  `garmin-trace`) reads and writes

Agent state never touches disk: each user's long-term memory, conversation
history and tool-call trace live in the `user_state` table (RLS-scoped), so a
returning web session — or a later `garmin-ask` — resumes exactly where the
last one left off.

Per-account sync window (set on the website): each account's excluded data
types (the fetchable daily-metric set is `DAILY_TYPES` in `datatypes.py`) and
its backfill start date. The settings UI shows a friendly label + description
per type, so you can toggle off the ones you don't have or want (e.g. "Cycling
FTP" unless you ride with a power meter; "Pulse Ox" without an SpO2 watch).
Note that running FTP lives under `lactate_threshold` (`running_ftp_watts`,
running power) and is a different number from cycling FTP (`cycling_ftp_watts`).
Changing these takes effect retroactively on the next sync: dates before the
start date are deleted, and newly-excluded types have their raw rows removed
and their exclusive `daily_metrics` columns NULLed (shared columns kept).

The legacy single-user CLI (`garmin-fetch` / `garmin-ask`) can still read
`GARMIN_EMAIL` / `GARMIN_PASSWORD` / `GARMIN_TOKENS_PATH` and LLM settings from
the environment, but the server ignores them.

## Derived metrics

Both scores are derived in `src/garmin_fetch/derived.py` and stored in
`derived_metrics` once per sync (after parsing), so the agent can answer
questions about them from the same table the UI renders.

- **Training readiness** (`readiness.py`) — 28-day trailing baselines
  (`min_samples=7`) for nightly HRV, resting HR and sleep score; z-scores
  (HRV clamped at `+2.0`, resting HR inverted); composite
  `0.50·Z_HRV + 0.30·Z_RHR + 0.20·Z_Sleep`; then auto-scaled to 0-100 via
  `SCORE_ANCHORS` calibrated against the trailing 90 days of composites (per
  day, never including the day itself, so no future leakage). Adjust
  `SCORE_ANCHORS` / `SCALE_WINDOW_DAYS` in `readiness.py` to reshape.
- **ACWR** (`workload.py`) — daily training load summed from
  `activity_summaries.training_load`; `acute = EMA₇`, `chronic = EMA₂₈`,
  `ACWR = acute/chronic`. Rest days count as zero load, so tapers show up as a
  falling ratio.

## Security

- Per-user Garmin credentials and OAuth tokens are AES-GCM encrypted at rest
  (`GARMIN_ENC_KEY`); tokens auto-refresh and are re-encrypted after each sync.
  They are never logged.
- RLS is the primary boundary: runtime roles are non-superuser
  (`garmin_app` writes, `garmin_readonly` reads), and every data table carries
  `user_id` with a FORCED RLS policy on `current_setting('app.user_id')`.
- The agent's statement gate + `_ALLOWED_TABLES` list remain as a second layer;
  the read-only role's REVOKEs are the real enforcement.
- Background syncs back off exponentially per account on failure, and signup
  verifies the Garmin credentials by logging in once — wrong credentials or a
  rate-limit/Cloudflare block fail signup with a clear message, and two-step
  verification keeps the account pending until the code is confirmed.
- Run the server behind TLS for anything beyond localhost.

## Tech stack

Python 3.14 · `uv` · PostgreSQL 17 (RLS) · FastAPI · APScheduler · Pydantic AI ·
Gradio (legacy single-user CLI UI) · Plotly · Open-Meteo · garminconnect ·
PyJWT · bcrypt · `cryptography` (AES-GCM)

## Project layout

```
src/garmin_fetch/
  db.py         # Postgres-only schema (users + user_id data tables), RLS setup
  fetcher.py    # Garmin Connect sync (per-user, token-string store)
  parser.py     # raw JSON -> typed table projections
  datatypes.py  # data-type registry
  derived.py    # rebuilds all derived metrics (readiness + ACWR) into derived_metrics
  readiness.py  # custom training-readiness score (z-composite + rolling auto-scale)
  workload.py   # ACWR (7d/28d EMA of daily training load)
  ask.py        # read-only AI agent (ReadOnlyDB, statement gate, RLS-scoped)
  ask_web.py    # legacy single-user Gradio UI
  trace.py      # per-turn tool-call tracing
  server/
    app.py        # FastAPI routes + static JS frontend
    auth.py       # users table, bcrypt, JWT, single-step signup
    crypto.py     # AES-GCM for Garmin credentials/tokens
    sync_worker.py# background sync pool + APScheduler cron + backoff
    setup_db.py   # garmin_app / garmin_readonly role bootstrap
    state.py      # per-user memory/session + training-plan store
    static/       # index.html (Chat · Readiness · ACWR · Training Plan · Settings)
docker/
  initdb/       # 01-roles.sh (creates non-superuser roles on fresh volume)
```
