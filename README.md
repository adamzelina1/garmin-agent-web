# Garmin Health-Data Agent

> Ask questions about your Garmin data in plain English, such as sleep, HRV,
> resting HR, training load or running form, and get answers backed by your own
> numbers, with charts, weather context and a training plan that both you and
> the AI can edit.

This is a self-hosted, multi-user app. It syncs Garmin Connect into Postgres and
puts a **read-only AI agent** on top of that data. The dashboard and the agent
read the same derived metrics, so a chart on screen and an answer in chat always
agree.

![Chat with the agent: a dual-axis chart with a short analysis](images/chat.png)

## Highlights

- **An agent that actually analyses your data.** It writes its own read-only SQL
  across days, activities and lap splits, then reasons over what comes back.
  That covers lagged correlations ("does last night's HRV predict today's
  pace?"), root-causing a bad session from sleep, prior load and weather, and
  spotting form breakdown under fatigue.
- **Training load you can trust.** ACWR, a running-only ACWR (so cycling can't
  mask a running spike) and a pace-normalised cadence-drift score are computed
  once per sync and shared by the UI and the agent.
- **A training plan you share with the AI.** Goals are split into periodized
  blocks, then weekly targets, then dated workouts. You can drag workouts
  between days, see the forecast on the calendar, or ask the agent to reshape
  the week. One undo reverts the last destructive edit.
- **Multi-user and isolated by design.** Every row carries a `user_id` and is
  protected by Postgres Row-Level Security. Garmin credentials are AES-GCM
  encrypted, and the agent connects through a SELECT-only role.
- **Gentle on Garmin.** Background sync runs on a bounded worker pool with
  per-account exponential backoff and ban detection. MFA is handled as a
  separate code step at signup.

<table>
  <tr>
    <td><img src="images/today_tab_v2.png" alt="Today tab with ACWR and Running ACWR scorecards"></td>
    <td><img src="images/training_plan_v4.png" alt="Training Plan calendar with weather and daily briefing"></td>
  </tr>
  <tr>
    <td align="center"><b>Today</b>: ACWR and Running ACWR scorecards</td>
    <td align="center"><b>Training Plan</b>: calendar, weather and briefing</td>
  </tr>
</table>

## How it works

```
Garmin Connect ─▶ sync worker ─▶ raw JSON (metrics, activities)   ◀── source of truth
                                      │
                                      ▼
                                   parser ─▶ typed tables ─▶ derived_metrics (ACWR, run ACWR, drift)
                                                   │                 │
   browser (JS) ─▶ FastAPI + JWT ─▶ agent (garmin_readonly role) ◀───┘
                                     every table RLS-scoped by user_id
```

1. **Sync.** The worker pulls daily metrics, activities (including per-second
   detail and lap splits), profile data and an Open-Meteo forecast for
   each account.
2. **Store raw, then parse.** Raw JSON is kept as the source of truth, and
   `parser.py` projects it into typed tables. `daily_metrics` grows new columns
   as new fields appear.
3. **Derive.** `derived.py` rebuilds `derived_metrics` wholesale after every
   sync.
4. **Serve.** FastAPI serves the single-page UI and a JSON API. `/ask` streams
   the agent's answer over SSE.

## Quick start

You need **Python 3.14+**, [`uv`](https://docs.astral.sh/uv/) and Docker.

```sh
git clone https://github.com/adamzelina1/garmin-agent-web.git
cd garmin-agent-web
uv sync
cp .env.example .env      # then fill in the secrets (see Configuration)
docker compose up -d --build
```

This starts Postgres (`garmin-db`) and the app (`garmin-server`) on
<http://127.0.0.1:8000>. On a fresh volume, `docker/initdb/01-roles.sh` creates
the non-superuser `garmin_app` and `garmin_readonly` roles.

To run the server from source against the containerised database instead:

```sh
uv run garmin-server --port 8000
```

If you point the app at an existing database, bootstrap the roles once as
superuser:

```sh
uv run python -c "from garmin_fetch.server.setup_db import ensure_roles; import os
ensure_roles(os.environ['GARMIN_ADMIN_DB_URL'], os.environ['GARMIN_DB_URL'], os.environ['GARMIN_READONLY_DB_URL'])"
```

Then open the page and register. Signup logs into Garmin once to verify your
credentials and asks for a verification code if Garmin requires one. After
that, press **Sync now**.

## Using the app

The UI has four tabs:

| Tab | What's there |
| --- | --- |
| **Chat** | Streaming conversation with the agent, with inline charts. History persists per account and resumes on reload. Includes **New chat** and **Compact** controls and a live context-token meter. |
| **Today** | Daily briefing, ACWR and Running ACWR scorecards. Click a card to see zones and history. |
| **Training Plan** | Weekly calendar with weather, drag-and-drop workouts, weekly planned-vs-actual progress, goals and blocks. |
| **Settings** | Sync controls (**Sync now**, **Sync + full re-parse**, auto-sync toggle) and per-account config: home city, excluded data types, sync start date (with **Detect from Garmin**), reasoning effort, training principles and agent memory. |

### Settings that apply retroactively

Changing excluded data types or the sync start date takes effect on the next
sync. Dates before the start date are deleted. Newly excluded types lose their
raw rows, and their exclusive `daily_metrics` columns are set to NULL (shared
columns are kept). Excluded columns are also hidden from the agent.

**Detect from Garmin** finds the first day of an account's history. It anchors
on the oldest activity, gallops backwards probing daily summaries, and then
binary-searches to the exact day. A gap where the watch wasn't worn is not
mistaken for the start.

### API

Every route except `/`, register and login needs `Authorization: Bearer <token>`.

```sh
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"..."}' | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')
H="Authorization: Bearer $TOKEN"

curl -s -X POST localhost:8000/sync       -H "$H"   # incremental sync
curl -s -X POST localhost:8000/sync/full  -H "$H"   # sync + full re-parse from raw
curl -s localhost:8000/acwr               -H "$H"   # ACWR series + today
curl -s localhost:8000/run-acwr           -H "$H"   # running-only ACWR
curl -s "localhost:8000/activities?from_date=2026-08-01&to_date=2026-08-31" -H "$H"
curl -s "localhost:8000/weather?from_date=2026-08-24&to_date=2026-08-30"    -H "$H"
curl -s -X POST localhost:8000/ask -H "$H" -H 'Content-Type: application/json' \
  -d '{"question":"avg sleep last week?"}'
```

| Group | Routes |
| --- | --- |
| Auth & config | `POST /auth/register`, `POST /auth/register/mfa`, `POST /auth/login`, `GET /auth/me`, `GET`/`PUT /auth/config` |
| Sync | `POST /sync`, `POST /sync/full`, `POST /sync/detect-start`, `GET /sync/status`, `POST /cron/sync` (uses `GARMIN_CRON_TOKEN`) |
| Data | `GET /acwr`, `GET /run-acwr`, `GET /activities`, `GET /weather` |
| Training | `GET`/`POST /training/workouts`, `PATCH`/`DELETE /training/workouts/{id}`, `GET`/`POST /training/anchor`, `PATCH`/`DELETE /training/anchor/{goal_id}`, `POST /training/undo` |
| Agent | `POST /ask` (SSE), `GET /ask/history`, `POST /ask/clear`, `POST /ask/compact`, `POST /ask/chart` |

### CLI

These single-user tools act on the account set by `GARMIN_LOCAL_USER_ID`.

```sh
uv run garmin-fetch                        # incremental sync of all types
uv run garmin-fetch --parse --full         # full re-parse of stored raw data
uv run garmin-fetch --range 2026-01-01 2026-02-01 --type sleep
uv run garmin-ask "avg sleep last week?"   # ask the agent from the terminal
uv run garmin-ask-web                      # legacy Gradio chat UI
uv run garmin-trace                        # inspect the agent's tool-call trace
```

Syncs with an explicit `--range` or `--type` never prune data.

## The agent

The agent is built on Pydantic AI and is read-only at every layer:

1. It connects as `garmin_readonly`, a SELECT-only role with real REVOKEs.
2. RLS limits it to the signed-in user's rows.
3. A statement gate allows only `SELECT`, `WITH` and `EXPLAIN`, and an
   allowed-tables list blocks raw and internal tables.

| Tool | Purpose |
| --- | --- |
| `table_schema`, `run_sql` | Inspect tables and run arbitrary read-only SQL |
| `get_day_summary`, `get_metric_trend`, `get_recent_activities`, `get_activity_detail` | Fast helpers for common questions, no SQL needed |
| `chart` | Returns a Plotly spec that the UI re-runs and renders |
| `weather` | Stored forecast and per-activity observed weather |
| `memory` | Long-term notes about the user (never facts that the DB already holds) |
| `training` | `get` returns the season with the resolved current block and week plus planned-vs-actual coverage. `apply` performs one atomic edit (or `undo`). |

The agent never writes health data. Edits to the plan, memory and session go
through the app's own stores. Tool outputs are trimmed before they are saved to
history to keep context small.

## Derived metrics

`derived.py` computes these metrics after each sync and stores them in
`derived_metrics`:

- **ACWR** (`workload.py`): daily training load from
  `activity_summaries.training_load`. The ratio is acute load (EMA₇) over
  chronic load (EMA₂₈), banded as Detraining, Sweet Spot, Elevated or Danger.
  Rest days count as zero, so a taper shows up as a falling ratio.
- **Running ACWR** (`run_workload.py`): the same EMAs built from running
  distance only.
- **Cadence drift** (`run_cadence_drift`): a cadence z-score against your own
  cadence-vs-pace fit over the trailing 90 days. A negative score means
  overstriding or worse mechanics.

## Data model

Every data table is keyed by `(user_id, …)` and has `FORCE ROW LEVEL SECURITY`.
`users` is the only exception.

| Table | Holds |
| --- | --- |
| `users` | Email, bcrypt hash, encrypted Garmin credentials and tokens, sync status and backoff, per-account settings |
| `metrics` | Raw daily Garmin JSON by `(data_type, calendar_date)`. This is the source of truth. |
| `activities` | Raw activity summaries plus detail, weather and splits payloads |
| `sync_state` | Per-account sync bookkeeping |
| `daily_metrics` | One wide parsed row per day. Columns are created on demand. |
| `activity_summaries` | Parsed activities: duration, HR, zones, power, cadence, elevation, weather |
| `activity_detail_series` | Per-tick HR, cadence, power, speed, elevation and GPS |
| `activity_splits` | Per-lap splits |
| `hr_zones`, `power_zones` | Per-sport zone ranges (and FTP) from the device profile |
| `race_predictions` | Current 5k, 10k, half and full predictions |
| `user_profile`, `gear`, `devices` | Profile snapshots, current gear with totals, current devices |
| `derived_metrics` | `(calendar_date, metric)` rows for ACWR, run ACWR and cadence drift. Replaced each sync. |
| `weather_forecast` | Stored daily Open-Meteo forecast, refreshed each sync |
| `training_goal` | Goals (one per event) with sport, start date and target date |
| `training_block` | Periodized phases of a goal, with name, focus, dates and baseline weekly km |
| `training_week` | Per-week targets (distance, duration, deload flag) keyed by Monday |
| `training_workout` | Dated workouts with a status (`planned`, `completed`, `partial` or `skipped`) and target pace, HR zone or power |
| `user_state` | Agent memory, chat history, tool trace and the season undo snapshot |

## Configuration

Server-wide settings live in `.env` (see [`.env.example`](.env.example)).
Per-account settings live in the `users` table and are edited from the UI.

| Variable | Meaning |
| --- | --- |
| `GARMIN_DB_URL`, `GARMIN_READONLY_DB_URL`, `GARMIN_ADMIN_DB_URL` | App (writer), agent (read-only) and superuser DSNs. The superuser DSN is used only for role bootstrap. |
| `GARMIN_ENC_KEY` | Base64 of a 32-byte AES-GCM key |
| `GARMIN_JWT_SECRET`, `GARMIN_JWT_TTL_HOURS` | JWT signing secret and token lifetime |
| `GARMIN_CRON_TOKEN` | Bearer token for `POST /cron/sync` |
| `GARMIN_SYNC_INTERVAL_MIN`, `GARMIN_SYNC_MAX_WORKERS`, `GARMIN_SYNC_TIMEOUT_MIN` | Background worker tuning |
| `GARMIN_FETCH_SLEEP_SEC` | Delay between Garmin API calls (`2` is a good value for backfills) |
| `GARMIN_AUTO_SYNC` | Default for the per-account auto-sync toggle (off unless set) |
| `GARMIN_ACTIVITY_FREEZE_DAYS` | Trailing days re-scanned for late activity uploads |
| `GARMIN_LOCAL_USER_ID` | Account used by the CLI tools |
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Server-wide LLM. `openai` covers any OpenAI-compatible endpoint, including Ollama; `gemini` is also supported. |
| `LLM_REASONING_EFFORT` | Default reasoning effort (`low`, `medium` or `high`), which each account can override |

To generate an encryption key:

```sh
uv run python -c "import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

## Security

- **RLS is the boundary.** The runtime roles are not superusers, and every data
  table enforces a `user_isolation` policy on `current_setting('app.user_id')`.
  As a side effect, a raw `psql` session sees zero rows until it runs
  `SELECT set_config('app.user_id', '<id>', false)`.
- **Secrets at rest.** Garmin credentials and OAuth tokens are AES-GCM
  encrypted. Tokens are re-encrypted after each refresh and never logged.
- **Defense in depth for the agent.** The read-only role is the real
  enforcement. The statement gate and the allowed-tables list are a second
  layer.
- **Rate limits.** Failed syncs back off 30 min × 2ⁿ, capped at 8 h. A detected
  ban jumps straight to 8 h.
- Run the server behind TLS for anything beyond localhost, and change the
  default database passwords in `docker-compose.yml`.

## Project layout

```
src/garmin_fetch/
  fetcher.py        Garmin Connect sync (+ weather forecast refresh)
  parser.py         raw JSON → typed tables
  datatypes.py      data-type registry
  db.py             schema, RLS, per-user Database
  derived.py        rebuilds derived_metrics
  workload.py       ACWR
  run_workload.py   running ACWR + cadence drift
  ask.py            the read-only agent and its tools
  ask_web.py        legacy Gradio UI
  trace.py          per-turn tool-call tracing
  config.py         .env loading
  server/
    app.py          FastAPI routes
    auth.py         users, bcrypt, JWT, Garmin-verified signup + MFA
    crypto.py       AES-GCM
    sync_worker.py  worker pool, APScheduler cron, backoff
    geocode.py      home city → lat/lon
    setup_db.py     role bootstrap
    state/          agent memory/session/trace + training season store
    static/         index.html (the whole frontend)
docker/initdb/      role bootstrap for a fresh Postgres volume
```

## Tech stack

Python 3.14 · uv · PostgreSQL 17 · FastAPI · APScheduler · Pydantic AI · Plotly ·
Open-Meteo · garminconnect · PyJWT · bcrypt · cryptography
