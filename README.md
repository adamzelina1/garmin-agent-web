# Garmin Health-Data Agent

Fetch your Garmin Connect data into a local SQLite database, then ask a read-only
AI agent questions about it — sleep, resting HR, HRV, training load, activity
zones — and get answers with charts and weather context.

A complete, production-shaped personal data pipeline:

```
Garmin Connect ──▶ fetcher ──▶ SQLite (raw JSON, source of truth) ──▶ parser ──▶ typed tables
                                                                        │
                                          garmin-ask / garmin-ask-web ◀──┘
```

## What it does

- **Two-step ETL.** A fetcher syncs daily Garmin data into SQLite storing the
  *raw JSON payloads* as the source of truth; a separate parser projects them
  into typed, wide tables (`daily_metrics`, `activity_summaries`,
  `activity_detail_series`, `hr_zones`). Raw data is never thrown away — you can
  re-parse anything at any time.
- **Read-only AI agent.** `garmin-ask` is a Pydantic AI agent that answers
  natural-language questions about your data. It can *never* modify the
  database — safety is enforced by construction (`PRAGMA query_only`, a SQLite
  authorizer denying write opcodes, and a statement gate that only allows
  SELECT/WITH/EXPLAIN/PRAGMA).
- **Tool-calling agent.** The model inspects the schema, runs read-only SQL,
  validates and embeds Plotly chart specs, and consults a stateless weather tool
  (Open-Meteo, historical + short forecast) to contextualise the numbers. An
  optional long-term memory profile persists stable facts about you across
  sessions.
- **Browser chat UI.** `garmin-ask-web` is a Gradio front-end over the same
  agent, with auto-persisting sessions and chart rendering.
- **Token-efficient by design.** The schema is baked into the system prompt so
  the model writes correct SQL instead of guessing column names, tool errors are
  deduplicated, and prompt caching is surfaced per turn via `garmin-trace`.
- **Fully tested.** 136 tests run offline (fake Garmin client + a scripted
  test model) — no API keys or network needed.

## Data model

| Table | Purpose |
| --- | --- |
| `metrics` | Raw Garmin JSON payloads, keyed by `(data_type, calendar_date)` — the single source of truth |
| `daily_metrics` | Parsed daily projection, one wide row per date; columns auto-created on demand |
| `activities` | Raw activity summaries + intra-activity detail payloads, keyed by `activityId` |
| `activity_summaries` | Parsed activity projection (durations, HR, zones, elevation, power, cadence, body-battery drain, estimated sweat, PR flag, observed weather) |
| `activity_detail_series` | Parsed intra-activity time series (HR / cadence / power / speed / elevation / GPS per tick) |
| `hr_zones` | Derived per-sport heart-rate zone ranges from the device profile |
| `race_predictions` | Current race-prediction snapshot (5k/10k/half/full finish times) |
| `user_profile` | Raw profile snapshots (HR-zone config, race predictions) |

Data types supported: heart rate, sleep, HRV, stress, respiration, SpO2,
steps, body battery, VO2max / fitness age, intensity minutes, floors,
training status, lactate threshold / FTP, race predictions, plus full activity
summaries (with observed weather) and detail series.

## Agent tools

- `list_tables`, `table_schema`, `date_range`, `run_sql` — read-only database
  access
- `chart` — validates a model-authored Plotly spec and returns it for the UI to
  render (the raw data never travels through the model)
- `weather` — Open-Meteo historical + today/tomorrow forecast (degC, mm, km/h),
  stateless, home location from `GARMIN_HOME_LAT`/`GARMIN_HOME_LON`
- `get_memory` / `remember_memory` / `forget_memory` — optional long-term memory
  profile
- Interactive commands: `/clear` (fresh session, no context) and `/new`
  (compact the conversation into a summary and continue from it)

## Setup

Requirements: Python 3.14 and [`uv`](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/adamzelina1/garmin_agent.git
cd garmin_agent
uv sync

# configure your credentials (see .env.example)
cp .env.example .env
# edit .env: Garmin login + LLM API key (cloud) or LLM_BASE_URL (local Ollama)
```

## Usage

```sh
uv run garmin-fetch                        # incremental sync of all data types + activities
uv run garmin-fetch --type heart_rate      # sync a subset (repeatable)
uv run garmin-fetch --parse activities     # incremental reparse of stored activity data
uv run garmin-fetch --parse --full         # force a full re-parse of every stored row
uv run garmin-ask "avg sleep last week?"   # one-shot question
uv run garmin-ask                          # interactive session (/clear, /new)
uv run garmin-ask-web                      # browser chat UI (auto-persists sessions)
uv run garmin-trace                        # inspect the recorded agent tool-call trace
uv run pytest -q                           # run the test suite (offline)
```

First run may prompt for a Garmin MFA code; tokens are cached in
`~/.garminconnect`.

### Configuration

All settings live in `.env` (see [`.env.example`](.env.example)):

- `GARMIN_EMAIL` / `GARMIN_PASSWORD` — Garmin Connect credentials (required)
- `GARMIN_START_DATE` — backfill window (e.g. `2025-01-01`)
- `LLM_MODEL` / `LLM_API_KEY` (or `OPENAI_API_KEY`) / `LLM_BASE_URL` — model
  provider; cloud by default, or a local Ollama endpoint
- `GARMIN_HOME_LAT` / `GARMIN_HOME_LON` — home location for the weather tool
- `GARMIN_DB_PATH`, `GARMIN_MEMORY_FILE`, `GARMIN_WEB_SESSION_FILE`,
  `GARMIN_TRACE_FILE` — runtime file locations

## Security

- Credentials and personal data are gitignored; only code ships to the repo.
- The agent is read-only by construction: connections open per call with
  `PRAGMA query_only`, an authorizer denies all write opcodes, and only
  single-statement SELECT/WITH/EXPLAIN/PRAGMA queries are allowed.
- The weather tool is stateless — one HTTP request per call, nothing stored.

## Tech stack

Python 3.14 · `uv` · SQLite · Pydantic AI · Gradio · Plotly · Open-Meteo ·
garminconnect · pytest

## Project layout

```
src/garmin_fetch/
  fetcher.py    # Garmin Connect sync (raw data -> SQLite)
  parser.py     # raw JSON -> typed table projections
  datatypes.py  # data-type registry
  db.py         # SQLite schema + storage
  ask.py        # read-only AI agent (CLI)
  ask_web.py    # browser chat UI (Gradio)
  trace.py      # per-turn tool-call tracing
tests/          # offline test suite (fake client, scripted model)
```
