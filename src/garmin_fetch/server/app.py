"""FastAPI server: auth (JWT), per-user sync triggers, the read-only ask agent.

Serves a small same-origin JS frontend (``/static/index.html``) and a JSON
API:

- ``POST /auth/register``   — create account + bind Garmin (single-step login)
- ``POST /auth/login``      — email+password -> JWT
- ``GET  /auth/me``         — current user (JWT)
- ``POST /sync``            — enqueue the caller's own sync (JWT)
- ``GET  /sync/status``     — sync status for the caller (JWT)
- ``POST /cron/sync``       — daemon-only: enqueue every active user
- ``GET  /readiness``       — custom training-readiness score (JWT)
- ``GET  /acwr``            — acute-to-chronic workload ratio (JWT)
- ``GET  /training-plan``   — the caller's planned workouts (JWT, optional range)
- ``POST /training-plan``   — add a workout (JWT)
- ``PUT  /training-plan/{id}`` — update a workout (JWT)
- ``DELETE /training-plan/{id}`` — delete a workout (JWT)
- ``POST /ask``             — run the read-only agent (JWT, per-user rows)
- ``GET  /ask/history``     — the stored conversation for the caller (JWT)
- ``POST /ask/clear``       — drop the stored conversation, fresh session (JWT)
- ``POST /ask/chart``       — render a chart spec to Plotly JSON (JWT)

Every data-touching request runs as the read-only PG role with
``app.user_id`` set to the authenticated account; Row-Level Security scopes
all rows even if a future code path drops the agent's statement gate. Agent
state (long-term memory, conversation history, tool-call trace) is persisted
per user in the ``user_state`` table, so nothing user-facing lives on disk.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..ask import (
    ReadOnlyDB,
    _build_agent,
    _build_chart_figure,
    _record_turn,
)
from ..ask_web import (
    _extract_charts,
    _history_to_messages,
    _messages_to_history,
    _replace_plan_dumps,
)
from ..config import load_config
from ..db import ensure_schema
from .auth import AuthError, AuthService, UserStore
from .setup_db import ensure_roles
from .state import PgMemory, TrainingPlan, TrainingPlanStore, UserState
from .sync_worker import SyncManager

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


# -- Request bodies ----------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    garmin_email: str
    garmin_password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AskRequest(BaseModel):
    question: str
    history: list[dict[str, Any]] = []


class ChartRequest(BaseModel):
    spec: dict[str, Any]


class ConfigRequest(BaseModel):
    api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    home_lat: str = ""
    home_lon: str = ""
    excluded_data_types: list[str] = []
    clear_api_key: bool = False


class TrainingPlanWorkout(BaseModel):
    planned_date: str
    activity_type: str
    title: str | None = None
    description: str | None = None
    duration_min: int | None = None
    distance_km: float | None = None
    intensity: str | None = None
    completed: bool = False


# -- App state ---------------------------------------------------------------

def _per_user_cfg(cfg: dict[str, Any], user_id: int) -> dict[str, Any]:
    """Config copy for one user's agent run.

    Agent state (memory, session, trace) lives in the ``user_state`` table, so
    there is nothing to write under a per-user directory anymore.
    """
    return dict(cfg)


def _readonly(cfg: dict[str, Any], user_id: int) -> ReadOnlyDB:
    url = cfg.get("readonly_db_url") or cfg["db_url"]
    if not cfg.get("readonly_db_url"):
        logger.warning(
            "GARMIN_READONLY_DB_URL not set — agent connects as the writer role"
        )
    return ReadOnlyDB.from_url(url, user_id=user_id)


def _user_agent_cfg(cfg: dict[str, Any], user: dict[str, Any], auth: Any) -> dict[str, Any]:
    """Strict per-user config for the LLM agent (no server .env fallback).

    Every LLM/weather/exclusion setting comes from the user's own row; the
    API key is decrypted here, so it is never stored or logged in the clear.
    """
    user_cfg = _per_user_cfg(cfg, user["id"])
    user_cfg["llm_api_key"] = (
        auth.encryptor.decrypt(user["llm_api_key_enc"])
        if user.get("llm_api_key_enc")
        else ""
    )
    user_cfg["llm_base_url"] = user.get("llm_base_url") or ""
    user_cfg["llm_model"] = user.get("llm_model") or ""
    user_cfg["weather_home_lat"] = user.get("home_lat") or ""
    user_cfg["weather_home_lon"] = user.get("home_lon") or ""
    user_cfg["excluded_data_types"] = user.get("excluded_data_types") or ""
    return user_cfg


def create_app(cfg: dict[str, Any] | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ensure_roles(
            cfg.get("admin_db_url", ""), cfg["db_url"], cfg.get("readonly_db_url", "")
        )
        ensure_schema(cfg["db_url"])
        auth = AuthService(cfg)
        sync = SyncManager(cfg)
        state = UserState(cfg["db_url"])
        plan = TrainingPlanStore(cfg["db_url"])
        app.state.auth = auth
        app.state.sync = sync
        app.state.state = state
        app.state.plan = plan
        app.state.cfg = cfg
        sync.start()
        logger.info("garmin server started")
        try:
            yield
        finally:
            sync.shutdown()
            state.close()
            plan.close()
            auth.close()

    app = FastAPI(title="Garmin Agent", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # -- dependencies ---------------------------------------------------------

    def get_user(request: Request, authorization: str | None = Header(None)) -> dict:
        auth: AuthService = request.app.state.auth
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        try:
            return auth.current_user(token)
        except AuthError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

    # -- frontend -------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

    # -- auth -----------------------------------------------------------------

    @app.post("/auth/register")
    def register(body: RegisterRequest, request: Request) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        try:
            return auth.register(
                email=body.email,
                password=body.password,
                garmin_email=body.garmin_email,
                garmin_password=body.garmin_password,
            )
        except AuthError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

    @app.post("/auth/login")
    def login(body: LoginRequest, request: Request) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        try:
            return auth.login(email=body.email, password=body.password)
        except AuthError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

    @app.get("/auth/me")
    def me(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        return {
            "id": user["id"],
            "email": user["email"],
            "garmin_email": user["garmin_email"],
            "confirmed": user["confirmed"],
            "active": user["active"],
            "last_sync_at": user["last_sync_at"],
            "sync_error": user["sync_error"],
            "llm_configured": auth.user_llm_configured(user),
        }

    @app.get("/auth/config")
    def get_config(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        return auth.get_user_config(user["id"])

    @app.put("/auth/config")
    def put_config(
        body: ConfigRequest, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        auth.save_user_config(
            user["id"],
            api_key=body.api_key,
            llm_base_url=body.llm_base_url.strip(),
            llm_model=body.llm_model.strip(),
            home_lat=body.home_lat.strip(),
            home_lon=body.home_lon.strip(),
            excluded_data_types=[
                t.strip().lower() for t in body.excluded_data_types if t.strip()
            ],
            clear_api_key=body.clear_api_key,
        )
        return {"status": "ok"}

    # -- sync -----------------------------------------------------------------

    @app.post("/sync")
    def trigger_sync(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        sync: SyncManager = request.app.state.sync
        return {
            "queued": sync.enqueue(user["id"]),
            "running": sync.is_running(user["id"]),
        }

    @app.post("/sync/full")
    def trigger_sync_full(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """Sync + full re-parse: fetch new data, then rebuild the typed tables
        (daily_metrics, activity summaries, detail series) from raw, ignoring
        the incremental parse markers — for recovering bugged rows.
        """
        sync: SyncManager = request.app.state.sync
        return {
            "queued": sync.enqueue(user["id"], force_reparse=True),
            "running": sync.is_running(user["id"]),
        }

    @app.get("/sync/status")
    def sync_status(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        store: UserStore = request.app.state.auth.store
        row = store.get(user["id"]) or {}
        return {
            "running": request.app.state.sync.is_running(user["id"]),
            "last_sync_at": row.get("last_sync_at"),
            "sync_error": row.get("sync_error"),
            "rate_limit_until": row.get("rate_limit_until"),
        }

    @app.post("/cron/sync")
    def cron_sync(
        request: Request, authorization: str | None = Header(None)
    ) -> dict[str, Any]:
        cfg = request.app.state.cfg
        expected = cfg.get("cron_token", "")
        if not expected or authorization != f"Bearer {expected}":
            raise HTTPException(403, "invalid cron token")
        enqueued = request.app.state.sync.cron_sync()
        return {"enqueued": enqueued}

    # -- readiness ------------------------------------------------------------

    @app.get("/readiness")
    def readiness(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """The user's custom training-readiness score.

        Read from the stored ``derived_metrics`` table (metric 'readiness'),
        which the sync pipeline recomputes once per sync after the night's
        sleep/HRV/RHR have landed. Returns the per-day series plus the most
        recent scored day, and the auto-fit scale used for the current series.
        """
        from ..db import PostgresBackend
        from ..readiness import effective_cutoffs, read_series

        backend = PostgresBackend(request.app.state.cfg["db_url"], user_id=user["id"])
        conn = backend.connect()
        try:
            days = read_series(conn)
        finally:
            conn.close()
        scored = [d for d in days if d.get("score") is not None]
        return {
            "today": scored[-1] if scored else None,
            "days": days,
            "meta": {"scale": effective_cutoffs(days)},
        }

    @app.get("/acwr")
    def acwr(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """The user's Acute-to-Chronic Workload Ratio.

        Read from the stored ``derived_metrics`` table (metric 'acwr'),
        recomputed once per sync from the daily training load in
        ``activity_summaries``. Returns the per-day series plus the most
        recent day with a ratio.
        """
        from ..db import PostgresBackend
        from ..workload import read_series

        backend = PostgresBackend(request.app.state.cfg["db_url"], user_id=user["id"])
        conn = backend.connect()
        try:
            days = read_series(conn)
        finally:
            conn.close()
        scored = [d for d in days if d.get("acwr") is not None]
        return {"today": scored[-1] if scored else None, "days": days}

    # -- training plan --------------------------------------------------------

    @app.get("/training-plan")
    def training_plan_list(
        request: Request,
        user: dict = Depends(get_user),
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """The user's planned workouts, optionally in an inclusive date range."""
        from datetime import date as _date

        for label, value in (("from_date", from_date), ("to_date", to_date)):
            if value:
                try:
                    _date.fromisoformat(value)
                except ValueError as exc:
                    raise HTTPException(
                        400, f"{label} must be a YYYY-MM-DD date (or blank)"
                    ) from exc
        plan: TrainingPlanStore = request.app.state.plan
        return {"workouts": plan.list(user["id"], from_date, to_date)}

    @app.post("/training-plan")
    def training_plan_create(
        body: TrainingPlanWorkout, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        plan: TrainingPlanStore = request.app.state.plan
        try:
            return plan.create(user["id"], body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/training-plan/{workout_id}")
    def training_plan_update(
        workout_id: int,
        body: TrainingPlanWorkout,
        request: Request,
        user: dict = Depends(get_user),
    ) -> dict[str, Any]:
        plan: TrainingPlanStore = request.app.state.plan
        try:
            row = plan.update(user["id"], workout_id, body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if row is None:
            raise HTTPException(404, "workout not found")
        return row

    @app.delete("/training-plan/{workout_id}")
    def training_plan_delete(
        workout_id: int, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        plan: TrainingPlanStore = request.app.state.plan
        if not plan.delete(user["id"], workout_id):
            raise HTTPException(404, "workout not found")
        return {"status": "ok"}

    # -- ask ------------------------------------------------------------------

    @app.post("/ask")
    def ask(
        body: AskRequest, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        if not auth.user_llm_configured(user):
            raise HTTPException(
                503,
                "the LLM agent is not configured for your account: add an API key "
                "(or a local LLM base URL) in Settings → Config",
            )
        user_cfg = _user_agent_cfg(request.app.state.cfg, user, auth)
        db = _readonly(request.app.state.cfg, user["id"])
        state: UserState = request.app.state.state
        try:
            memory = PgMemory(state, user["id"])
            agent = _build_agent(
                user_cfg,
                db,
                memory=memory,
                plan=TrainingPlan(request.app.state.plan, user["id"]),
            )
            # The DB is the source of truth for the conversation: a caller that
            # sends no history resumes the stored session (full replay), so a
            # returning user continues exactly where they left off.
            if body.history:
                history = _history_to_messages(body.history)
            else:
                history = state.get_session_messages(user["id"])
            result = agent.run_sync(body.question, message_history=history)

            def _trace_writer(record: dict) -> None:
                state.append_trace(user["id"], record)

            _record_turn(user_cfg, body.question, result, trace_writer=_trace_writer)
            answer = str(result.output)
            text, specs = _extract_charts(answer)
            text = _replace_plan_dumps(text)
            state.set_session_messages(user["id"], result.all_messages())
            return {"answer": text, "chart_specs": specs}
        except Exception as exc:  # noqa: BLE001 - a bad question must not crash the server
            logger.exception("ask failed for user %s", user["id"])
            raise HTTPException(500, f"ask failed: {exc}") from exc
        finally:
            db.close()

    @app.get("/ask/history")
    def ask_history(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """The stored conversation in openai-style {role, content} pairs, for
        rendering the resumed session in the UI on login/reload. Stored
        assistant messages keep the raw markdown (including <chart> blocks), so
        the same post-processing as the live ask path is applied here: charts
        are extracted to chart_specs (so they re-render) and plan dumps are
        replaced with the <plan_table /> marker."""
        state: UserState = request.app.state.state
        messages = state.get_session_messages(user["id"]) or []
        history = _messages_to_history(messages)
        for item in history:
            if item.get("role") in ("assistant", "bot"):
                text, specs = _extract_charts(item["content"])
                item["content"] = _replace_plan_dumps(text)
                item["chart_specs"] = specs
        return {"history": history}

    @app.post("/ask/clear")
    def ask_clear(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """Start a fresh session: drop the stored conversation so the next ask
        resumes with no prior context (long-term memory is kept)."""
        state: UserState = request.app.state.state
        state.clear_session(user["id"])
        return {"status": "ok"}

    @app.post("/ask/chart")
    def ask_chart(
        body: ChartRequest, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        cfg = request.app.state.cfg
        spec = body.spec
        if not isinstance(spec.get("sql"), str):
            raise HTTPException(400, "chart spec needs a string 'sql' key")
        db = _readonly(cfg, user["id"])
        try:
            result = db.run_sql(spec["sql"])
            figure = _build_chart_figure(spec, result)
            return json.loads(figure.to_json())
        except Exception as exc:  # noqa: BLE001 - invalid spec -> client error
            raise HTTPException(400, f"invalid chart spec: {exc}") from exc
        finally:
            db.close()

    return app


app = create_app()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="garmin-server", description="Run the multi-user Garmin web server."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="bind host (default 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--reload", action="store_true", help="auto-reload on code changes"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug logging for the fetcher"
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import uvicorn

    uvicorn.run("garmin_fetch.server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
