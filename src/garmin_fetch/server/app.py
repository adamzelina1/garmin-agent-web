"""FastAPI server: auth (JWT), per-user sync triggers, the read-only ask agent.

Serves a small same-origin JS frontend (``/static/index.html``) and a JSON
API:

- ``POST /auth/register``   — create account + bind Garmin (single-step login)
- ``POST /auth/register/mfa`` — submit Garmin verification code, finish bound
- ``POST /auth/login``      — email+password -> JWT
- ``GET  /auth/me``         — current user (JWT)
- ``POST /sync``            — enqueue the caller's own sync (JWT)
- ``GET  /sync/status``     — sync status for the caller (JWT)
- ``POST /cron/sync``       — daemon-only: enqueue every active user
- ``GET  /acwr``            — acute-to-chronic workload ratio (JWT)
- ``GET  /training/workouts``   — the caller's planned workouts (JWT, optional range)
- ``POST /training/workouts``   — add a workout (JWT)
- ``PATCH /training/workouts/{id}`` — partial update of a workout (JWT)
- ``DELETE /training/workouts/{id}`` — delete a workout (JWT, undoable)
- ``POST /training/undo``    — restore the whole season pre-destructive-edit (JWT)
- ``GET  /training/anchor``    — goals + blocks + weeks + resolved week (JWT)
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
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..ask import (
    ReadOnlyDB,
    _auto_compact,
    _build_agent,
    _build_chart_figure,
    _final_answer,
    _messages_token_estimate,
    _prune_session_messages,
    _record_turn,
    _refresh_resumed_prompt,
)
from ..ask_web import (
    _extract_charts,
    _history_to_messages,
    _messages_to_history,
    _replace_workout_dumps,
)
from ..config import load_config
from ..db import ensure_schema
from .auth import AuthError, AuthService, UserStore
from .setup_db import ensure_roles
from .state import (
    PgMemory, PgPrinciples, TrainingAnchorStore, TrainingWorkoutStore,
    TrainingSeason, TrainingStore, UserState, _block_week_count,
    _resolve_block_weeks,
)
from .sync_worker import SyncManager

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


# -- Request bodies ----------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    garmin_email: str
    garmin_password: str


class RegisterMFARequest(BaseModel):
    challenge: str
    code: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AskRequest(BaseModel):
    question: str
    history: list[dict[str, Any]] = []
    stream: bool = False


class ChartRequest(BaseModel):
    spec: dict[str, Any]


class ConfigRequest(BaseModel):
    home_lat: str | None = None
    home_lon: str | None = None
    home_city: str | None = None
    home_country: str | None = None
    excluded_data_types: list[str] | None = None
    auto_sync: bool | None = None
    sync_start_date: str | None = None
    reasoning_effort: str | None = None
    training_principles: str | None = None
    memory: dict[str, str] | None = None


class WorkoutStep(BaseModel):
    """One interval in a structured workout.

    ``kind`` is the step role (warmup/steady/work/recovery/cooldown/rest) and a
    positive duration is required — use the friendly ``duration`` string
    (``"15m"``, ``"90s"``, ``"1:30"``) or a plain number of minutes.
    ``repeat`` > 1 expands the step, and the optional targets mirror the flat
    workout targets.
    """

    kind: str = "steady"
    label: str | None = None
    duration: str | float | None = None
    repeat: int | None = None
    intensity: str | None = None
    target_pace_min_km: float | None = None
    target_hr_zone: str | None = None
    target_power_w: int | None = None
    notes: str | None = None


class TrainingWorkout(BaseModel):
    """A planned workout (full create/replace body)."""

    planned_date: str
    activity_type: str
    title: str | None = None
    description: str | None = None
    duration_min: int | None = None
    distance_km: float | None = None
    intensity: str | None = None
    target_pace_min_km: float | None = None
    target_hr_zone: str | None = None
    target_power_w: int | None = None
    steps: list[WorkoutStep] | None = None
    status: str | None = None
    goal_id: int | None = None


class TrainingWorkoutPatch(BaseModel):
    """A partial workout edit — only the supplied fields change (PATCH)."""

    planned_date: str | None = None
    activity_type: str | None = None
    title: str | None = None
    description: str | None = None
    duration_min: int | None = None
    distance_km: float | None = None
    intensity: str | None = None
    target_pace_min_km: float | None = None
    target_hr_zone: str | None = None
    target_power_w: int | None = None
    steps: list[WorkoutStep] | None = None
    status: str | None = None
    goal_id: int | None = None


class TrainingWeek(BaseModel):
    """One week target of a block.

    The ``weeks`` list is the block's complete vector — entry 1 is the calendar
    week containing the block start — so ``week_start`` is server-derived and
    only meaningful when a week row is read back.
    """

    week_start: str | None = None
    distance_km: float | None = None
    duration_min: int | None = None
    is_deload: bool | None = None


class TrainingBlock(BaseModel):
    id: int | None = None
    name: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    focus: str | None = None
    target_weekly_km: float | None = None
    weeks: list[TrainingWeek] | None = None


class TrainingAnchor(BaseModel):
    title: str | None = None
    sport: str | None = None
    start_date: str | None = None
    target_date: str | None = None
    target_distance_km: float | None = None
    target_time: str | None = None
    blocks: list[TrainingBlock] | None = None
    delete_blocks: list[int] | None = None


# -- App state ---------------------------------------------------------------

def _per_user_cfg(cfg: dict[str, Any], user_id: int) -> dict[str, Any]:
    """Config copy for one user's agent run.

    Agent state (memory, session, trace) lives in the ``user_state`` table, so
    there is nothing to write under a per-user directory anymore.
    """
    return dict(cfg)


def _readonly(
    cfg: dict[str, Any], user_id: int, excluded_types: str = ""
) -> ReadOnlyDB:
    url = cfg.get("readonly_db_url") or cfg["db_url"]
    if not cfg.get("readonly_db_url"):
        logger.warning(
            "GARMIN_READONLY_DB_URL not set — agent connects as the writer role"
        )
    # Hide columns the account can't have data for (disabled data types) so the
    # agent's schema introspection stays lean and never costs tokens on metrics
    # it will only ever see as NULL.
    return ReadOnlyDB.from_url(url, user_id=user_id, excluded_types=excluded_types)


def _user_agent_cfg(cfg: dict[str, Any], user: dict[str, Any], auth: Any) -> dict[str, Any]:
    """Per-user config for the LLM agent.

    The LLM provider (API key, base URL, model) is server-wide, sourced only
    from the server ``.env`` — it is never per-account. Only the weather
    location and the excluded data types remain user-specific overrides.
    """
    user_cfg = _per_user_cfg(cfg, user["id"])
    user_cfg["llm_api_key"] = cfg.get("llm_api_key") or ""
    user_cfg["llm_base_url"] = cfg.get("llm_base_url") or ""
    user_cfg["llm_model"] = cfg.get("llm_model") or ""
    user_cfg["llm_provider"] = cfg.get("llm_provider") or ""
    user_cfg["llm_reasoning_effort"] = (
        user.get("reasoning_effort") or cfg.get("llm_reasoning_effort") or ""
    )
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
        training = TrainingStore(cfg["db_url"])
        app.state.auth = auth
        app.state.sync = sync
        app.state.state = state
        app.state.training = training
        app.state.workouts = training.workouts
        app.state.anchor = training.anchor
        app.state.cfg = cfg
        app.state.chart_cache = {}
        sync.start()
        logger.info("garmin server started")
        try:
            yield
        finally:
            sync.shutdown()
            state.close()
            training.close()
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

    @app.post("/auth/register/mfa")
    def register_mfa(body: RegisterMFARequest, request: Request) -> dict[str, Any]:
        """Finish a signup that hit Garmin two-step verification by submitting
        the code the user was sent. ``challenge`` is from the ``mfa_required``
        response of ``POST /auth/register``."""
        auth: AuthService = request.app.state.auth
        try:
            return auth.confirm_mfa(challenge=body.challenge, code=body.code)
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
            "auto_sync": bool(user.get("auto_sync")),
            "llm_configured": auth.user_llm_configured(user),
        }

    @app.get("/auth/config")
    def get_config(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        data = auth.get_user_config(user["id"])
        state: UserState = request.app.state.state
        data["training_principles"] = PgPrinciples(state, user["id"]).get()
        data["memory"] = PgMemory(state, user["id"]).get()
        return data

    @app.put("/auth/config")
    def put_config(
        body: ConfigRequest, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        auth: AuthService = request.app.state.auth
        # Only write fields the client actually sent. Partial updates (e.g. the
        # auto-sync toggle, the sync start date) must never reset unrelated
        # settings to empty via Pydantic defaults.
        kwargs: dict[str, Any] = {}
        if body.home_lat is not None:
            kwargs["home_lat"] = body.home_lat.strip()
        if body.home_lon is not None:
            kwargs["home_lon"] = body.home_lon.strip()
        if body.home_city is not None:
            kwargs["home_city"] = body.home_city.strip()
        if body.home_country is not None:
            kwargs["home_country"] = body.home_country.strip()
        if body.excluded_data_types is not None:
            kwargs["excluded_data_types"] = [
                t.strip().lower() for t in body.excluded_data_types if t.strip()
            ]
        if body.auto_sync is not None:
            kwargs["auto_sync"] = body.auto_sync
        if body.sync_start_date is not None:
            kwargs["sync_start_date"] = body.sync_start_date
        if body.reasoning_effort is not None:
            kwargs["reasoning_effort"] = body.reasoning_effort
        # Training principles live in ``user_state`` (not the users table), so
        # they are written separately. Validate them before the config save so a
        # rejected value never leaves the request half-applied.
        if body.training_principles is not None:
            state: UserState = request.app.state.state
            try:
                PgPrinciples(state, user["id"]).set(body.training_principles)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        # Long-term memory is edited here as a whole-profile replace (the same
        # store the agent writes through), with the fact cap enforced.
        if body.memory is not None:
            state: UserState = request.app.state.state
            try:
                PgMemory(state, user["id"]).replace(body.memory)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        auth.save_user_config(user["id"], **kwargs)
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

    @app.get("/run-acwr")
    def run_acwr(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """The user's running-isolated Acute-to-Chronic Workload Ratio.

        Read from the stored ``derived_metrics`` table (metric 'run_acwr'),
        recomputed once per sync from the daily running distance in
        ``activity_summaries`` — foot-strike volume only, so cycling/swimming
        cannot mask low running volume. Returns the per-day series plus the most
        recent day with a ratio.
        """
        from ..db import PostgresBackend
        from ..run_workload import read_series

        backend = PostgresBackend(request.app.state.cfg["db_url"], user_id=user["id"])
        conn = backend.connect()
        try:
            days = read_series(conn)
        finally:
            conn.close()
        scored = [d for d in days if d.get("run_acwr") is not None]
        return {"today": scored[-1] if scored else None, "days": days}

    # -- training workouts ----------------------------------------------------

    @app.get("/training/workouts")
    def training_workout_list(
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
        workouts: TrainingWorkoutStore = request.app.state.workouts
        return {"workouts": workouts.list(user["id"], from_date, to_date)}

    @app.post("/training/workouts")
    def training_workout_create(
        body: TrainingWorkout, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        workouts: TrainingWorkoutStore = request.app.state.workouts
        try:
            return workouts.create(user["id"], body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/undo")
    def training_undo(
        request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        """Restore the whole season from the last destructive-edit snapshot."""
        training: TrainingStore = request.app.state.training
        try:
            restored = training.undo(user["id"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"restored": restored}

    @app.patch("/training/workouts/{workout_id}")
    def training_workout_patch(
        workout_id: int,
        body: TrainingWorkoutPatch,
        request: Request,
        user: dict = Depends(get_user),
    ) -> dict[str, Any]:
        """Partial workout update — only the fields sent change.

        This is the edit path the plan tab uses (including drag-and-drop), so
        moving a workout never clears its title, targets or goal/block link.
        """
        workouts: TrainingWorkoutStore = request.app.state.workouts
        try:
            row = workouts.update(
                user["id"], workout_id, body.model_dump(exclude_unset=True),
                partial=True,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if row is None:
            raise HTTPException(404, "workout not found")
        return row

    @app.delete("/training/workouts/{workout_id}")
    def training_workout_delete(
        workout_id: int, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        workouts: TrainingWorkoutStore = request.app.state.workouts
        training: TrainingStore = request.app.state.training
        training.snapshot(user["id"])
        if not workouts.delete(user["id"], workout_id):
            raise HTTPException(404, "workout not found")
        return {"status": "ok"}

    @app.get("/activities")
    def activities_get(
        request: Request,
        user: dict = Depends(get_user),
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """The user's synced activities in an inclusive date range.

        Lightweight read for the Training Plan tab's weekly progress (actual
        mileage vs the week target). Returns activity_id, start_date,
        activity_type, distance_km and duration_hours.
        """
        from datetime import date as _date

        from ..db import PostgresBackend

        for label, value in (("from_date", from_date), ("to_date", to_date)):
            if value:
                try:
                    _date.fromisoformat(value)
                except ValueError as exc:
                    raise HTTPException(
                        400, f"{label} must be a YYYY-MM-DD date (or blank)"
                    ) from exc
        if not from_date or not to_date:
            raise HTTPException(400, "pass both from_date and to_date")

        backend = PostgresBackend(request.app.state.cfg["db_url"], user_id=user["id"])
        conn = backend.connect()
        try:
            rows = conn.execute(
                "SELECT activity_id, start_date, activity_type, distance_km, "
                "duration_hours FROM activity_summaries "
                "WHERE user_id = %s AND start_date >= %s AND start_date <= %s "
                "ORDER BY start_date",
                (user["id"], from_date, to_date),
            ).fetchall()
        finally:
            conn.close()
        return {"activities": [dict(r) for r in rows]}

    # -- long-term anchor -----------------------------------------------------

    def _goal_dump(
        store: TrainingAnchorStore, user_id: int, gid: int,
        week_start: str | None = None,
    ) -> dict[str, Any]:
        """One goal with its blocks (+ weeks), resolved block and weekly target.

        Thin wrapper over ``TrainingAnchorStore.resolve_goal`` (the single
        composition shared with the agent tools), resolved for ``week_start``
        (default today) so the plan tab never re-derives block/week selection.
        """
        resolved = store.resolve_goal(user_id, gid, week_start)
        if resolved is None:
            raise HTTPException(404, "goal not found")
        return resolved

    def _block_dump(store: TrainingAnchorStore, user_id: int, block_id: int) -> dict[str, Any]:
        """Nest one block with its weeks (404 when it does not exist)."""
        row = store.get_block(user_id, block_id)
        if row is None:
            raise HTTPException(404, "block not found")
        row = dict(row)
        raw_weeks = store.list_weeks(user_id, block_id)
        row["weeks"] = _resolve_block_weeks(row, raw_weeks)
        row["week_count"] = _block_week_count(row)
        return row

    @app.get("/training/anchor")
    def training_anchor_get(
        request: Request, user: dict = Depends(get_user),
        week_start: str | None = None,
    ) -> dict[str, Any]:
        """The user's long-term anchors (goal + blocks + weeks + current block).

        ``week_start`` (YYYY-MM-DD, default today) selects the week whose week
        target each goal reports as ``weekly_target``.
        """
        if week_start:
            from datetime import date as _date

            try:
                _date.fromisoformat(week_start)
            except ValueError as exc:
                raise HTTPException(
                    400, "week_start must be a YYYY-MM-DD date"
                ) from exc
        anchor: TrainingAnchorStore = request.app.state.anchor
        goals = [
            _goal_dump(anchor, user["id"], g["id"], week_start)
            for g in anchor.list_goals(user["id"])
        ]
        return {"goals": goals}

    @app.post("/training/anchor")
    def training_anchor_create(
        body: TrainingAnchor, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        """Create a new long-term goal (appends it; never wipes other goals)."""
        anchor: TrainingAnchorStore = request.app.state.anchor
        try:
            saved = anchor.create_goal(user["id"], body.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _goal_dump(anchor, user["id"], saved["id"])

    @app.get("/training/anchor/{goal_id}")
    def training_anchor_get_one(
        goal_id: int, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        anchor: TrainingAnchorStore = request.app.state.anchor
        if anchor.get_goal(user["id"], goal_id) is None:
            raise HTTPException(404, "goal not found")
        return _goal_dump(anchor, user["id"], goal_id)

    @app.patch("/training/anchor/{goal_id}")
    def training_anchor_update(
        goal_id: int,
        body: TrainingAnchor,
        request: Request,
        user: dict = Depends(get_user),
    ) -> dict[str, Any]:
        """Partial update of one goal (only the fields sent change)."""
        anchor: TrainingAnchorStore = request.app.state.anchor
        training: TrainingStore = request.app.state.training
        data = body.model_dump(exclude_unset=True)
        if "delete_blocks" in data:
            # Destructive block edits are restorable via the season undo.
            training.snapshot(user["id"])
        try:
            saved = anchor.update_goal(user["id"], goal_id, data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if saved is None:
            raise HTTPException(404, "goal not found")
        return _goal_dump(anchor, user["id"], goal_id)

    @app.delete("/training/anchor/{goal_id}")
    def training_anchor_delete(
        goal_id: int, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        anchor: TrainingAnchorStore = request.app.state.anchor
        training: TrainingStore = request.app.state.training
        training.snapshot(user["id"])
        if not anchor.delete_goal(user["id"], goal_id):
            raise HTTPException(404, "goal not found")
        return {"status": "ok"}

    @app.get("/training/anchor/block/{block_id}/weeks")
    def training_week_list(
        block_id: int, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        anchor: TrainingAnchorStore = request.app.state.anchor
        if anchor.get_block(user["id"], block_id) is None:
            raise HTTPException(404, "block not found")
        return {"weeks": anchor.list_weeks(user["id"], block_id)}

    @app.put("/training/anchor/block/{block_id}/weeks")
    def training_week_replace(
        block_id: int,
        body: TrainingBlock,
        request: Request,
        user: dict = Depends(get_user),
    ) -> dict[str, Any]:
        """Replace a block's explicit week target list (one row per week)."""
        anchor: TrainingAnchorStore = request.app.state.anchor
        training: TrainingStore = request.app.state.training
        weeks = [w.model_dump(exclude_none=True) for w in (body.weeks or [])]
        training.snapshot(user["id"])
        try:
            result = anchor.replace_weeks(user["id"], block_id, weeks)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"weeks": result}

    @app.patch("/training/anchor/block/{block_id}")
    def training_block_update(
        block_id: int,
        body: TrainingBlock,
        request: Request,
        user: dict = Depends(get_user),
    ) -> dict[str, Any]:
        """Partial block update — adjust dates/weeks without rewriting the season."""
        anchor: TrainingAnchorStore = request.app.state.anchor
        try:
            row = anchor.update_block(
                user["id"], block_id, body.model_dump(exclude_unset=True)
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if row is None:
            raise HTTPException(404, "block not found")
        return _block_dump(anchor, user["id"], block_id)

    @app.delete("/training/anchor/block/{block_id}")
    def training_block_delete(
        block_id: int, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        anchor: TrainingAnchorStore = request.app.state.anchor
        training: TrainingStore = request.app.state.training
        training.snapshot(user["id"])
        if not anchor.delete_block(user["id"], block_id):
            raise HTTPException(404, "block not found")
        return {"status": "ok"}

    # -- weather ---------------------------------------------------------------

    @app.get("/weather")
    def weather_get(
        request: Request,
        user: dict = Depends(get_user),
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """The user's stored daily weather forecast (min/max degC, precip, wind).

        Populated by the sync worker from Open-Meteo on each sync, so this is
        read-only and never hits an external API on a page view. With no dates
        the window today..(today+15) is returned (whatever has been synced).
        """
        from datetime import date as _date, timedelta as _timedelta

        from ..db import PostgresBackend

        for label, value in (("from_date", from_date), ("to_date", to_date)):
            if value:
                try:
                    _date.fromisoformat(value)
                except ValueError as exc:
                    raise HTTPException(
                        400, f"{label} must be a YYYY-MM-DD date (or blank)"
                    ) from exc

        today = _date.today()
        if from_date is None and to_date is None:
            start, end = today, today + _timedelta(days=15)
        elif from_date is None or to_date is None:
            raise HTTPException(400, "pass both from_date and to_date, or neither")
        else:
            start, end = _date.fromisoformat(from_date), _date.fromisoformat(to_date)
        if end < start:
            raise HTTPException(400, "to_date must not be before from_date")

        backend = PostgresBackend(request.app.state.cfg["db_url"], user_id=user["id"])
        conn = backend.connect()
        try:
            rows = conn.execute(
                "SELECT calendar_date AS date, temp_max_c, temp_min_c, precip_mm, "
                "wind_max_kmh, condition_code, source "
                "FROM weather_forecast "
                "WHERE user_id = %s AND calendar_date >= %s AND calendar_date <= %s "
                "ORDER BY calendar_date",
                (user["id"], start.isoformat(), end.isoformat()),
            ).fetchall()
        finally:
            conn.close()
        return {"days": [dict(r) for r in rows]}

    # -- ask ------------------------------------------------------------------

    @app.post("/ask")
    async def ask(
        body: AskRequest, request: Request, user: dict = Depends(get_user)
    ):
        auth: AuthService = request.app.state.auth
        if not auth.user_llm_configured(user):
            raise HTTPException(
                503,
                "the LLM agent is not configured: set LLM_API_KEY (or a local "
                "LLM_BASE_URL with LLM_MODEL) in the server .env and restart",
            )
        user_cfg = _user_agent_cfg(request.app.state.cfg, user, auth)
        db = _readonly(
            request.app.state.cfg,
            user["id"],
            excluded_types=user.get("excluded_data_types") or "",
        )
        state: UserState = request.app.state.state
        chart_cache = getattr(request.app.state, "chart_cache", {})

        is_streaming = body.stream or (request.headers.get("accept") == "text/event-stream")

        if not is_streaming:
            try:
                memory = PgMemory(state, user["id"])
                agent = _build_agent(
                    user_cfg,
                    db,
                    memory=memory,
                    principles=PgPrinciples(state, user["id"]),
                    training=TrainingSeason(request.app.state.training, user["id"]),
                    chart_cache=chart_cache,
                )
                if body.history:
                    history = _history_to_messages(body.history)
                else:
                    history = state.get_session_messages(user["id"])
                    if history:
                        _refresh_resumed_prompt(history, db)
                result = await agent.run(body.question, message_history=history)

                def _trace_writer(record: dict) -> None:
                    state.append_trace(user["id"], record)

                _record_turn(
                    user_cfg,
                    body.question,
                    result,
                    trace_writer=_trace_writer,
                )
                answer = _final_answer(result)
                text, specs = _extract_charts(answer)
                text = _replace_workout_dumps(text)
                figures = [chart_cache.get(s.get("sql", "").strip()) for s in specs]
                messages = _prune_session_messages(result.all_messages())
                state.set_session_messages(user["id"], messages)
                return {
                    "answer": text,
                    "chart_specs": specs,
                    "chart_figures": figures,
                    "tokens": _messages_token_estimate(messages),
                }
            except Exception as exc:  # noqa: BLE001 - a bad question must not crash the server
                logger.exception("ask failed for user %s", user["id"])
                raise HTTPException(500, f"ask failed: {exc}") from exc
            finally:
                db.close()

        # Streaming path via Server-Sent Events (SSE)
        import asyncio

        async def event_generator():
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_status(msg: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, ("status", {"text": msg}))

            async def runner():
                try:
                    memory = PgMemory(state, user["id"])
                    agent = _build_agent(
                        user_cfg,
                        db,
                        memory=memory,
                        principles=PgPrinciples(state, user["id"]),
                        training=TrainingSeason(request.app.state.training, user["id"]),
                        chart_cache=chart_cache,
                        on_status=on_status,
                    )
                    if body.history:
                        history = _history_to_messages(body.history)
                    else:
                        history = state.get_session_messages(user["id"])
                        if history:
                            _refresh_resumed_prompt(history, db)

                    # ``agent.run_stream`` treats the *first* text part as the
                    # final output and stops the graph there, so a model that
                    # narrates before a tool call ("Let me check…") would have
                    # that preamble returned as the whole answer. ``agent.iter``
                    # streams the same text but always runs the graph to
                    # completion, so the real final output is captured.
                    #
                    # Text emitted in a response that *also* carries a tool call
                    # is pre-tool narration, not the answer — only the response
                    # with no tool calls is the final output. Buffer each model
                    # response's text and forward it only once that response is
                    # known to be tool-call-free, so the live bubble shows the
                    # answer (plus the tool ``status`` updates) rather than the
                    # model's running commentary.
                    from pydantic_ai.messages import (
                        PartDeltaEvent,
                        PartStartEvent,
                        TextPart,
                        TextPartDelta,
                        ToolCallPart,
                    )

                    async with agent.iter(body.question, message_history=history) as agent_run:
                        async for node in agent_run:
                            if not agent.is_model_request_node(node):
                                continue
                            buffered: list[str] = []
                            async with node.stream(agent_run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                        delta = event.part.content
                                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                        delta = event.delta.content_delta
                                    else:
                                        continue
                                    if delta:
                                        buffered.append(delta)
                                has_tool_calls = any(
                                    isinstance(part, ToolCallPart) for part in stream.response.parts
                                )
                            if buffered and not has_tool_calls:
                                await queue.put(("delta", {"text": "".join(buffered)}))

                        if agent_run.result is None:
                            raise RuntimeError("agent run finished without a result")
                        result = agent_run.result
                        answer = _final_answer(result)

                        def _trace_writer(record: dict) -> None:
                            state.append_trace(user["id"], record)

                        _record_turn(
                            user_cfg,
                            body.question,
                            result,
                            trace_writer=_trace_writer,
                            answer=answer,
                        )
                        text, specs = _extract_charts(answer)
                        text = _replace_workout_dumps(text)
                        figures = [chart_cache.get(s.get("sql", "").strip()) for s in specs]
                        messages = _prune_session_messages(result.all_messages())
                        state.set_session_messages(user["id"], messages)

                        await queue.put((
                            "done",
                            {
                                "answer": text,
                                "chart_specs": specs,
                                "chart_figures": figures,
                                "tokens": _messages_token_estimate(messages),
                            },
                        ))
                except Exception as exc:
                    logger.exception("streaming ask failed for user %s", user["id"])
                    await queue.put(("error", {"error": str(exc)}))
                finally:
                    db.close()

            runner_task = asyncio.create_task(runner())
            try:
                while True:
                    event_type, payload = await queue.get()
                    yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if event_type in ("done", "error"):
                        break
            finally:
                if not runner_task.done():
                    runner_task.cancel()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

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
        cache = getattr(request.app.state, "chart_cache", {})
        for item in history:
            if item.get("role") in ("assistant", "bot"):
                text, specs = _extract_charts(item["content"])
                item["content"] = _replace_workout_dumps(text)
                item["chart_specs"] = specs
                item["chart_figures"] = [
                    cache.get(s.get("sql", "").strip()) for s in specs
                ] if cache else []
        return {
            "history": history,
            "tokens": _messages_token_estimate(messages),
        }

    @app.post("/ask/clear")
    def ask_clear(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """Start a fresh session: drop the stored conversation so the next ask
        resumes with no prior context (long-term memory is kept)."""
        state: UserState = request.app.state.state
        state.clear_session(user["id"])
        return {"status": "ok"}

    @app.post("/ask/compact")
    def ask_compact(request: Request, user: dict = Depends(get_user)) -> dict[str, Any]:
        """Fold the stored conversation into a compact summary (system-prompt
        head kept), so the next turn resumes from a small seed instead of the
        whole transcript. Called by the chat's Compact button — it is never run
        automatically, so a slow tool-calling turn is never slowed further."""
        auth: AuthService = request.app.state.auth
        if not auth.user_llm_configured(user):
            raise HTTPException(
                503, "the LLM agent is not configured: set LLM_API_KEY (or a "
                "local LLM_BASE_URL with LLM_MODEL) in the server .env and restart"
            )
        user_cfg = _user_agent_cfg(request.app.state.cfg, user, auth)
        db = _readonly(
            request.app.state.cfg,
            user["id"],
            excluded_types=user.get("excluded_data_types") or "",
        )
        state: UserState = request.app.state.state
        try:
            messages = state.get_session_messages(user["id"]) or []
            if not messages:
                return {"tokens": 0, "unchanged": True}
            agent = _build_agent(
                user_cfg,
                db,
                memory=PgMemory(state, user["id"]),
                principles=PgPrinciples(state, user["id"]),
                training=TrainingSeason(request.app.state.training, user["id"]),
            )
            compacted = _auto_compact(agent, messages, max_tokens=0)
            state.set_session_messages(user["id"], compacted)
            return {
                "tokens": _messages_token_estimate(compacted),
                "unchanged": compacted is messages,
            }
        finally:
            db.close()

    @app.post("/ask/chart")
    def ask_chart(
        body: ChartRequest, request: Request, user: dict = Depends(get_user)
    ) -> dict[str, Any]:
        cfg = request.app.state.cfg
        spec = body.spec
        sql = spec.get("sql")
        if not isinstance(sql, str):
            raise HTTPException(400, "chart spec needs a string 'sql' key")
        cache = getattr(request.app.state, "chart_cache", None)
        if cache:
            cached = cache.get(sql.strip()) or cache.get(json.dumps(spec, sort_keys=True))
            if cached:
                return cached
        db = _readonly(cfg, user["id"])
        try:
            result = db.run_sql(sql)
            figure = _build_chart_figure(spec, result)
            fig_dict = json.loads(figure.to_json())
            if cache is not None:
                if len(cache) > 500:
                    cache.clear()
                cache[sql.strip()] = fig_dict
                cache[json.dumps(spec, sort_keys=True)] = fig_dict
            return fig_dict
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
