"""Per-user agent state (session, memory, trace) persisted in Postgres.

The web agent keeps its conversation history, long-term memory and tool-call
trace as rows in the ``user_state`` table (``user_id`` + ``key`` + ``value``),
scoped by the same Row-Level Security as every data table, so each account
reads and writes only its own rows. Nothing user-facing is stored on disk.

Connections come from a shared pool and set ``app.user_id`` per transaction
(the ``true`` flag makes ``set_config`` transaction-scoped), exactly like the
read-only agent does, so RLS applies to every statement.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from ..db import open_pg_pool

_KEY_MEMORY = "memory"
_KEY_SESSION = "web_session"
_KEY_TRACE = "trace"

_MAX_KEY = 80
_MAX_VALUE = 2000

#: Allowed training-plan values (kept here so the store, the API and the agent
#: tools share one vocabulary).
ACTIVITY_TYPES = ("run", "cycle", "swim", "strength", "rest", "other")
INTENSITIES = ("easy", "moderate", "hard", "race_pace")

#: Garmin activity typeKeys that satisfy each plan ``activity_type`` when
#: auto-matching completed activities. ``rest`` and ``other`` are handled
#: specially (absence / any type) and are intentionally absent here.
_GARMIN_TYPE_MAP: dict[str, frozenset[str]] = {
    "run": frozenset({
        "running", "track_running", "trail_running", "indoor_running",
        "treadmill_running", "virtual_run", "street_running", "ultra_run",
        "running_treadmill", "running_street", "running_track", "running_trail",
    }),
    "cycle": frozenset({
        "cycling", "road_biking", "mountain_biking", "indoor_cycling",
        "e_biking", "e_mountain_biking", "cyclocross", "gravel_cycling", "bmx",
        "track_cycling", "recumbent_cycling", "hand_cycling",
    }),
    "swim": frozenset({
        "lap_swimming", "open_water_swimming", "pool_swimming",
    }),
    "strength": frozenset({
        "strength_training", "weight_training",
    }),
}


class UserState:
    """Per-user key/value rows in ``user_state`` (RLS-scoped)."""

    def __init__(self, url: str) -> None:
        self._pool = open_pg_pool(url, min_size=1, max_size=4)

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    def get(self, user_id: int, key: str) -> str | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "SELECT value FROM user_state WHERE user_id = %s AND key = %s",
                (user_id, key),
            ).fetchone()
        return row["value"] if row else None

    def set(self, user_id: int, key: str, value: str) -> None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "INSERT INTO user_state (user_id, key, value, updated_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (user_id, key) DO UPDATE SET "
                "value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
                (user_id, key, value, datetime.now(timezone.utc).isoformat()),
            )

    def delete(self, user_id: int, key: str) -> None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "DELETE FROM user_state WHERE user_id = %s AND key = %s",
                (user_id, key),
            )

    def get_session_messages(self, user_id: int) -> list[Any] | None:
        """Load the persisted conversation as pydantic-ai messages (or None)."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        raw = self.get(user_id, _KEY_SESSION)
        if not raw:
            return None
        return ModelMessagesTypeAdapter.validate_json(raw)

    def set_session_messages(self, user_id: int, messages: list[Any]) -> None:
        """Persist the conversation as pydantic-ai message JSON under a stable key."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        self.set(
            user_id,
            _KEY_SESSION,
            ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8"),
        )

    def clear_session(self, user_id: int) -> None:
        """Drop the stored conversation so the next turn starts a fresh session."""
        self.delete(user_id, _KEY_SESSION)

    def append_trace(self, user_id: int, record: dict[str, Any]) -> None:
        """Append one trace record to the user's stored trace list (JSON)."""
        raw = self.get(user_id, _KEY_TRACE)
        rows: list[dict[str, Any]] = []
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    rows = parsed
            except json.JSONDecodeError:
                rows = []
        rows.append(record)
        self.set(user_id, _KEY_TRACE, json.dumps(rows, ensure_ascii=False))

    def close(self) -> None:
        self._pool.close()


class PgMemory:
    """DB-backed long-term memory implementing the agent's memory interface.

    Facts are stored as one JSON dict under the ``memory`` key of the user's
    ``user_state`` row. Same validation rules as the file-backed version.
    """

    def __init__(self, state: UserState, user_id: int) -> None:
        self._state = state
        self._user_id = user_id

    def _read(self) -> dict[str, str]:
        raw = self._state.get(self._user_id, _KEY_MEMORY)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)
        }

    def get(self) -> dict[str, str]:
        return self._read()

    def remember(self, key: str, value: str) -> None:
        key, value = key.strip(), value.strip()
        if not key or len(key) > _MAX_KEY:
            raise ValueError(f"key must be 1..{_MAX_KEY} characters")
        if len(value) > _MAX_VALUE:
            raise ValueError(f"value must be at most {_MAX_VALUE} characters")
        data = self._read()
        data[key] = value
        self._state.set(
            self._user_id,
            _KEY_MEMORY,
            json.dumps(data, indent=2, ensure_ascii=False),
        )

    def forget(self, key: str) -> bool:
        data = self._read()
        if key not in data:
            return False
        del data[key]
        self._state.set(
            self._user_id,
            _KEY_MEMORY,
            json.dumps(data, indent=2, ensure_ascii=False),
        )
        return True


def _normalize_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise one workout dict; raises ``ValueError``.

    ``planned_date`` (YYYY-MM-DD) and ``activity_type`` are required; every
    optional field is coerced to its column type (None/null stays NULL).
    """
    planned = data.get("planned_date")
    if not isinstance(planned, str) or not planned.strip():
        raise ValueError("planned_date is required (YYYY-MM-DD)")
    try:
        date.fromisoformat(planned.strip())
    except ValueError as exc:
        raise ValueError(f"planned_date must be YYYY-MM-DD, got {planned!r}") from exc
    planned = planned.strip()

    atype = (data.get("activity_type") or "").strip().lower()
    if atype not in ACTIVITY_TYPES:
        raise ValueError(
            f"activity_type must be one of: {', '.join(ACTIVITY_TYPES)}"
        )

    intensity = data.get("intensity")
    if intensity in (None, ""):
        intensity = None
    else:
        intensity = str(intensity).strip().lower()
        if intensity not in INTENSITIES:
            raise ValueError(
                f"intensity must be one of: {', '.join(INTENSITIES)}"
            )

    duration_min = data.get("duration_min")
    if duration_min in (None, ""):
        duration_min = None
    else:
        try:
            duration_min = int(duration_min)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_min must be an integer (minutes)") from exc
        if duration_min < 0:
            raise ValueError("duration_min must be >= 0")

    distance_km = data.get("distance_km")
    if distance_km in (None, ""):
        distance_km = None
    else:
        try:
            distance_km = float(distance_km)
        except (TypeError, ValueError) as exc:
            raise ValueError("distance_km must be a number (km)") from exc
        if distance_km < 0:
            raise ValueError("distance_km must be >= 0")

    title = data.get("title")
    description = data.get("description")
    completed = data.get("completed", False)
    if isinstance(completed, str):
        completed = completed.strip().lower() in ("1", "true", "yes", "on")
    return {
        "planned_date": planned,
        "activity_type": atype,
        "title": str(title) if title not in (None, "") else None,
        "description": str(description) if description not in (None, "") else None,
        "duration_min": duration_min,
        "distance_km": distance_km,
        "intensity": intensity,
        "completed": bool(completed),
    }


def _plan_row(row: Any) -> dict[str, Any]:
    """Shape one DB row into the JSON the API/agent/tab all consume."""
    return {
        "id": row["id"],
        "planned_date": row["planned_date"],
        "activity_type": row["activity_type"],
        "title": row["title"],
        "description": row["description"],
        "duration_min": row["duration_min"],
        "distance_km": row["distance_km"],
        "intensity": row["intensity"],
        "completed": row["completed"],
        "completed_activity_id": row["completed_activity_id"],
    }


class TrainingPlanStore:
    """Per-account rows in the ``training_plan`` table (RLS-scoped).

    Like ``UserState``, connections come from a shared writer-role pool and set
    ``app.user_id`` per transaction, so Row-Level Security isolates every
    workout to its account.
    """

    def __init__(self, url: str) -> None:
        self._pool = open_pg_pool(url, min_size=1, max_size=4)

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    def list(
        self,
        user_id: int,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict[str, Any]]:
        """All workouts, optionally bounded by inclusive planned_date range."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            sql = "SELECT * FROM training_plan WHERE user_id = %s"
            params: list[Any] = [user_id]
            if date_start:
                sql += " AND planned_date >= %s"
                params.append(date_start)
            if date_end:
                sql += " AND planned_date <= %s"
                params.append(date_end)
            sql += " ORDER BY planned_date, id"
            rows = conn.execute(sql, params).fetchall()
        return [_plan_row(r) for r in rows]

    def get(self, user_id: int, workout_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "SELECT * FROM training_plan WHERE user_id = %s AND id = %s",
                (user_id, workout_id),
            ).fetchone()
        return _plan_row(row) if row else None

    def create(self, user_id: int, data: dict[str, Any]) -> dict[str, Any]:
        fields = _normalize_plan(data)
        now = datetime.now(timezone.utc).isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "INSERT INTO training_plan (user_id, planned_date, activity_type, "
                "title, description, duration_min, distance_km, intensity, "
                "completed, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    user_id, fields["planned_date"], fields["activity_type"],
                    fields["title"], fields["description"], fields["duration_min"],
                    fields["distance_km"], fields["intensity"], fields["completed"],
                    now, now,
                ),
            ).fetchone()
        return _plan_row(row)

    def update(
        self, user_id: int, workout_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        fields = _normalize_plan(data)
        now = datetime.now(timezone.utc).isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "UPDATE training_plan SET planned_date = %s, activity_type = %s, "
                "title = %s, description = %s, duration_min = %s, distance_km = %s, "
                "intensity = %s, completed = %s, updated_at = %s "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (
                    fields["planned_date"], fields["activity_type"], fields["title"],
                    fields["description"], fields["duration_min"], fields["distance_km"],
                    fields["intensity"], fields["completed"], now, user_id, workout_id,
                ),
            ).fetchone()
        return _plan_row(row) if row else None

    def delete(self, user_id: int, workout_id: int) -> bool:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            cur = conn.execute(
                "DELETE FROM training_plan WHERE user_id = %s AND id = %s",
                (user_id, workout_id),
            )
        return cur.rowcount > 0

    def delete_all(self, user_id: int) -> int:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            cur = conn.execute(
                "DELETE FROM training_plan WHERE user_id = %s", (user_id,)
            )
        return cur.rowcount

    def apply(self, user_id: int, spec: dict[str, Any]) -> dict[str, Any]:
        """Apply a batch edit (the agent's ``update_training_plan`` tool).

        ``spec`` is a JSON object with optional keys: ``replace`` (bool, wipe
        the whole plan first), ``workouts`` (list of workout dicts — a dict
        with an ``id`` updates that workout, otherwise it is created) and
        ``delete_ids`` (list of ids to delete). Returns an action summary.
        """
        deleted = 0
        if spec.get("replace"):
            deleted += self.delete_all(user_id)
        for wid in spec.get("delete_ids") or []:
            try:
                wid = int(wid)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"delete_ids must be integers, got {wid!r}") from exc
            if self.delete(user_id, wid):
                deleted += 1
        added = 0
        updated = 0
        for workout in spec.get("workouts") or []:
            if not isinstance(workout, dict):
                raise ValueError("each workout must be a JSON object")
            wid = workout.get("id")
            if wid is not None:
                try:
                    wid = int(wid)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"workout id must be an integer, got {wid!r}"
                    ) from exc
                if self.update(user_id, wid, workout):
                    updated += 1
                else:
                    raise ValueError(f"workout id {wid} not found")
            else:
                self.create(user_id, workout)
                added += 1
        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "total": len(self.list(user_id)),
        }

    def autocomplete(self, user_id: int) -> dict[str, int]:
        """Mark planned workouts complete by matching synced activities.

        Best-effort and idempotent: only flips ``completed`` from false to
        true, never un-completes. A non-rest workout is linked to the same-day
        activity whose Garmin type maps to its ``activity_type`` (closest by
        distance/duration); a past rest workout is completed when no activity
        exists that day. Future workouts are never touched. Returns
        {"completed": n}.
        """
        today = date.today().isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            workouts = conn.execute(
                "SELECT * FROM training_plan WHERE user_id = %s AND completed = false "
                "ORDER BY planned_date, id",
                (user_id,),
            ).fetchall()
            if not workouts:
                return {"completed": 0}
            lo = min(w["planned_date"] for w in workouts)
            hi = max(w["planned_date"] for w in workouts)
            acts = conn.execute(
                "SELECT activity_id, start_date, activity_type, distance_km, "
                "duration_hours FROM activity_summaries "
                "WHERE user_id = %s AND start_date >= %s AND start_date <= %s",
                (user_id, lo, hi),
            ).fetchall()
        by_date: dict[str, list[dict[str, Any]]] = {}
        for a in acts:
            by_date.setdefault(a["start_date"], []).append(a)
        completed = 0
        for w in workouts:
            if w["planned_date"] > today:
                continue
            day_acts = by_date.get(w["planned_date"], [])
            if w["activity_type"] == "rest":
                if w["planned_date"] < today and not day_acts:
                    self._mark_completed(user_id, w["id"], None)
                    completed += 1
                continue
            candidates = [
                a for a in day_acts
                if _activity_matches(w["activity_type"], a["activity_type"])
            ]
            best = _closest_activity(candidates, w)
            if best is not None:
                self._mark_completed(user_id, w["id"], best["activity_id"])
                completed += 1
        return {"completed": completed}

    def _mark_completed(
        self, user_id: int, workout_id: int, activity_id: int | None
    ) -> None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "UPDATE training_plan SET completed = true, "
                "completed_activity_id = %s, updated_at = %s "
                "WHERE user_id = %s AND id = %s",
                (activity_id, datetime.now(timezone.utc).isoformat(), user_id, workout_id),
            )

    def close(self) -> None:
        self._pool.close()


def _activity_matches(plan_type: str, garmin_type: str | None) -> bool:
    """True when a Garmin activity typeKey satisfies a plan ``activity_type``."""
    if plan_type == "other":
        return True
    allowed = _GARMIN_TYPE_MAP.get(plan_type)
    return allowed is not None and garmin_type in allowed


def _closest_activity(
    candidates: list[dict[str, Any]], workout: dict[str, Any]
) -> dict[str, Any] | None:
    """Pick the candidate nearest the planned distance/duration."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if workout.get("distance_km") is not None:
        with_distance = [a for a in candidates if a.get("distance_km") is not None]
        if with_distance:
            return min(
                with_distance,
                key=lambda a: abs(workout["distance_km"] - a["distance_km"]),
            )
    if workout.get("duration_min") is not None:
        with_duration = [a for a in candidates if a.get("duration_hours") is not None]
        if with_duration:
            return min(
                with_duration,
                key=lambda a: abs(workout["duration_min"] - a["duration_hours"] * 60),
            )
    return candidates[0]


class TrainingPlan:
    """Per-user facade over ``TrainingPlanStore`` (mirrors the ``PgMemory``
    pattern) so the agent's plan tools stay user-scoped without threading a
    user id through ``build_agent``."""

    def __init__(self, store: TrainingPlanStore, user_id: int) -> None:
        self._store = store
        self._user_id = user_id

    def list(
        self,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._store.list(self._user_id, date_start, date_end)

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._store.apply(self._user_id, spec)
