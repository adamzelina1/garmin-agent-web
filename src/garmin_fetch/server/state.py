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
from datetime import date, datetime, timedelta, timezone
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

#: Workout lifecycle. ``completed`` and ``partial`` both count as done; the
#: legacy ``completed`` boolean column is kept in sync with this so older
#: queries (and the read-only agent's SQL) keep working.
PLAN_STATUSES = ("planned", "completed", "partial", "skipped")

#: Garmin activity typeKeys that satisfy each plan ``activity_type`` when
#: auto-matching completed activities. ``rest`` and ``other`` are handled
#: specially (absence / any type) and are intentionally absent here. This is the
#: single source of truth: the plan tab's weekly volume matching fetches it from
#: ``GET /training-plan/activity-types`` rather than keeping its own copy.
GARMIN_TYPE_MAP: dict[str, frozenset[str]] = {
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


def _normalize_plan(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate and normalise one workout dict; raises ``ValueError``.

    ``planned_date`` (YYYY-MM-DD) and ``activity_type`` are required; every
    optional field is coerced to its column type (None/null stays NULL). With
    ``partial`` only the keys present in ``data`` are returned, so an edit can
    change one field without re-sending (or nulling) the rest.

    ``status`` is the authoritative lifecycle value (planned/completed/partial/
    skipped); a legacy ``completed`` boolean is accepted and mapped onto it, and
    the boolean column is always written in sync with the status.
    """
    keys = list(data) if partial else (
        "planned_date", "activity_type", "title", "description", "duration_min",
        "distance_km", "intensity", "target_pace_min_km", "target_hr_zone",
        "target_power_w", "status", "completed", "goal_id", "block_id",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key == "planned_date":
            fields[key] = _parse_required_date(value, "planned_date")
        elif key == "activity_type":
            atype = (str(value) if value is not None else "").strip().lower()
            if atype not in ACTIVITY_TYPES:
                raise ValueError(
                    f"activity_type must be one of: {', '.join(ACTIVITY_TYPES)}"
                )
            fields[key] = atype
        elif key in ("title", "description", "target_hr_zone"):
            fields[key] = _text(value)
        elif key == "intensity":
            if value in (None, ""):
                fields[key] = None
            else:
                intensity = str(value).strip().lower()
                if intensity not in INTENSITIES:
                    raise ValueError(
                        f"intensity must be one of: {', '.join(INTENSITIES)}"
                    )
                fields[key] = intensity
        elif key == "duration_min":
            fields[key] = _parse_opt_nonneg_int(value, "duration_min")
        elif key == "distance_km":
            fields[key] = _parse_opt_float(value, "distance_km")
        elif key == "target_pace_min_km":
            pace = _parse_opt_float(value, "target_pace_min_km")
            if pace is not None and not (1.0 <= pace <= 60.0):
                raise ValueError(
                    "target_pace_min_km must be between 1 and 60 (decimal min/km)"
                )
            fields[key] = pace
        elif key == "target_power_w":
            fields[key] = _parse_opt_nonneg_int(value, "target_power_w")
        elif key == "goal_id":
            fields[key] = _opt_int(value, "goal_id")
        elif key == "block_id":
            fields[key] = _opt_int(value, "block_id")
        elif key == "status":
            # A model-dumped null status means "not supplied" — the lifecycle is
            # then derived from the completed boolean below.
            if value in (None, ""):
                continue
            fields[key] = _normalize_status(value)
        elif key == "completed":
            done = value
            if isinstance(done, str):
                done = done.strip().lower() in ("1", "true", "yes", "on")
            # An explicit status wins when both are supplied.
            fields.setdefault("status", "completed" if done else "planned")
    if "status" in fields:
        fields["completed"] = fields["status"] in ("completed", "partial")
    return fields


def _normalize_status(value: Any) -> str:
    """Validate a workout status against :data:`PLAN_STATUSES`."""
    status = str(value).strip().lower()
    if status not in PLAN_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(PLAN_STATUSES)}")
    return status



def _plan_row(row: Any) -> dict[str, Any]:
    """Shape one DB row into the JSON the API/agent/tab all consume.

    ``status`` is the authoritative lifecycle value; a legacy row with no
    status falls back to its ``completed`` boolean, and ``completed`` is always
    derived from the status so the two can never disagree.
    """
    status = row.get("status")
    if status not in PLAN_STATUSES:
        status = "completed" if row["completed"] else "planned"
    return {
        "id": row["id"],
        "planned_date": row["planned_date"],
        "activity_type": row["activity_type"],
        "title": row["title"],
        "description": row["description"],
        "duration_min": row["duration_min"],
        "distance_km": row["distance_km"],
        "intensity": row["intensity"],
        "target_pace_min_km": row.get("target_pace_min_km"),
        "target_hr_zone": row.get("target_hr_zone"),
        "target_power_w": row.get("target_power_w"),
        "status": status,
        "completed": status in ("completed", "partial"),
        "completed_activity_id": row["completed_activity_id"],
        "goal_id": row["goal_id"],
        "block_id": row["block_id"],
    }


def _parse_required_date(value: Any, label: str) -> str:
    """Parse a required YYYY-MM-DD -> normalised str; raises ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required (YYYY-MM-DD)")
    try:
        date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD, got {value!r}") from exc
    return value.strip()


def _parse_opt_nonneg_int(value: Any, label: str) -> int | None:
    """Parse an optional non-negative integer -> int | None."""
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{label} must be >= 0")
    return number



def _parse_opt_date(value: Any, label: str) -> str | None:
    """Parse an optional YYYY-MM-DD (or '' / None) -> str | None."""
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be YYYY-MM-DD")
    try:
        date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD, got {value!r}") from exc
    return value.strip()


def _parse_opt_float(value: Any, label: str) -> float | None:
    """Parse an optional non-negative number -> float | None."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if number < 0:
        raise ValueError(f"{label} must be >= 0")
    return number


def _text(value: Any) -> str | None:
    """Strip a scalar to its trimmed text (''/None -> None)."""
    return str(value).strip() if value not in (None, "") else None


def _opt_int(value: Any, label: str) -> int | None:
    """Parse an optional integer (''/None -> None); raises ``ValueError``."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


#: Goal columns with a plain text normalizer (strip; '' -> NULL).
_GOAL_TEXT_FIELDS = (
    "title", "sport", "event_type", "target_time", "notes", "target_id", "meta",
)
#: Block columns with a plain text normalizer.
_BLOCK_TEXT_FIELDS = ("name", "focus", "notes", "meta")


def _normalize_goal(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate/normalise one goal dict; raises ``ValueError``.

    The goal is a long-term anchor (an event/target such as a marathon or a
    half-marathon). sport/event_type are free text (no enum); ``target_date``
    must be YYYY-MM-DD when present. With ``partial`` only the keys present in
    ``data`` are returned (absent columns keep their stored value).
    """
    keys = list(data) if partial else _GOAL_TEXT_FIELDS + (
        "target_date", "target_distance_km", "target_id",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key in _GOAL_TEXT_FIELDS:
            fields[key] = _text(value)
        elif key == "target_date":
            fields[key] = _parse_opt_date(value, "target_date")
        elif key == "target_distance_km":
            fields[key] = _parse_opt_float(value, "target_distance_km")
    return fields


def _normalize_block(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate/normalise one block dict; raises ``ValueError``.

    Blocks are periodised phases of a goal: ``name``/``focus``/``notes`` are free
    text, and the dates are optional (a block may be undated, defined only by
    sort_order + notes). With ``partial`` only the keys present in ``data`` are
    returned (absent columns keep their stored value).
    """
    keys = list(data) if partial else _BLOCK_TEXT_FIELDS + (
        "goal_id", "start_date", "end_date", "sort_order",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key in _BLOCK_TEXT_FIELDS:
            fields[key] = _text(value)
        elif key in ("start_date", "end_date"):
            fields[key] = _parse_opt_date(value, key)
        elif key == "goal_id":
            fields[key] = _opt_int(value, "goal_id")
        elif key == "sort_order":
            fields[key] = _opt_int(value, "sort_order")
    return fields


def _normalize_wave(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate/normalise one wave dict (one week of a microcycle); errors on
    bad ``week_index``.

    A wave carries the target for one week of a block's progression: distance_km
    and/or duration_min, an optional ``intensity`` intent and ``is_deload``
    flag, plus free-text notes. ``week_index`` is 1-based.
    """
    keys = list(data) if partial else (
        "week_index", "distance_km", "duration_min", "intensity", "is_deload",
        "notes",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key == "week_index":
            week_index = _opt_int(value, "week_index")
            if week_index is not None and week_index < 1:
                raise ValueError("week_index must be >= 1")
            fields["week_index"] = week_index
        elif key == "distance_km":
            fields[key] = _parse_opt_float(value, "distance_km")
        elif key == "duration_min":
            fields[key] = _parse_opt_nonneg_int(value, "duration_min")
        elif key == "intensity":
            fields[key] = _text(value)
        elif key == "is_deload":
            deload = value
            if isinstance(deload, str):
                deload = deload.strip().lower() in ("1", "true", "yes", "on")
            fields[key] = None if value is None else bool(deload)
        elif key == "notes":
            fields[key] = _text(value)
    return fields


def _validate_block_dates(start: Any, end: Any) -> None:
    """Reject a block whose ``end_date`` precedes its ``start_date``.

    An inverted range is invisible to the date resolver but still renders in
    the phase strip, so it is rejected up front instead of stored.
    """
    if start and end and str(end) < str(start):
        raise ValueError(
            f"block end_date ({end}) must be on/after start_date ({start})"
        )


def _goal_row(row: Any) -> dict[str, Any]:
    """Shape one ``training_goal`` row into API/agent JSON."""
    return {
        "id": row["id"],
        "title": row["title"],
        "sport": row["sport"],
        "event_type": row["event_type"],
        "target_date": row["target_date"],
        "target_distance_km": row["target_distance_km"],
        "target_time": row["target_time"],
        "notes": row["notes"],
        "target_id": row["target_id"],
        "meta": row["meta"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _block_row(row: Any) -> dict[str, Any]:
    """Shape one ``training_block`` row into API/agent JSON."""
    return {
        "id": row["id"],
        "goal_id": row["goal_id"],
        "name": row["name"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "focus": row["focus"],
        "notes": row["notes"],
        "sort_order": row["sort_order"],
        "meta": row["meta"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _wave_row(row: Any) -> dict[str, Any]:
    """Shape one ``training_wave`` row into API/agent JSON."""
    return {
        "id": row["id"],
        "block_id": row["block_id"],
        "week_index": row["week_index"],
        "distance_km": row["distance_km"],
        "duration_min": row["duration_min"],
        "intensity": row.get("intensity"),
        "is_deload": row.get("is_deload"),
        "notes": row["notes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
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

    def create(self, user_id: int, data: dict[str, Any]) -> dict[str, Any]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._create(conn, user_id, data)

    def _create(
        self, conn: Any, user_id: int, data: dict[str, Any]
    ) -> dict[str, Any]:
        fields = _normalize_plan(data)
        self._validate_links(
            conn, user_id, fields.get("goal_id"), fields.get("block_id")
        )
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "INSERT INTO training_plan (user_id, planned_date, activity_type, "
            "title, description, duration_min, distance_km, intensity, "
            "target_pace_min_km, target_hr_zone, target_power_w, status, "
            "completed, goal_id, block_id, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s) RETURNING *",
            (
                user_id, fields["planned_date"], fields["activity_type"],
                fields.get("title"), fields.get("description"),
                fields.get("duration_min"), fields.get("distance_km"),
                fields.get("intensity"), fields.get("target_pace_min_km"),
                fields.get("target_hr_zone"), fields.get("target_power_w"),
                fields.get("status"), fields.get("completed", False),
                fields.get("goal_id"), fields.get("block_id"), now, now,
            ),
        ).fetchone()
        return _plan_row(row)

    def update(
        self,
        user_id: int,
        workout_id: int,
        data: dict[str, Any],
        *,
        partial: bool = False,
    ) -> dict[str, Any] | None:
        """Update a workout; ``partial`` changes only the supplied fields.

        The default (full) mode rewrites every editable column, so a caller
        must send the complete desired state. ``partial=True`` is the PATCH
        path used by the UI's edit modal and the agent's ``update_training_plan``
        upsert, so changing one field never clears the others (e.g. the
        goal/block link).
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._update(conn, user_id, workout_id, data, partial=partial)

    def _update(
        self,
        conn: Any,
        user_id: int,
        workout_id: int,
        data: dict[str, Any],
        *,
        partial: bool = False,
    ) -> dict[str, Any] | None:
        fields = _normalize_plan(data, partial=partial)
        current = conn.execute(
            "SELECT * FROM training_plan WHERE user_id = %s AND id = %s",
            (user_id, workout_id),
        ).fetchone()
        if current is None:
            return None
        if partial:
            if not fields:
                return _plan_row(current)
            eff_goal = fields.get("goal_id", current["goal_id"])
            eff_block = fields.get("block_id", current["block_id"])
        else:
            eff_goal = fields.get("goal_id")
            eff_block = fields.get("block_id")
        self._validate_links(conn, user_id, eff_goal, eff_block)
        now = datetime.now(timezone.utc).isoformat()
        if partial:
            assigns, params = _build_set(fields, now)
            row = conn.execute(
                f"UPDATE training_plan SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, workout_id),
            ).fetchone()
        else:
            row = conn.execute(
                "UPDATE training_plan SET planned_date = %s, activity_type = %s, "
                "title = %s, description = %s, duration_min = %s, distance_km = %s, "
                "intensity = %s, target_pace_min_km = %s, target_hr_zone = %s, "
                "target_power_w = %s, status = %s, completed = %s, goal_id = %s, "
                "block_id = %s, updated_at = %s "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (
                    fields["planned_date"], fields["activity_type"],
                    fields.get("title"), fields.get("description"),
                    fields.get("duration_min"), fields.get("distance_km"),
                    fields.get("intensity"), fields.get("target_pace_min_km"),
                    fields.get("target_hr_zone"), fields.get("target_power_w"),
                    fields.get("status"), fields.get("completed", False),
                    fields.get("goal_id"), fields.get("block_id"), now,
                    user_id, workout_id,
                ),
            ).fetchone()
        return _plan_row(row) if row else None

    def delete(self, user_id: int, workout_id: int) -> bool:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._delete(conn, user_id, workout_id)

    def _delete(self, conn: Any, user_id: int, workout_id: int) -> bool:
        cur = conn.execute(
            "DELETE FROM training_plan WHERE user_id = %s AND id = %s",
            (user_id, workout_id),
        )
        return cur.rowcount > 0

    def _delete_all(self, conn: Any, user_id: int) -> int:
        cur = conn.execute(
            "DELETE FROM training_plan WHERE user_id = %s", (user_id,)
        )
        return cur.rowcount

    @staticmethod
    def _validate_links(
        conn: Any, user_id: int, goal_id: int | None, block_id: int | None
    ) -> None:
        """Reject dangling or contradictory long-term links.

        ``goal_id``/``block_id`` are plain BIGINTs (no FK), so a workout could
        otherwise be pointed at a goal/block that was never created or was
        deleted — leaving an orphan the agent then reasons about.
        """
        if goal_id is not None:
            row = conn.execute(
                "SELECT 1 FROM training_goal WHERE user_id = %s AND id = %s",
                (user_id, goal_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"goal_id {goal_id} does not exist")
        if block_id is not None:
            row = conn.execute(
                "SELECT goal_id FROM training_block WHERE user_id = %s AND id = %s",
                (user_id, block_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"block_id {block_id} does not exist")
            if goal_id is not None and row["goal_id"] != goal_id:
                raise ValueError(
                    f"block_id {block_id} belongs to goal {row['goal_id']}, "
                    f"not goal {goal_id}"
                )

    def apply(self, user_id: int, spec: dict[str, Any]) -> dict[str, Any]:
        """Apply a batch edit (the agent's ``update_training_plan`` tool).

        ``spec`` keys (all optional):
        - ``replace`` (bool): wipe the whole plan first.
        - ``workouts`` (list): a dict with an ``id`` PATCHes that workout (only
          the supplied fields change); one without an ``id`` creates a workout.
        - ``delete_ids`` (list[int]) / ``delete_range`` ({from,to}): deletions.
        - ``shift`` ({days, from?, to?, ids?, goal_id?}): move matching dates.
        - ``repeat_week`` ({week_start, weeks?, include_completed?}): copy a week.
        - ``undo`` (bool): restore the snapshot taken before the last
          destructive edit.

        The whole batch runs in ONE transaction, so a bad row rolls the entire
        edit back. Destructive edits snapshot the prior plan first, restorable
        with ``undo``.
        """
        deleted = added = updated = shifted = repeated = restored = 0
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            if spec.get("undo"):
                restored = self._undo(conn, user_id)
            destructive = bool(
                spec.get("replace") or spec.get("delete_ids")
                or spec.get("delete_range")
            )
            if destructive:
                self._snapshot(conn, user_id)
            if spec.get("replace"):
                deleted += self._delete_all(conn, user_id)
            for wid in spec.get("delete_ids") or []:
                wid = _opt_int(wid, "delete_ids")
                if wid is None:
                    raise ValueError("delete_ids must contain integer workout ids")
                if self._delete(conn, user_id, wid):
                    deleted += 1
            if spec.get("delete_range"):
                lo, hi = _range_bounds(spec["delete_range"], "delete_range")
                cur = conn.execute(
                    "DELETE FROM training_plan WHERE user_id = %s "
                    "AND planned_date >= %s AND planned_date <= %s",
                    (user_id, lo, hi),
                )
                deleted += cur.rowcount
            if spec.get("shift"):
                shifted = self._shift(conn, user_id, spec["shift"])
            if spec.get("repeat_week"):
                made, _copies = self._repeat_week(conn, user_id, spec["repeat_week"])
                repeated += made
                added += made
            for workout in spec.get("workouts") or []:
                if not isinstance(workout, dict):
                    raise ValueError("each workout must be a JSON object")
                wid = workout.get("id")
                if wid is not None:
                    wid = _opt_int(wid, "workout id")
                    if wid is None:
                        raise ValueError("workout id must be an integer")
                    if self._update(conn, user_id, wid, workout, partial=True) is None:
                        raise ValueError(f"workout id {wid} not found")
                    updated += 1
                else:
                    self._create(conn, user_id, workout)
                    added += 1
            total = conn.execute(
                "SELECT count(*) AS n FROM training_plan WHERE user_id = %s",
                (user_id,),
            ).fetchone()["n"]
        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "shifted": shifted,
            "repeated": repeated,
            "restored": restored,
            "total": total,
        }

    def _shift(self, conn: Any, user_id: int, spec: Any) -> int:
        """Move matching workouts by ``days`` (positive = later)."""
        if not isinstance(spec, dict):
            raise ValueError("shift must be an object")
        days = _opt_int(spec.get("days"), "shift.days")
        if days is None:
            raise ValueError("shift.days is required (integer, may be negative)")
        if days == 0:
            return 0
        where = ["user_id = %s"]
        params: list[Any] = [days, datetime.now(timezone.utc).isoformat(), user_id]
        ids = spec.get("ids")
        if ids:
            try:
                ids = [int(i) for i in ids]
            except (TypeError, ValueError) as exc:
                raise ValueError("shift.ids must contain integers") from exc
            where.append("id = ANY(%s)")
            params.append(ids)
        else:
            if spec.get("from"):
                where.append("planned_date >= %s")
                params.append(_parse_required_date(spec["from"], "shift.from"))
            if spec.get("to"):
                where.append("planned_date <= %s")
                params.append(_parse_required_date(spec["to"], "shift.to"))
            if spec.get("goal_id") is not None:
                where.append("goal_id = %s")
                params.append(_opt_int(spec["goal_id"], "shift.goal_id"))
        cur = conn.execute(
            "UPDATE training_plan SET planned_date = "
            "(planned_date::date + %s)::text, updated_at = %s "
            f"WHERE {' AND '.join(where)}",
            tuple(params),
        )
        return cur.rowcount

    def _repeat_week(
        self, conn: Any, user_id: int, spec: Any
    ) -> tuple[int, list[str]]:
        """Copy one week's workouts into the following ``weeks`` weeks."""
        if not isinstance(spec, dict):
            raise ValueError("repeat_week must be an object")
        week_start = _parse_required_date(
            spec.get("week_start"), "repeat_week.week_start"
        )
        weeks = _opt_int(spec.get("weeks"), "repeat_week.weeks")
        weeks = 1 if weeks is None else weeks
        if weeks < 1 or weeks > 52:
            raise ValueError("repeat_week.weeks must be between 1 and 52")
        include_completed = bool(spec.get("include_completed"))
        start = date.fromisoformat(week_start)
        end = start + timedelta(days=6)
        rows = conn.execute(
            "SELECT * FROM training_plan WHERE user_id = %s "
            "AND planned_date >= %s AND planned_date <= %s "
            "ORDER BY planned_date, id",
            (user_id, start.isoformat(), end.isoformat()),
        ).fetchall()
        if not rows:
            raise ValueError(f"no workouts in the week starting {week_start}")
        created: list[str] = []
        for week in range(1, weeks + 1):
            for row in rows:
                if row["completed"] and not include_completed:
                    continue
                target = (
                    date.fromisoformat(row["planned_date"]) + timedelta(days=7 * week)
                ).isoformat()
                exists = conn.execute(
                    "SELECT 1 FROM training_plan WHERE user_id = %s "
                    "AND planned_date = %s AND activity_type = %s",
                    (user_id, target, row["activity_type"]),
                ).fetchone()
                if exists:
                    continue
                self._create(conn, user_id, {
                    "planned_date": target,
                    "activity_type": row["activity_type"],
                    "title": row["title"],
                    "description": row["description"],
                    "duration_min": row["duration_min"],
                    "distance_km": row["distance_km"],
                    "intensity": row["intensity"],
                    "target_pace_min_km": row.get("target_pace_min_km"),
                    "target_hr_zone": row.get("target_hr_zone"),
                    "target_power_w": row.get("target_power_w"),
                    "goal_id": row["goal_id"],
                    "block_id": row["block_id"],
                })
                created.append(target)
        return len(created), created

    def week_view(
        self,
        user_id: int,
        week_start: str | None = None,
        goal_id: int | None = None,
    ) -> dict[str, Any]:
        """One week's planned workouts + the actual activities synced that week.

        ``week_start`` (default today) is snapped back to Monday; ``goal_id``
        limits the planned side to one long-term goal. Weeks run Mon..Sun.
        """
        day = date.fromisoformat(week_start) if week_start else date.today()
        ws = day - timedelta(days=day.weekday())
        we = ws + timedelta(days=6)
        planned = self.list(user_id, ws.isoformat(), we.isoformat())
        if goal_id is not None:
            planned = [w for w in planned if w["goal_id"] == int(goal_id)]
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            acts = conn.execute(
                "SELECT activity_id, start_date, activity_type, distance_km, "
                "duration_hours FROM activity_summaries "
                "WHERE user_id = %s AND start_date >= %s AND start_date <= %s "
                "ORDER BY start_date",
                (user_id, ws.isoformat(), we.isoformat()),
            ).fetchall()
        return {
            "week_start": ws.isoformat(),
            "week_end": we.isoformat(),
            "planned": planned,
            "actual": [dict(a) for a in acts],
        }

    def undo(self, user_id: int) -> int:
        """Restore the plan from the last destructive-edit snapshot."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._undo(conn, user_id)

    def snapshot(self, user_id: int) -> None:
        """Capture the current plan so a later destructive edit can be undone.

        Called by the REST layer before a delete (the agent's ``apply`` does it
        itself for ``replace``/``delete_ids``/``delete_range``).
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            self._snapshot(conn, user_id)

    def _snapshot(self, conn: Any, user_id: int) -> None:
        rows = conn.execute(
            "SELECT * FROM training_plan WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        _write_snapshot(conn, user_id, "plan", [dict(r) for r in rows])

    def _undo(self, conn: Any, user_id: int) -> int:
        payload = _read_snapshot(conn, user_id, "plan")
        if payload is None:
            raise ValueError("nothing to undo: no plan snapshot stored")
        conn.execute("DELETE FROM training_plan WHERE user_id = %s", (user_id,))
        for w in payload:
            conn.execute(
                "INSERT INTO training_plan (user_id, planned_date, activity_type, "
                "title, description, duration_min, distance_km, intensity, "
                "target_pace_min_km, target_hr_zone, target_power_w, status, "
                "completed, completed_activity_id, goal_id, block_id, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s)",
                (
                    user_id, w.get("planned_date"), w.get("activity_type"),
                    w.get("title"), w.get("description"), w.get("duration_min"),
                    w.get("distance_km"), w.get("intensity"),
                    w.get("target_pace_min_km"), w.get("target_hr_zone"),
                    w.get("target_power_w"), w.get("status"), w.get("completed"),
                    w.get("completed_activity_id"), w.get("goal_id"),
                    w.get("block_id"), w.get("created_at"), w.get("updated_at"),
                ),
            )
        return len(payload)

    def autocomplete(self, user_id: int) -> dict[str, int]:
        """Mark planned workouts complete by matching synced activities.

        Best-effort and idempotent: only moves a workout from planned to
        completed, never un-completes and never touches a skipped one. A
        non-rest workout is matched to an activity of the same family on the
        planned day, then on the adjacent days (±1) — a long run logged the next
        morning still counts. A past rest workout is completed when no activity
        exists that day. Dead ``completed_activity_id`` links (an activity
        removed by a re-parse) are cleared first. Future workouts are never
        touched. Returns {"completed": n}.
        """
        today = date.today().isoformat()
        now = datetime.now(timezone.utc).isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "UPDATE training_plan SET completed_activity_id = NULL, "
                "updated_at = %s WHERE user_id = %s "
                "AND completed_activity_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM activity_summaries a "
                "WHERE a.user_id = %s "
                "AND a.activity_id = training_plan.completed_activity_id)",
                (now, user_id, user_id),
            )
            workouts = conn.execute(
                "SELECT * FROM training_plan WHERE user_id = %s "
                "AND completed = false AND (status IS NULL OR status = 'planned') "
                "ORDER BY planned_date, id",
                (user_id,),
            ).fetchall()
            if not workouts:
                return {"completed": 0}
            lo = (
                date.fromisoformat(min(w["planned_date"] for w in workouts))
                - timedelta(days=1)
            ).isoformat()
            hi = (
                date.fromisoformat(max(w["planned_date"] for w in workouts))
                + timedelta(days=1)
            ).isoformat()
            acts = conn.execute(
                "SELECT activity_id, start_date, activity_type, distance_km, "
                "duration_hours FROM activity_summaries "
                "WHERE user_id = %s AND start_date >= %s AND start_date <= %s",
                (user_id, lo, hi),
            ).fetchall()
            by_date: dict[str, list[dict[str, Any]]] = {}
            for a in acts:
                by_date.setdefault(a["start_date"], []).append(a)
            updates: list[tuple[Any, str, int, int]] = []
            for w in workouts:
                if w["planned_date"] > today:
                    continue
                day_acts = by_date.get(w["planned_date"], [])
                if w["activity_type"] == "rest":
                    if w["planned_date"] < today and not day_acts:
                        updates.append((None, now, user_id, w["id"]))
                    continue
                candidates = [
                    a for a in day_acts
                    if _activity_matches(w["activity_type"], a["activity_type"])
                ]
                if not candidates:
                    for offset in (-1, 1):
                        other = (
                            date.fromisoformat(w["planned_date"])
                            + timedelta(days=offset)
                        ).isoformat()
                        candidates = [
                            a for a in by_date.get(other, [])
                            if _activity_matches(w["activity_type"], a["activity_type"])
                        ]
                        if candidates:
                            break
                best = _closest_activity(candidates, w)
                if best is not None:
                    updates.append((best["activity_id"], now, user_id, w["id"]))
            if updates:
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE training_plan SET completed = true, "
                        "status = 'completed', completed_activity_id = %s, "
                        "updated_at = %s WHERE user_id = %s AND id = %s",
                        updates,
                    )
        return {"completed": len(updates)}

    def close(self) -> None:
        self._pool.close()


def _range_bounds(spec: Any, label: str) -> tuple[str, str]:
    """Read a ``{from, to}`` inclusive date range (order-normalised)."""
    if not isinstance(spec, dict):
        raise ValueError(f"{label} must be an object with 'from' and 'to'")
    lo = _parse_required_date(spec.get("from"), f"{label}.from")
    hi = _parse_required_date(spec.get("to"), f"{label}.to")
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


#: ``user_state`` keys holding the one undo snapshot per domain. Reusing the
#: existing per-user key/value table keeps the snapshots in the same RLS scope
#: and inside the editing transaction, with no extra table.
_SNAPSHOT_KEYS = {"plan": "plan_undo", "anchor": "anchor_undo"}


def _write_snapshot(
    conn: Any, user_id: int, kind: str, payload: Any
) -> None:
    """Store (replace) the one undo snapshot for ``kind`` ('plan'/'anchor')."""
    conn.execute(
        "INSERT INTO user_state (user_id, key, value, updated_at) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (user_id, key) DO UPDATE SET "
        "value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
        (
            user_id, _SNAPSHOT_KEYS[kind], json.dumps(payload, default=str),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _read_snapshot(conn: Any, user_id: int, kind: str) -> Any | None:
    """Load the stored undo snapshot for ``kind`` (None when absent)."""
    row = conn.execute(
        "SELECT value FROM user_state WHERE user_id = %s AND key = %s",
        (user_id, _SNAPSHOT_KEYS[kind]),
    ).fetchone()
    if row is None or not row["value"]:
        return None
    return json.loads(row["value"])


def _activity_matches(plan_type: str, garmin_type: str | None) -> bool:
    """True when a Garmin activity typeKey satisfies a plan ``activity_type``."""
    if plan_type == "other":
        return True
    allowed = GARMIN_TYPE_MAP.get(plan_type)
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

    def week_view(
        self, week_start: str | None = None, goal_id: int | None = None
    ) -> dict[str, Any]:
        return self._store.week_view(self._user_id, week_start, goal_id)


class TrainingGoalStore:
    """Per-account long-term anchors in ``training_goal`` + ``training_block`` +
    ``training_wave`` (RLS-scoped).

    Mirrors ``TrainingPlanStore``: an RLS-scoped writer pool with ``app.user_id``
    set per transaction. Unlike the original single-active design, an account may
    hold *many* goals (a marathon and a half-marathon), each with its own
    periodised blocks; every block may carry a repeating **wave** (microcycle)
    of weekly targets (e.g. Week 1: 50km, Week 2: 55km, Week 3: 60km, Week 4:
    45km deload) that drives the plan tab's weekly progress bars.

    Updates are **partial**: pass only the fields you want changed (PATCH
    semantics), so adjusting one block's dates/notes never rewrites the season.
    """

    def __init__(self, url: str) -> None:
        self._pool = open_pg_pool(url, min_size=1, max_size=4)

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    # -- goals -----------------------------------------------------------------

    def list_goals(self, user_id: int) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            rows = conn.execute(
                "SELECT * FROM training_goal WHERE user_id = %s "
                "ORDER BY id",
                (user_id,),
            ).fetchall()
        return [_goal_row(r) for r in rows]

    def get_goal(self, user_id: int, goal_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "SELECT * FROM training_goal WHERE user_id = %s AND id = %s",
                (user_id, goal_id),
            ).fetchone()
        return _goal_row(row) if row else None

    def create_goal(self, user_id: int, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new goal (appends it; never wipes other goals/blocks).

        ``data`` may carry nested ``blocks`` (list). Each block may carry
        ``waves`` (list of per-week targets). Returns the saved goal row.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._create_goal(conn, user_id, data)

    def _create_goal(
        self, conn: Any, user_id: int, data: dict[str, Any]
    ) -> dict[str, Any]:
        fields = _normalize_goal(data)
        blocks = data.get("blocks") or data.get("replace_blocks") or []
        if not isinstance(blocks, list):
            raise ValueError("goal['blocks'] must be a list")
        now = datetime.now(timezone.utc).isoformat()
        goal_row = conn.execute(
            "INSERT INTO training_goal (user_id, title, sport, event_type, "
            "target_date, target_distance_km, target_time, notes, target_id, "
            "meta, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING *",
            (
                user_id, fields.get("title"), fields.get("sport"),
                fields.get("event_type"), fields.get("target_date"),
                fields.get("target_distance_km"), fields.get("target_time"),
                fields.get("notes"), fields.get("target_id"), fields.get("meta"),
                now, now,
            ),
        ).fetchone()
        self._insert_blocks(conn, user_id, goal_row["id"], blocks, now)
        return _goal_row(goal_row)

    def update_goal(
        self, user_id: int, goal_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Partial update of one goal (never touches other goals).

        Only the fields present in ``data`` change. Block handling:
        ``blocks`` upserts (an entry with an ``id`` patches that block, one
        without creates a block), ``replace_blocks`` swaps the whole set, and
        ``delete_blocks`` removes specific blocks **of this goal**.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._update_goal(conn, user_id, goal_id, data)

    def _update_goal(
        self, conn: Any, user_id: int, goal_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if "replace_blocks" in data and "blocks" in data:
            raise ValueError(
                "pass either 'blocks' (upsert) or 'replace_blocks', not both"
            )
        fields = _normalize_goal(data, partial=True)
        has_children = any(
            key in data for key in ("blocks", "replace_blocks", "delete_blocks")
        )
        if not fields and not has_children:
            return self._get_goal(conn, user_id, goal_id)
        row = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s AND id = %s",
            (user_id, goal_id),
        ).fetchone()
        if row is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        if fields:
            assigns, params = _build_set(fields, now)
            row = conn.execute(
                f"UPDATE training_goal SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, goal_id),
            ).fetchone()
        if "replace_blocks" in data:
            self._replace_blocks(conn, user_id, goal_id, data["replace_blocks"], now)
        elif "blocks" in data:
            self._upsert_blocks(conn, user_id, goal_id, data["blocks"], now)
        for block_id in data.get("delete_blocks") or []:
            block_id = _opt_int(block_id, "delete_blocks")
            if block_id is None:
                raise ValueError("delete_blocks must contain integer block ids")
            if not self._delete_block(conn, user_id, block_id, goal_id=goal_id):
                raise ValueError(f"block {block_id} not found in goal {goal_id}")
        return _goal_row(row)

    def _get_goal(
        self, conn: Any, user_id: int, goal_id: int
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s AND id = %s",
            (user_id, goal_id),
        ).fetchone()
        return _goal_row(row) if row else None

    def delete_goal(self, user_id: int, goal_id: int) -> bool:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._delete_goal(conn, user_id, goal_id)

    def _delete_goal(self, conn: Any, user_id: int, goal_id: int) -> bool:
        """Delete a goal + its blocks/waves, detaching any planned workouts.

        ``training_plan.goal_id``/``block_id`` have no FK, so the links are
        NULLed first — otherwise deleting a goal leaves workouts pointing at a
        goal/block that no longer exists (the agent then reasons about a phase
        it cannot resolve).
        """
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE training_plan SET block_id = NULL, updated_at = %s "
            "WHERE user_id = %s AND block_id IN ("
            "SELECT id FROM training_block WHERE user_id = %s AND goal_id = %s)",
            (now, user_id, user_id, goal_id),
        )
        conn.execute(
            "UPDATE training_plan SET goal_id = NULL, updated_at = %s "
            "WHERE user_id = %s AND goal_id = %s",
            (now, user_id, goal_id),
        )
        conn.execute(
            "DELETE FROM training_wave WHERE user_id = %s AND block_id IN "
            "(SELECT id FROM training_block WHERE user_id = %s AND goal_id = %s)",
            (user_id, user_id, goal_id),
        )
        conn.execute(
            "DELETE FROM training_block WHERE user_id = %s AND goal_id = %s",
            (user_id, goal_id),
        )
        cur = conn.execute(
            "DELETE FROM training_goal WHERE user_id = %s AND id = %s",
            (user_id, goal_id),
        )
        return cur.rowcount > 0

    def apply_spec(
        self, user_id: int, spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply one agent anchor edit atomically (the ``update_training_anchor``
        tool): optional goal deletions, then a goal create/update — all in one
        transaction, so a failure never leaves the season half-reshaped.

        ``undo`` restores the snapshot taken before the last destructive edit
        (goal deletion or ``replace_blocks``). Returns
        ``{"goal": <row|None>, "deleted_goals": n, "restored": n}``.
        """
        deleted = 0
        restored = 0
        row: dict[str, Any] | None = None
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            if spec.get("undo"):
                restored = self._undo(conn, user_id)
            destructive = bool(
                spec.get("delete_goal_ids") or spec.get("replace_blocks")
            )
            if destructive:
                self._snapshot(conn, user_id)
            for gid in spec.get("delete_goal_ids") or []:
                gid = _opt_int(gid, "delete_goal_ids")
                if gid is None:
                    raise ValueError("delete_goal_ids must contain integer goal ids")
                if self._delete_goal(conn, user_id, gid):
                    deleted += 1
            goal_id = spec.get("goal_id")
            has_create_fields = any(
                key in spec for key in (
                    "title", "sport", "event_type", "target_date",
                    "target_distance_km", "target_time", "notes", "target_id",
                    "meta", "blocks", "replace_blocks",
                )
            )
            if goal_id is not None:
                gid = _opt_int(goal_id, "goal_id")
                if gid is None:
                    raise ValueError("goal_id must be an integer")
                row = self._update_goal(conn, user_id, gid, spec)
                if row is None:
                    raise ValueError(f"goal {gid} not found")
            elif has_create_fields:
                row = self._create_goal(conn, user_id, spec)
        return {
            "goal": row,
            "deleted_goals": deleted,
            "restored": restored,
        }

    def week_index(self, user_id: int, block_id: int, day: str | None = None) -> int | None:
        """1-based week of ``day`` (default today) within a dated block."""
        block = self.get_block(user_id, block_id)
        if block is None or not block["start_date"]:
            return None
        day = day or date.today().isoformat()
        offset = (date.fromisoformat(day) - date.fromisoformat(block["start_date"])).days
        return max(1, offset // 7 + 1)

    def resolve_goal(
        self, user_id: int, goal_id: int, day: str | None = None
    ) -> dict[str, Any] | None:
        """One goal with its phases (+ waves), resolved phase and weekly target.

        The single composition of the goal/block/wave resolution used by the REST
        API, the agent tools and (indirectly) the plan tab, so those views can
        never disagree. Returns None when the goal does not exist.
        """
        goal = self.get_goal(user_id, goal_id)
        if goal is None:
            return None
        day = day or date.today().isoformat()
        blocks, waves_by_block = self.list_blocks_with_waves(user_id, goal_id)
        for block in blocks:
            block["waves"] = waves_by_block.get(block["id"], [])
        current = self.current_block(user_id, day, goal_id, blocks=blocks)
        week_index = (
            self.week_index(user_id, current["id"], day) if current else None
        )
        target = self.weekly_target(
            user_id, day, goal_id, blocks=blocks, waves_by_block=waves_by_block
        )
        return {
            "goal": goal,
            "blocks": blocks,
            "current_block": current,
            "week_index": week_index,
            # ``{"block": <row>, "wave": <row>}`` (the API/plan-tab shape); the
            # agent tools slim it down.
            "weekly_target": target,
        }

    def undo(self, user_id: int) -> int:
        """Restore goals/blocks/waves from the last destructive-edit snapshot."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._undo(conn, user_id)

    def snapshot(self, user_id: int) -> None:
        """Capture the current anchor so a later destructive edit can be undone.

        Called by the REST layer before a goal delete or a ``replace_blocks``
        (the agent's ``apply_spec`` does it itself).
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            self._snapshot(conn, user_id)

    def _snapshot(self, conn: Any, user_id: int) -> None:
        goals = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        blocks = conn.execute(
            "SELECT * FROM training_block WHERE user_id = %s ORDER BY goal_id, sort_order, id",
            (user_id,),
        ).fetchall()
        waves = conn.execute(
            "SELECT * FROM training_wave WHERE user_id = %s ORDER BY block_id, week_index",
            (user_id,),
        ).fetchall()
        # A destructive edit re-links or detaches planned workouts, so their
        # goal/block pointers must be restored alongside the anchor itself.
        links = conn.execute(
            "SELECT id, goal_id, block_id FROM training_plan WHERE user_id = %s "
            "AND (goal_id = ANY(%s) OR block_id = ANY(%s))",
            (
                user_id,
                [g["id"] for g in goals] or [0],
                [b["id"] for b in blocks] or [0],
            ),
        ).fetchall()
        _write_snapshot(conn, user_id, "anchor", {
            "goals": [dict(r) for r in goals],
            "blocks": [dict(r) for r in blocks],
            "waves": [dict(r) for r in waves],
            "links": [dict(r) for r in links],
        })

    def _undo(self, conn: Any, user_id: int) -> int:
        """Restore goals/blocks/waves (with their original ids) and re-link the
        planned workouts that pointed at them.

        Ids are re-inserted explicitly: the snapshot rows were deleted by the
        destructive edit, so restoring them cannot collide with the identity
        sequence (which has already advanced past them).
        """
        payload = _read_snapshot(conn, user_id, "anchor")
        if payload is None:
            raise ValueError("nothing to undo: no anchor snapshot stored")
        conn.execute("DELETE FROM training_wave WHERE user_id = %s", (user_id,))
        conn.execute("DELETE FROM training_block WHERE user_id = %s", (user_id,))
        conn.execute("DELETE FROM training_goal WHERE user_id = %s", (user_id,))
        for g in payload.get("goals", []):
            conn.execute(
                "INSERT INTO training_goal (id, user_id, title, sport, event_type, "
                "target_date, target_distance_km, target_time, notes, target_id, "
                "meta, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    g.get("id"), user_id, g.get("title"), g.get("sport"),
                    g.get("event_type"), g.get("target_date"),
                    g.get("target_distance_km"), g.get("target_time"),
                    g.get("notes"), g.get("target_id"), g.get("meta"),
                    g.get("created_at"), g.get("updated_at"),
                ),
            )
        for b in payload.get("blocks", []):
            conn.execute(
                "INSERT INTO training_block (id, user_id, goal_id, name, start_date, "
                "end_date, focus, notes, sort_order, meta, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    b.get("id"), user_id, b.get("goal_id"), b.get("name"),
                    b.get("start_date"), b.get("end_date"), b.get("focus"),
                    b.get("notes"), b.get("sort_order"), b.get("meta"),
                    b.get("created_at"), b.get("updated_at"),
                ),
            )
        for w in payload.get("waves", []):
            conn.execute(
                "INSERT INTO training_wave (id, user_id, block_id, week_index, "
                "distance_km, duration_min, intensity, is_deload, notes, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    w.get("id"), user_id, w.get("block_id"), w.get("week_index"),
                    w.get("distance_km"), w.get("duration_min"),
                    w.get("intensity"), w.get("is_deload"), w.get("notes"),
                    w.get("created_at"), w.get("updated_at"),
                ),
            )
        now = datetime.now(timezone.utc).isoformat()
        for link in payload.get("links", []):
            conn.execute(
                "UPDATE training_plan SET goal_id = %s, block_id = %s, "
                "updated_at = %s WHERE user_id = %s AND id = %s",
                (
                    link.get("goal_id"), link.get("block_id"), now,
                    user_id, link.get("id"),
                ),
            )
        return len(payload.get("goals", []))

    # -- blocks ----------------------------------------------------------------

    def list_blocks(
        self, user_id: int, goal_id: int | None = None
    ) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._list_blocks(conn, user_id, goal_id)

    @staticmethod
    def _list_blocks(
        conn: Any, user_id: int, goal_id: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM training_block WHERE user_id = %s"
        params: list[Any] = [user_id]
        if goal_id is not None:
            sql += " AND goal_id = %s"
            params.append(goal_id)
        sql += " ORDER BY goal_id, sort_order, id"
        rows = conn.execute(sql, params).fetchall()
        return [_block_row(r) for r in rows]

    def list_blocks_with_waves(
        self, user_id: int, goal_id: int | None = None
    ) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
        """Blocks + their waves in two queries (not one per block).

        Returns ``(blocks, {block_id: [wave, ...]})`` so a caller dumping a
        whole goal does not issue an N+1 query per phase.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            blocks = self._list_blocks(conn, user_id, goal_id)
            waves: dict[int, list[dict[str, Any]]] = {}
            if not blocks:
                return blocks, waves
            rows = conn.execute(
                "SELECT * FROM training_wave WHERE user_id = %s "
                "AND block_id = ANY(%s) ORDER BY block_id, week_index",
                (user_id, [b["id"] for b in blocks]),
            ).fetchall()
            for row in rows:
                waves.setdefault(row["block_id"], []).append(_wave_row(row))
        return blocks, waves

    def get_block(self, user_id: int, block_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "SELECT * FROM training_block WHERE user_id = %s AND id = %s",
                (user_id, block_id),
            ).fetchone()
        return _block_row(row) if row else None

    def create_block(
        self, user_id: int, goal_id: int, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Add one block (plus optional ``waves``) to a goal."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._create_block(
                conn, user_id, goal_id, data,
                datetime.now(timezone.utc).isoformat(),
            )

    def _create_block(
        self, conn: Any, user_id: int, goal_id: int, data: dict[str, Any],
        now: str, *, sort_order_default: int | None = None,
    ) -> dict[str, Any]:
        fields = _normalize_block(data)
        fields.pop("goal_id", None)
        waves = data.get("waves") or []
        if not isinstance(waves, list):
            raise ValueError("block['waves'] must be a list")
        _validate_block_dates(fields.get("start_date"), fields.get("end_date"))
        if fields.get("sort_order") is None and sort_order_default is not None:
            fields["sort_order"] = sort_order_default
        self._check_exists(conn, "training_goal", user_id, goal_id)
        columns = ("user_id", "goal_id", "name", "start_date", "end_date",
                   "focus", "notes", "sort_order", "meta", "created_at", "updated_at")
        row = conn.execute(
            f"INSERT INTO training_block ({', '.join(columns)}) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (
                user_id, goal_id, fields.get("name"), fields.get("start_date"),
                fields.get("end_date"), fields.get("focus"), fields.get("notes"),
                fields.get("sort_order"), fields.get("meta"), now, now,
            ),
        ).fetchone()
        self._replace_waves(conn, user_id, row["id"], waves, now)
        return _block_row(row)

    def update_block(
        self, user_id: int, block_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Partial update of one block; ``waves`` (if present) replaces its
        microcycle, ``clear_waves`` drops them. Other blocks stay untouched."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._update_block(conn, user_id, block_id, data)

    def _update_block(
        self, conn: Any, user_id: int, block_id: int, data: dict[str, Any],
        *, goal_id: int | None = None,
    ) -> dict[str, Any] | None:
        fields = _normalize_block(data, partial=True)
        fields.pop("goal_id", None)
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "SELECT * FROM training_block WHERE user_id = %s AND id = %s",
            (user_id, block_id),
        ).fetchone()
        if row is None:
            return None
        if goal_id is not None and row["goal_id"] != goal_id:
            raise ValueError(f"block {block_id} does not belong to goal {goal_id}")
        _validate_block_dates(
            fields.get("start_date", row["start_date"]),
            fields.get("end_date", row["end_date"]),
        )
        if fields:
            assigns, params = _build_set(fields, now)
            row = conn.execute(
                f"UPDATE training_block SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, block_id),
            ).fetchone()
        if "waves" in data:
            waves = data["waves"]
            if not isinstance(waves, list):
                raise ValueError("block['waves'] must be a list")
            self._replace_waves(conn, user_id, block_id, waves, now)
        elif data.get("clear_waves"):
            self._delete_block(conn, user_id, block_id, waves_only=True)
        return _block_row(row)

    def delete_block(self, user_id: int, block_id: int) -> bool:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._delete_block(conn, user_id, block_id)

    # -- waves -----------------------------------------------------------------

    def list_waves(self, user_id: int, block_id: int) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            rows = conn.execute(
                "SELECT * FROM training_wave WHERE user_id = %s AND block_id = %s "
                "ORDER BY week_index",
                (user_id, block_id),
            ).fetchall()
        return [_wave_row(r) for r in rows]

    def replace_waves(
        self, user_id: int, block_id: int, waves: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Wipe a block's waves and write the supplied list (validated)."""
        if not isinstance(waves, list):
            raise ValueError("waves must be a list")
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            now = datetime.now(timezone.utc).isoformat()
            self._replace_waves(conn, user_id, block_id, waves, now)
        return self.list_waves(user_id, block_id)

    # -- resolution ------------------------------------------------------------

    def current_block(
        self, user_id: int, day: str | None = None, goal_id: int | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Best-effort resolver: the block whose date range covers ``day``.

        Returns None when no block has dates (undated anchors) or nothing covers
        the day — the caller surfaces that so the agent can assign dates or treat
        the phase as undated. When several blocks overlap, the *shortest* wins
        (the most specific phase). Pass ``blocks`` to reuse an existing fetch.
        """
        if blocks is None:
            blocks = self.list_blocks(user_id, goal_id)
        day = day or date.today().isoformat()
        covering = [
            b for b in blocks
            if b["start_date"] and b["end_date"]
            and b["start_date"] <= day <= b["end_date"]
        ]
        if not covering:
            return None
        return min(covering, key=lambda b: (
            date.fromisoformat(b["end_date"]) - date.fromisoformat(b["start_date"])
        ).days)

    def weekly_target(
        self, user_id: int, day: str | None = None, goal_id: int | None = None,
        blocks: list[dict[str, Any]] | None = None,
        waves_by_block: dict[int, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve the wave target for the block that covers ``day``.

        A block's ``waves`` define a repeating weekly microcycle. Given the day,
        the week is 1-based relative to the block start; the pattern cycles with
        the period = number of waves, so a 4-week deload wave re-tunes a longer
        block automatically. Waves are stored contiguous (``week_index`` 1..N),
        so positional cycling and ``week_index`` agree. Returns
        ``{"block": ..., "wave": ...}`` or None.
        """
        block = self.current_block(user_id, day, goal_id, blocks=blocks)
        if block is None:
            return None
        if waves_by_block is not None:
            waves = waves_by_block.get(block["id"], [])
        else:
            waves = self.list_waves(user_id, block["id"])
        if not waves:
            return {"block": block, "wave": None}
        day = day or date.today().isoformat()
        block_start = date.fromisoformat(block["start_date"])
        offset = (date.fromisoformat(day) - block_start).days
        week_offset = max(0, offset // 7)
        wave = waves[week_offset % len(waves)]
        return {"block": block, "wave": wave}

    def close(self) -> None:
        self._pool.close()

    # -- private ---------------------------------------------------------------

    def _insert_blocks(
        self, conn: Any, user_id: int, goal_id: int,
        blocks: list[dict[str, Any]], now: str,
    ) -> None:
        """Write a goal's initial block set (create path only)."""
        if not isinstance(blocks, list):
            raise ValueError("goal['blocks'] must be a list")
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise ValueError("each block must be a JSON object")
            self._create_block(
                conn, user_id, goal_id, block, now, sort_order_default=index
            )

    def _upsert_blocks(
        self, conn: Any, user_id: int, goal_id: int,
        blocks: list[dict[str, Any]], now: str,
    ) -> None:
        """Add/patch blocks without touching the others.

        An entry with an ``id`` patches that block (partial); one without an
        ``id`` creates a new block. Nothing is deleted, so block ids — and the
        ``training_plan.block_id`` links pointing at them — stay stable.
        """
        if not isinstance(blocks, list):
            raise ValueError("goal['blocks'] must be a list")
        # New blocks append after the existing phases rather than landing at
        # position 0 (the list index is meaningless when the list mixes patches
        # with additions).
        row = conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) AS m FROM training_block "
            "WHERE user_id = %s AND goal_id = %s",
            (user_id, goal_id),
        ).fetchone()
        next_sort = (row["m"] if row["m"] is not None else -1) + 1
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("each block must be a JSON object")
            block_id = block.get("id")
            if block_id is None:
                self._create_block(
                    conn, user_id, goal_id, block, now, sort_order_default=next_sort
                )
                next_sort += 1
                continue
            block_id = _opt_int(block_id, "block id")
            if block_id is None:
                raise ValueError("block id must be an integer")
            if self._update_block(
                conn, user_id, block_id, block, goal_id=goal_id
            ) is None:
                raise ValueError(f"block {block_id} not found in goal {goal_id}")

    def _replace_blocks(
        self, conn: Any, user_id: int, goal_id: int,
        blocks: list[dict[str, Any]], now: str,
    ) -> None:
        """Swap a goal's whole block set, keeping calendar links where a phase
        with the same name still exists (the rest are detached, never left
        dangling)."""
        if not isinstance(blocks, list):
            raise ValueError("goal['replace_blocks'] must be a list")
        old = conn.execute(
            "SELECT id, name FROM training_block WHERE user_id = %s AND goal_id = %s "
            "ORDER BY sort_order, id",
            (user_id, goal_id),
        ).fetchall()
        old_ids = [row["id"] for row in old]
        old_names = {row["id"]: row["name"] for row in old}
        # Remember which workouts pointed at which old block BEFORE the rows go
        # away — the link is re-established by name after the new set exists.
        linked = (
            conn.execute(
                "SELECT id, block_id FROM training_plan "
                "WHERE user_id = %s AND block_id = ANY(%s)",
                (user_id, old_ids),
            ).fetchall()
            if old_ids else []
        )
        conn.execute(
            "DELETE FROM training_wave WHERE user_id = %s AND block_id IN "
            "(SELECT id FROM training_block WHERE user_id = %s AND goal_id = %s)",
            (user_id, user_id, goal_id),
        )
        conn.execute(
            "DELETE FROM training_block WHERE user_id = %s AND goal_id = %s",
            (user_id, goal_id),
        )
        new = [
            self._create_block(
                conn, user_id, goal_id, block, now, sort_order_default=index
            )
            for index, block in enumerate(blocks)
        ]
        by_name = {
            b["name"].strip().lower(): b["id"] for b in new if b.get("name")
        }
        for row in linked:
            name = old_names.get(row["block_id"])
            new_id = by_name.get(name.strip().lower()) if name else None
            conn.execute(
                "UPDATE training_plan SET block_id = %s, updated_at = %s "
                "WHERE user_id = %s AND id = %s",
                (new_id, now, user_id, row["id"]),
            )

    def _replace_waves(
        self, conn: Any, user_id: int, block_id: int,
        waves: list[dict[str, Any]], now: str,
    ) -> None:
        if not isinstance(waves, list):
            raise ValueError("block['waves'] must be a list")
        normalized: list[dict[str, Any]] = []
        for index, wave in enumerate(waves):
            if not isinstance(wave, dict):
                raise ValueError("each wave must be a JSON object")
            w = _normalize_wave(wave, partial=False)
            if w.get("week_index") is None:
                w["week_index"] = index + 1
            normalized.append(w)
        indexes = sorted(w["week_index"] for w in normalized)
        if indexes != list(range(1, len(normalized) + 1)):
            raise ValueError(
                "wave week_index must be 1..N with no gaps or duplicates, got "
                f"{indexes}"
            )
        conn.execute(
            "DELETE FROM training_wave WHERE user_id = %s AND block_id = %s",
            (user_id, block_id),
        )
        for w in sorted(normalized, key=lambda item: item["week_index"]):
            conn.execute(
                "INSERT INTO training_wave (user_id, block_id, week_index, "
                "distance_km, duration_min, intensity, is_deload, notes, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id, block_id, w["week_index"], w.get("distance_km"),
                    w.get("duration_min"), w.get("intensity"),
                    w.get("is_deload"), w.get("notes"), now, now,
                ),
            )

    def _delete_block(
        self, conn: Any, user_id: int, block_id: int, *, goal_id: int | None = None,
        waves_only: bool = False,
    ) -> bool:
        if goal_id is not None:
            row = conn.execute(
                "SELECT goal_id FROM training_block WHERE user_id = %s AND id = %s",
                (user_id, block_id),
            ).fetchone()
            if row is None or row["goal_id"] != goal_id:
                return False
        if waves_only:
            conn.execute(
                "DELETE FROM training_wave WHERE user_id = %s AND block_id = %s",
                (user_id, block_id),
            )
            return True
        conn.execute(
            "DELETE FROM training_wave WHERE user_id = %s AND block_id = %s",
            (user_id, block_id),
        )
        cur = conn.execute(
            "DELETE FROM training_block WHERE user_id = %s AND id = %s",
            (user_id, block_id),
        )
        if cur.rowcount:
            conn.execute(
                "UPDATE training_plan SET block_id = NULL, updated_at = %s "
                "WHERE user_id = %s AND block_id = %s",
                (datetime.now(timezone.utc).isoformat(), user_id, block_id),
            )
        return cur.rowcount > 0

    @staticmethod
    def _check_exists(conn: Any, table: str, user_id: int, row_id: int) -> None:
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE user_id = %s AND id = %s",
            (user_id, row_id),
        ).fetchone()
        if row is None:
            raise LookupError(f"{table} id {row_id} not found")


def _build_set(fields: dict[str, Any], now: str) -> tuple[str, list[Any]]:
    """Build a ``col = %s, ...`` SET clause + params for a partial update."""
    assigns = ", ".join(f"{key} = %s" for key in fields) + ", updated_at = %s"
    params = [fields[key] for key in fields] + [now]
    return assigns, params


class TrainingAnchor:
    """Per-user facade over ``TrainingGoalStore`` (mirrors ``TrainingPlan``).

    Only what the agent tools call is exposed — the REST layer talks to the
    store directly, and the tools go through ``resolve_goal`` for anything
    date-resolved.
    """

    def __init__(self, store: TrainingGoalStore, user_id: int) -> None:
        self._store = store
        self._user_id = user_id

    def list_goals(self) -> list[dict[str, Any]]:
        return self._store.list_goals(self._user_id)

    def resolve_goal(
        self, goal_id: int, day: str | None = None
    ) -> dict[str, Any] | None:
        return self._store.resolve_goal(self._user_id, goal_id, day)

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._store.apply_spec(self._user_id, spec)
