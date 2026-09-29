"""``TrainingWorkoutStore``: per-account planned workouts (RLS-scoped)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from ...db import open_pg_pool
from .training_rules import _activity_matches, _closest_activity
from .training_validation import (
    _build_set,
    _normalize_workout,
    _opt_int,
    _workout_row,
)


class TrainingWorkoutStore:
    """Per-account rows in the ``training_workout`` table (RLS-scoped).

    Like ``UserState``, connections come from a shared writer-role pool and set
    ``app.user_id`` per transaction, so Row-Level Security isolates every
    workout to its account.
    """

    def __init__(self, url: str | None = None, *, pool: Any | None = None) -> None:
        if pool is not None:
            self._pool = pool
            self._owns_pool = False
        else:
            self._pool = open_pg_pool(url, min_size=1, max_size=4)
            self._owns_pool = True

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
            sql = "SELECT * FROM training_workout WHERE user_id = %s"
            params: list[Any] = [user_id]
            if date_start:
                sql += " AND planned_date >= %s"
                params.append(date_start)
            if date_end:
                sql += " AND planned_date <= %s"
                params.append(date_end)
            sql += " ORDER BY planned_date, id"
            rows = conn.execute(sql, params).fetchall()
        return [_workout_row(r) for r in rows]

    def create(self, user_id: int, data: dict[str, Any]) -> dict[str, Any]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._create(conn, user_id, data)

    def _create(
        self, conn: Any, user_id: int, data: dict[str, Any]
    ) -> dict[str, Any]:
        fields = _normalize_workout(data)
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "INSERT INTO training_workout (user_id, planned_date, activity_type, "
            "title, description, duration_min, distance_km, intensity, "
            "target_pace_min_km, target_hr_zone, target_power_w, steps, status, "
            "goal_id, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s) RETURNING *",
            (
                user_id, fields["planned_date"], fields["activity_type"],
                fields.get("title"), fields.get("description"),
                fields.get("duration_min"), fields.get("distance_km"),
                fields.get("intensity"), fields.get("target_pace_min_km"),
                fields.get("target_hr_zone"), fields.get("target_power_w"),
                fields.get("steps"), fields.get("status") or "planned",
                fields.get("goal_id"), now, now,
            ),
        ).fetchone()
        return _workout_row(row)

    def update(
        self, user_id: int, workout_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """PATCH a workout: only the supplied fields change (the UI's edit
        modal and drag-and-drop, and the agent's upsert), so changing one field
        never clears the others. None when the id is unknown."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._update(conn, user_id, workout_id, data)

    def _update(
        self, conn: Any, user_id: int, workout_id: int, data: dict[str, Any],
    ) -> dict[str, Any] | None:
        fields = _normalize_workout(data, partial=True)
        if not fields:
            row = conn.execute(
                "SELECT * FROM training_workout WHERE user_id = %s AND id = %s",
                (user_id, workout_id),
            ).fetchone()
        else:
            assigns, params = _build_set(
                fields, datetime.now(timezone.utc).isoformat()
            )
            row = conn.execute(
                f"UPDATE training_workout SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, workout_id),
            ).fetchone()
        return _workout_row(row) if row else None

    def _delete(self, conn: Any, user_id: int, workout_id: int) -> bool:
        cur = conn.execute(
            "DELETE FROM training_workout WHERE user_id = %s AND id = %s",
            (user_id, workout_id),
        )
        return cur.rowcount > 0

    def apply(
        self, user_id: int, spec: dict[str, Any], *, conn: Any | None = None
    ) -> dict[str, Any]:
        """Apply a batch workout edit (the agent's ``training`` workouts section).

        ``spec`` keys (both optional):
        - ``workouts`` (list): a dict with an ``id`` PATCHES that workout (only
          the supplied fields change); one without an ``id`` creates a workout.
        - ``delete_ids`` (list[int]): delete those workouts.

        The whole batch runs on ONE connection, so a bad row rolls the entire
        edit back. The caller (``TrainingStore``) snapshots the whole season
        before a deletion, restorable with ``{"undo": true}``. Returns the
        counts plus the ids that changed (``added_ids``/``updated_ids``) so the
        caller never has to re-read to learn what it touched.
        """
        if conn is not None:
            return self._apply(conn, user_id, spec)
        with self._pool.connection() as own:
            self._set_user(own, user_id)
            return self._apply(own, user_id, spec)

    def _apply(
        self, conn: Any, user_id: int, spec: dict[str, Any]
    ) -> dict[str, Any]:
        deleted = added = updated = 0
        added_ids: list[int] = []
        updated_ids: list[int] = []
        for wid in spec.get("delete_ids") or []:
            wid = _opt_int(wid, "delete_ids")
            if wid is None:
                raise ValueError("delete_ids must contain integer workout ids")
            if self._delete(conn, user_id, wid):
                deleted += 1
        for workout in spec.get("workouts") or []:
            if not isinstance(workout, dict):
                raise ValueError("each workout must be a JSON object")
            wid = workout.get("id")
            if wid is not None:
                wid = _opt_int(wid, "workout id")
                if wid is None:
                    raise ValueError("workout id must be an integer")
                if self._update(conn, user_id, wid, workout) is None:
                    raise ValueError(f"workout id {wid} not found")
                updated += 1
                updated_ids.append(wid)
                continue
            row = self._create(conn, user_id, workout)
            added += 1
            added_ids.append(row["id"])
        total = conn.execute(
            "SELECT count(*) AS n FROM training_workout WHERE user_id = %s",
            (user_id,),
        ).fetchone()["n"]
        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "total": total,
            "added_ids": added_ids,
            "updated_ids": updated_ids,
        }

    def activities(
        self,
        user_id: int,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Synced activities whose ``start_date`` falls in the inclusive range.

        The unplanned counterpart to ``list``: lets the agent see what was
        actually done across a whole multi-week window, not just one week.
        """
        sql = (
            "SELECT activity_id, start_date, activity_type, distance_km, "
            "duration_hours FROM activity_summaries WHERE user_id = %s"
        )
        params: list[Any] = [user_id]
        if date_start:
            sql += " AND start_date >= %s"
            params.append(date_start)
        if date_end:
            sql += " AND start_date <= %s"
            params.append(date_end)
        sql += " ORDER BY start_date"
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            rows = conn.execute(sql, params).fetchall()
        return [dict(a) for a in rows]

    def autocomplete(self, user_id: int) -> dict[str, int]:
        """Mark planned workouts complete by matching synced activities.

        Best-effort and idempotent: only moves a workout from planned to
        completed, never un-completes and never touches a skipped one. A
        non-rest workout is matched to an activity of the same family on the
        planned day, then on the adjacent days (±1) — a long run logged the next
        morning still counts. Each activity completes at most one workout, and
        same-day matches take priority over the adjacent-day fallback (so an
        activity is never claimed by two workouts, nor borrowed from a day whose
        own workout still needs it). A past rest workout is completed when no
        activity exists that day. Dead ``completed_activity_id`` links (an
        activity removed by a re-parse) are cleared first. Future workouts are
        never touched. Returns {"completed": n}.
        """
        today = date.today().isoformat()
        now = datetime.now(timezone.utc).isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "UPDATE training_workout SET completed_activity_id = NULL, "
                "updated_at = %s WHERE user_id = %s "
                "AND completed_activity_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM activity_summaries a "
                "WHERE a.user_id = %s "
                "AND a.activity_id = training_workout.completed_activity_id)",
                (now, user_id, user_id),
            )
            workouts = conn.execute(
                "SELECT * FROM training_workout WHERE user_id = %s "
                "AND (status IS NULL OR status = 'planned') "
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
            # Each synced activity may complete at most one workout across the
            # whole season, so seed ``used`` with the links already stored by
            # earlier autocomplete runs — otherwise a planned workout could be
            # satisfied by re-borrowing an activity that already completed
            # another one (e.g. today's run stealing yesterday's run). Exact
            # planned days are matched first across all workouts, then the
            # ±1-day fallback fills the still-unmatched ones, so an adjacent-day
            # borrow never steals an activity from the workout on its own day.
            used: set[Any] = {
                row["completed_activity_id"]
                for row in conn.execute(
                    "SELECT completed_activity_id FROM training_workout "
                    "WHERE user_id = %s AND completed_activity_id IS NOT NULL",
                    (user_id,),
                ).fetchall()
            }
            matched: set[int] = set()
            pending: list[Any] = []
            for w in workouts:
                if w["planned_date"] > today:
                    continue
                day_acts = by_date.get(w["planned_date"], [])
                if w["activity_type"] == "rest":
                    if w["planned_date"] < today and not day_acts:
                        updates.append((None, now, user_id, w["id"]))
                    continue
                pending.append(w)

            def _claim(workout: Any, day_acts: list[dict[str, Any]]) -> bool:
                candidates = [
                    a for a in day_acts
                    if a["activity_id"] not in used
                    and _activity_matches(workout["activity_type"], a["activity_type"])
                ]
                best = _closest_activity(candidates, workout)
                if best is None:
                    return False
                used.add(best["activity_id"])
                matched.add(workout["id"])
                updates.append((best["activity_id"], now, user_id, workout["id"]))
                return True

            for w in pending:
                _claim(w, by_date.get(w["planned_date"], []))
            for w in pending:
                if w["id"] in matched:
                    continue
                for offset in (-1, 1):
                    other = (
                        date.fromisoformat(w["planned_date"])
                        + timedelta(days=offset)
                    ).isoformat()
                    if _claim(w, by_date.get(other, [])):
                        break
            if updates:
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE training_workout SET status = 'completed', "
                        "completed_activity_id = %s, "
                        "updated_at = %s WHERE user_id = %s AND id = %s",
                        updates,
                    )
        return {"completed": len(updates)}

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()
