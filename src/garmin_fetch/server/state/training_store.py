"""``TrainingStore``: the one transactional writer for the whole season,
its season-wide undo snapshot, and the agent-facing ``TrainingSeason``
facade."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ...db import open_pg_pool
from .training_workouts import TrainingWorkoutStore
from .training_anchor import TrainingAnchorStore


#: ``user_state`` key holding the one season undo snapshot. Reusing the existing
#: per-user key/value table keeps the snapshot in the same RLS scope and inside
#: the editing transaction, with no extra table.
_SNAPSHOT_KEY = "season_undo"


def _write_snapshot(conn: Any, user_id: int, payload: Any) -> None:
    """Store (replace) the season undo snapshot."""
    conn.execute(
        "INSERT INTO user_state (user_id, key, value, updated_at) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (user_id, key) DO UPDATE SET "
        "value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
        (
            user_id, _SNAPSHOT_KEY, json.dumps(payload, default=str),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _read_snapshot(conn: Any, user_id: int) -> Any | None:
    """Load the stored season undo snapshot (None when absent)."""
    row = conn.execute(
        "SELECT value FROM user_state WHERE user_id = %s AND key = %s",
        (user_id, _SNAPSHOT_KEY),
    ).fetchone()
    if row is None or not row["value"]:
        return None
    return json.loads(row["value"])


class TrainingStore:
    """One owner of the whole training season on a single connection pool.

    The season is one tree — goal → block → week → workout — and this store is
    the only place it is written. The workout and anchor sub-stores share this
    pool, and one season-wide snapshot spans both, so a single destructive edit
    (in either half) is restored by a single ``undo``. Every write still sets
    ``app.user_id`` per transaction, so RLS isolates accounts.
    """

    def __init__(self, url: str) -> None:
        self._pool = open_pg_pool(url, min_size=1, max_size=8)
        self.workouts = TrainingWorkoutStore(pool=self._pool)
        self.anchor = TrainingAnchorStore(pool=self._pool)

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    def close(self) -> None:
        self._pool.close()

    # -- destructive edits (one season snapshot, one undo) --------------------

    def apply(self, user_id: int, spec: dict[str, Any]) -> dict[str, Any]:
        """One season edit: optional ``anchor`` and/or ``workouts`` sections.

        This is the single write surface the agent uses. Both sections run on ONE
        connection/transaction, so a failure in either rolls the whole edit back —
        the season is never left half-reshaped. A destructive section snapshots
        the whole season once, before either section runs, so a single ``undo``
        rolls the entire edit back. ``{"undo": true}`` restores it.
        """
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object")
        if spec.get("undo"):
            with self._pool.connection() as conn:
                self._set_user(conn, user_id)
                return {"restored": self._undo(conn, user_id)}
        anchor_spec = spec.get("anchor")
        raw_workouts = spec.get("workouts")
        workouts_spec: dict[str, Any] | None = None
        if isinstance(raw_workouts, dict):
            workouts_spec = dict(raw_workouts)
            if spec.get("delete_ids") and "delete_ids" not in workouts_spec:
                workouts_spec["delete_ids"] = spec["delete_ids"]
        elif isinstance(raw_workouts, list):
            workouts_spec = {"workouts": raw_workouts}
            if spec.get("delete_ids"):
                workouts_spec["delete_ids"] = spec["delete_ids"]
        elif spec.get("delete_ids"):
            workouts_spec = {"delete_ids": spec["delete_ids"]}

        if anchor_spec is None and workouts_spec is None:
            raise ValueError(
                "spec needs an 'anchor' or 'workouts' section (or {'undo': true})"
            )
        if anchor_spec is not None and not isinstance(anchor_spec, dict):
            raise ValueError("'anchor' must be an object")
        # Destructive = removes rows or overwrites weeks of an existing block.
        blocks = (anchor_spec or {}).get("blocks") or []
        rewrites_weeks = any(
            isinstance(b, dict) and b.get("id") is not None and "weeks" in b
            for b in blocks
        )
        destructive = bool(
            (anchor_spec and (
                anchor_spec.get("delete_goal_ids")
                or anchor_spec.get("delete_blocks")
                or rewrites_weeks
            ))
            or (workouts_spec and workouts_spec.get("delete_ids"))
        )
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            before = self._capture(conn, user_id) if destructive else None
            result: dict[str, Any] = {}
            if anchor_spec is not None:
                result["anchor"] = self.anchor.apply_spec(
                    user_id, anchor_spec, conn=conn
                )
            if workouts_spec is not None:
                result["workouts"] = self.workouts.apply(
                    user_id, workouts_spec, conn=conn
                )
            # Same transaction as the edit, and only when it really removed
            # something — deleting an unknown id must not clobber the last
            # good snapshot with a no-op one.
            if before is not None and (
                anchor_spec and (
                    anchor_spec.get("delete_blocks") or rewrites_weeks
                )
                or result.get("anchor", {}).get("deleted_goals")
                or result.get("workouts", {}).get("deleted")
            ):
                _write_snapshot(conn, user_id, before)
        return result

    # -- one season-wide snapshot / undo --------------------------------------

    @staticmethod
    def _capture(conn: Any, user_id: int) -> dict[str, Any]:
        """The whole season (plan + anchor), as the undo snapshot payload."""
        plan = conn.execute(
            "SELECT * FROM training_workout WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        goals = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        blocks = conn.execute(
            "SELECT * FROM training_block WHERE user_id = %s "
            "ORDER BY goal_id, start_date, id",
            (user_id,),
        ).fetchall()
        weeks = conn.execute(
            "SELECT * FROM training_week WHERE user_id = %s "
            "ORDER BY block_id, week_start",
            (user_id,),
        ).fetchall()
        return {
            "workouts": [dict(r) for r in plan],
            "goals": [dict(r) for r in goals],
            "blocks": [dict(r) for r in blocks],
            "weeks": [dict(r) for r in weeks],
        }

    def can_undo(self, user_id: int) -> bool:
        """True when a season snapshot exists (so ``undo`` would do something)."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return _read_snapshot(conn, user_id) is not None

    def undo(self, user_id: int) -> dict[str, int]:
        """Restore the whole season from the last snapshot (original ids kept).

        Plan rows carry their own goal/block ids, so restoring the tree in
        dependency-free order (goal, block, week, plan) re-links everything.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._undo(conn, user_id)

    def _undo(self, conn: Any, user_id: int) -> dict[str, int]:
        payload = _read_snapshot(conn, user_id)
        if payload is None:
            raise ValueError("nothing to undo: no season snapshot stored")
        for table in (
            "training_workout", "training_week", "training_block", "training_goal"
        ):
            conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
        for g in payload.get("goals", []):
            conn.execute(
                "INSERT INTO training_goal (id, user_id, title, sport, "
                "start_date, target_date, target_distance_km, target_time, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    g.get("id"), user_id, g.get("title"), g.get("sport"),
                    g.get("start_date"), g.get("target_date"),
                    g.get("target_distance_km"), g.get("target_time"),
                    g.get("created_at"), g.get("updated_at"),
                ),
            )
        for b in payload.get("blocks", []):
            conn.execute(
                "INSERT INTO training_block (id, user_id, goal_id, name, "
                "start_date, end_date, focus, target_weekly_km, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    b.get("id"), user_id, b.get("goal_id"), b.get("name"),
                    b.get("start_date"), b.get("end_date"), b.get("focus"),
                    b.get("target_weekly_km"),
                    b.get("created_at"), b.get("updated_at"),
                ),
            )
        for w in payload.get("weeks", []):
            conn.execute(
                "INSERT INTO training_week (id, user_id, block_id, week_start, "
                "distance_km, duration_min, is_deload, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    w.get("id"), user_id, w.get("block_id"),
                    w.get("week_start"),
                    w.get("distance_km"), w.get("duration_min"),
                    w.get("is_deload"),
                    w.get("created_at"), w.get("updated_at"),
                ),
            )
        for w in payload.get("workouts", []):
            conn.execute(
                "INSERT INTO training_workout (id, user_id, planned_date, "
                "activity_type, title, description, duration_min, distance_km, "
                "intensity, target_pace_min_km, target_hr_zone, target_power_w, "
                "steps, status, completed_activity_id, goal_id, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s)",
                (
                    w.get("id"), user_id, w.get("planned_date"),
                    w.get("activity_type"), w.get("title"),
                    w.get("description"), w.get("duration_min"),
                    w.get("distance_km"), w.get("intensity"),
                    w.get("target_pace_min_km"), w.get("target_hr_zone"),
                    w.get("target_power_w"), w.get("steps"), w.get("status"),
                    w.get("completed_activity_id"),
                    w.get("goal_id"),
                    w.get("created_at"), w.get("updated_at"),
                ),
            )
        return {
            "workouts": len(payload.get("workouts", [])),
            "goals": len(payload.get("goals", [])),
        }


class TrainingSeason:
    """Per-user, user-scoped view of the one ``TrainingStore`` for the agent.

    The agent gets exactly one training tool; this facade exposes what that tool
    needs — read the tree and the resolved week, and apply one season spec —
    without threading a user id through ``build_agent``.
    """

    def __init__(self, store: TrainingStore, user_id: int) -> None:
        self._store = store
        self._user_id = user_id

    def list_goals(self) -> list[dict[str, Any]]:
        return self._store.anchor.list_goals(self._user_id)

    def resolve_goal(
        self, goal_id: int, day: str | None = None
    ) -> dict[str, Any] | None:
        return self._store.anchor.resolve_goal(self._user_id, goal_id, day)

    def blocks(self, goal_id: int) -> list[dict[str, Any]]:
        return self._store.anchor.list_blocks(self._user_id, goal_id)

    def coverage_range(
        self, date_start: str, date_end: str, goal_id: int | None = None
    ) -> list[dict[str, Any]]:
        return self._store.anchor.coverage_range(
            self._user_id, date_start, date_end, goal_id
        )

    def activities(
        self, date_start: str | None = None, date_end: str | None = None
    ) -> list[dict[str, Any]]:
        return self._store.workouts.activities(self._user_id, date_start, date_end)

    def list_workouts(
        self, date_start: str | None = None, date_end: str | None = None
    ) -> list[dict[str, Any]]:
        return self._store.workouts.list(self._user_id, date_start, date_end)

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._store.apply(self._user_id, spec)

    def can_undo(self) -> bool:
        return self._store.can_undo(self._user_id)
