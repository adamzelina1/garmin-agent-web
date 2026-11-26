"""``TrainingStore``: the one transactional writer for the whole season,
its season-wide undo snapshot, and the agent-facing ``TrainingSeason``
facade."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from ...db import open_pg_pool
from .training_workouts import TrainingWorkoutStore
from .training_anchor import TrainingAnchorStore
from .training_agenda import build_agenda
from .training_rules import _first_monday


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

    The season is the goal, the week targets and the workouts, all lined up
    by date, and this store is the only place it is written. The workout and
    anchor sub-stores share this pool, and one season-wide snapshot spans
    both, so a single destructive edit (in either half) is restored by a
    single ``undo``. Every write still sets ``app.user_id`` per transaction,
    so RLS isolates accounts.
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
        """One season edit: optional ``anchor`` (the goal and its week
        targets), ``workouts`` (a list to create/patch) and ``delete_ids``.

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
        workouts = spec.get("workouts") or []
        delete_ids = spec.get("delete_ids") or []
        if anchor_spec is None and not workouts and not delete_ids:
            raise ValueError(
                "spec needs an 'anchor', 'workouts' or 'delete_ids' section "
                "(or {'undo': true})"
            )
        if anchor_spec is not None and not isinstance(anchor_spec, dict):
            raise ValueError("'anchor' must be an object")
        if not isinstance(workouts, list):
            raise ValueError("'workouts' must be a list")
        # Destructive = removes rows or overwrites week targets.
        goal_weeks = (anchor_spec or {}).get("weeks") or []
        destructive = bool(
            goal_weeks or (anchor_spec or {}).get("delete_goal") or delete_ids
        )
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            before = self._capture(conn, user_id) if destructive else None
            result: dict[str, Any] = {}
            if anchor_spec is not None:
                result["anchor"] = self.anchor.apply_spec(conn, user_id, anchor_spec)
            if workouts or delete_ids:
                result["workouts"] = self.workouts.apply(
                    user_id, {"workouts": workouts, "delete_ids": delete_ids}, conn=conn
                )
            # Same transaction as the edit, and only when it really removed
            # or rewrote something — deleting an unknown id must not clobber
            # the last good snapshot with a no-op one.
            if before is not None and (
                goal_weeks
                or result.get("anchor", {}).get("deleted_goal")
                or result.get("workouts", {}).get("deleted")
            ):
                _write_snapshot(conn, user_id, before)
        # Read after commit: every week the edit touched, with its target vs
        # the now-scheduled volume, so the caller sees a missed weekly target
        # immediately instead of summing workouts itself.
        dates = list(result.get("workouts", {}).pop("dates", []))
        dates += [
            str(w["week_start"]) for w in goal_weeks
            if isinstance(w, dict) and w.get("week_start")
        ]
        weeks = self._week_coverage(user_id, dates)
        if weeks:
            result["weeks"] = weeks
        return result

    def _week_coverage(self, user_id: int, dates: list[str]) -> list[dict[str, Any]]:
        """Compact target/scheduled/delta (minutes) plus the
        estimated km for the weeks of ``dates``."""
        mondays = {_first_monday(date.fromisoformat(d[:10])).isoformat() for d in dates}
        if not mondays:
            return []
        lo, hi = min(mondays), max(mondays)
        hi = (date.fromisoformat(hi) + timedelta(days=6)).isoformat()
        out = []
        for row in self.anchor.coverage_range(user_id, lo, hi):
            if row["week_start"] in mondays:
                out.append({
                    k: row[k] for k in (
                        "week_start", "target_min", "scheduled_min",
                        "delta_min", "change_pct", "est_km",
                    )
                })
        return out

    # -- one season-wide snapshot / undo --------------------------------------

    @staticmethod
    def _capture(conn: Any, user_id: int) -> dict[str, Any]:
        """The whole season (plan + anchor), as the undo snapshot payload."""
        plan = conn.execute(
            "SELECT * FROM training_workout WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        goal = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s", (user_id,),
        ).fetchone()
        weeks = conn.execute(
            "SELECT * FROM training_week WHERE user_id = %s ORDER BY week_start",
            (user_id,),
        ).fetchall()
        return {
            "workouts": [dict(r) for r in plan],
            "goal": dict(goal) if goal else None,
            "weeks": [dict(r) for r in weeks],
        }

    def can_undo(self, user_id: int) -> bool:
        """True when a season snapshot exists (so ``undo`` would do something)."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return _read_snapshot(conn, user_id) is not None

    def undo(self, user_id: int) -> dict[str, int]:
        """Restore the whole season from the last snapshot (original ids kept).

        The goal, week targets and workouts only line up by date, so they
        restore independently.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._undo(conn, user_id)

    def _undo(self, conn: Any, user_id: int) -> dict[str, int]:
        payload = _read_snapshot(conn, user_id)
        if payload is None:
            raise ValueError("nothing to undo: no season snapshot stored")
        for table in ("training_workout", "training_week", "training_goal"):
            conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
        if (g := payload["goal"]) is not None:
            conn.execute(
                "INSERT INTO training_goal (user_id, title, sport, start_date, "
                "target_date, target_distance_km, target_time) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id, g["title"], g["sport"], g["start_date"],
                    g["target_date"], g["target_distance_km"], g["target_time"],
                ),
            )
        for w in payload["weeks"]:
            conn.execute(
                "INSERT INTO training_week (user_id, week_start, duration_min, is_deload) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, w["week_start"], w["duration_min"], w["is_deload"]),
            )
        for w in payload["workouts"]:
            conn.execute(
                "INSERT INTO training_workout (id, user_id, planned_date, "
                "activity_type, title, description, duration_min, distance_km, "
                "intensity, target_hr_zone, steps, status, completed_activity_id, "
                "created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s)",
                (
                    w.get("id"), user_id, w.get("planned_date"),
                    w.get("activity_type"), w.get("title"),
                    w.get("description"), w.get("duration_min"),
                    w.get("distance_km"), w.get("intensity"),
                    w.get("target_hr_zone"), w.get("steps"), w.get("status"),
                    w.get("completed_activity_id"),
                    w.get("created_at"), w.get("updated_at"),
                ),
            )
        return {
            "workouts": len(payload["workouts"]),
            "weeks": len(payload["weeks"]),
            "goal": payload["goal"] is not None,
        }


class TrainingSeason:
    """Per-user view of the one ``TrainingStore`` for the agent.

    The agent gets exactly one training tool; this facade exposes what that tool
    needs — the windowed agenda and one atomic season edit — without threading
    a user id through ``build_agent``.

    ``today`` returns the user's calendar date (``ReadOnlyDB.today``): the
    server runs in UTC, so without it "this week" would follow the server's
    date.
    """

    def __init__(
        self,
        store: TrainingStore,
        user_id: int,
        *,
        today: Callable[[], date],
    ) -> None:
        self._store = store
        self._user_id = user_id
        self._today = today

    def agenda(self, **window: Any) -> dict[str, Any]:
        """See :func:`.training_agenda.build_agenda` for the arguments."""
        return build_agenda(self._store, self._user_id, today=self._today(), **window)

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._store.apply(self._user_id, spec)
