"""``TrainingAnchorStore``: the account's goal and its weekly targets (RLS-scoped)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from ...db import open_pg_pool
from .training_validation import (
    _goal_row,
    _normalize_goal,
    _normalize_week,
    _week_row,
)
from .training_rules import (
    _change_pct,
    _coverage_entry,
    _first_monday,
    _season_week,
    goal_activities,
    goal_workouts,
)

#: The goal's own fields; an anchor edit carrying any of them patches (or
#: creates) the goal.
_GOAL_KEYS = (
    "title", "sport", "start_date", "target_date", "target_distance_km", "target_time",
)


def _by_start(weeks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every stored week target, keyed by its Monday."""
    return {w["week_start"]: w for w in weeks}


def _week_label(number: int | None, count: int | None) -> str | None:
    """A week's place in the season: "12 of 31" (or "12" without a target date)."""
    if number is None:
        return None
    return f"{number} of {count}" if count else str(number)


class TrainingAnchorStore:
    """The account's ONE goal (``training_goal``) and its calendar-week
    targets (``training_week``, one row per Monday), RLS-scoped.

    The goal describes the event the season builds to; its ``start_date`` and
    ``target_date`` number the season's weeks ("12 of 31"). Week targets stand
    on their own — a ``duration_min`` target and an optional deload flag per
    ``week_start`` — so a week never depends on the goal or on any phase, and
    deleting the goal leaves the dated week targets and workouts in place.
    Edits are partial (PATCH semantics) and run inside the caller's
    transaction (``TrainingStore.apply``).
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

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()

    # -- reads -----------------------------------------------------------------

    def get_goal(self, user_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._get_goal(conn, user_id)

    @staticmethod
    def _get_goal(conn: Any, user_id: int) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s", (user_id,)
        ).fetchone()
        return _goal_row(row) if row else None

    def _read(self, user_id: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The goal (or None) and every stored week target, oldest first."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            goal = self._get_goal(conn, user_id)
            rows = conn.execute(
                "SELECT * FROM training_week WHERE user_id = %s ORDER BY week_start",
                (user_id,),
            ).fetchall()
        return goal, [_week_row(r) for r in rows]

    def resolve(self, user_id: int, day: str | None = None) -> dict[str, Any]:
        """The plan tab's view of the season.

        ``goal`` (or None), every stored week target with its ``change_pct``
        (vs the last build week, see ``_change_pct``) and its ``week_number`` in the season (None
        outside it), and ``weekly_target``: the calendar week holding
        ``day`` (default today) as ``{week_start, week_number, week_count,
        week, coverage}`` — ``week`` is that week's stored target and
        ``coverage`` its target vs planned/synced volume (both None when the
        week has no target)."""
        goal, weeks = self._read(user_id)
        monday = _first_monday(date.fromisoformat(day) if day else date.today())
        rows = self._rows(user_id, goal, weeks, monday, monday)
        number, count = _season_week(goal, monday)
        by_start = _by_start(weeks)
        return {
            "goal": goal,
            "weeks": [
                {
                    **w,
                    "change_pct": _change_pct(by_start, w["week_start"]),
                    "week_number": _season_week(goal, date.fromisoformat(w["week_start"]))[0],
                }
                for w in weeks
            ],
            "weekly_target": {
                "week_start": monday.isoformat(),
                "week_number": number,
                "week_count": count,
                "week": rows[0]["week"] if rows else None,
                "coverage": rows[0]["coverage"] if rows else None,
            },
        }

    def coverage_range(
        self, user_id: int, date_start: str | None = None, date_end: str | None = None,
    ) -> list[dict[str, Any]]:
        """One flat entry per targeted calendar week touched by the window
        (every stored week when a bound is None): ``week_start``, ``week``
        (its place in the season, "12 of 31") and the :func:`_coverage_entry`
        fields. Batched (one read, two volume queries), so a multi-week agenda
        stays cheap."""
        goal, weeks = self._read(user_id)
        return [
            {
                "week_start": row["week_start"],
                "week": _week_label(row["week_number"], row["week_count"]),
                **row["coverage"],
            }
            for row in self._rows(
                user_id, goal, weeks,
                date.fromisoformat(date_start) if date_start else None,
                date.fromisoformat(date_end) if date_end else None,
            )
        ]

    def _rows(
        self, user_id: int, goal: dict[str, Any] | None,
        weeks: list[dict[str, Any]], start: date | None, end: date | None,
    ) -> list[dict[str, Any]]:
        """``{week_start, week, week_number, week_count, coverage}`` for every
        stored week target whose week touches ``start``..``end`` (a None bound
        is open): the single week-resolution path behind the plan tab, the
        agent's agenda and an edit's result."""
        lo = _first_monday(start).isoformat() if start else ""
        hi = _first_monday(end).isoformat() if end else "9999"
        targeted = [w for w in weeks if lo <= w["week_start"] <= hi]
        if not targeted:
            return []
        planned, actual = self._weekly_volumes(
            user_id, goal,
            date.fromisoformat(targeted[0]["week_start"]),
            date.fromisoformat(targeted[-1]["week_start"]) + timedelta(days=6),
        )
        zero = {"km": 0.0, "min": 0.0}
        by_start = _by_start(weeks)
        rows: list[dict[str, Any]] = []
        for week in targeted:
            ws = week["week_start"]
            number, count = _season_week(goal, date.fromisoformat(ws))
            p, a = planned.get(ws, zero), actual.get(ws, zero)
            rows.append({
                "week_start": ws,
                "week": week,
                "week_number": number,
                "week_count": count,
                "coverage": _coverage_entry(
                    week, p["km"], p["min"], a["km"], a["min"], _change_pct(by_start, ws),
                ),
            })
        return rows

    def _weekly_volumes(
        self, user_id: int, goal: dict[str, Any] | None, start: date, end: date,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Planned and actual km/min per Mon..Sun week (keyed by Monday).

        Two queries for the whole window, never one per week. Both count only
        the goal's sport (:func:`goal_workouts` / :func:`goal_activities`).
        """
        lo, hi = start.isoformat(), end.isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            workouts = conn.execute(
                "SELECT planned_date, activity_type, distance_km, duration_min "
                "FROM training_workout WHERE user_id = %s "
                "AND planned_date >= %s AND planned_date <= %s",
                (user_id, lo, hi),
            ).fetchall()
            acts = conn.execute(
                "SELECT start_date, activity_type, distance_km, duration_hours "
                "FROM activity_summaries WHERE user_id = %s "
                "AND start_date >= %s AND start_date <= %s",
                (user_id, lo, hi),
            ).fetchall()
        planned: dict[str, dict[str, float]] = {}
        for w in goal_workouts(goal, workouts):
            ws = _first_monday(date.fromisoformat(w["planned_date"])).isoformat()
            bucket = planned.setdefault(ws, {"km": 0.0, "min": 0.0})
            bucket["km"] += float(w["distance_km"] or 0)
            bucket["min"] += float(w["duration_min"] or 0)
        actual: dict[str, dict[str, float]] = {}
        for a in goal_activities(goal, acts):
            ws = _first_monday(date.fromisoformat(a["start_date"])).isoformat()
            bucket = actual.setdefault(ws, {"km": 0.0, "min": 0.0})
            bucket["km"] += float(a["distance_km"] or 0)
            bucket["min"] += float(a["duration_hours"] or 0) * 60
        return planned, actual

    # -- the edit --------------------------------------------------------------

    def apply_spec(self, conn: Any, user_id: int, spec: dict[str, Any]) -> dict[str, Any]:
        """Apply one anchor edit (the agent's ``anchor`` section and the plan
        tab's goal editor) on the caller's connection, so it commits or rolls
        back with the rest of the season edit.

        ``delete_goal`` removes the goal first. Goal fields then patch the
        goal, creating it when the account has none, and ``weeks`` patch week
        targets by date (see :meth:`_merge_weeks`). Returns whether a goal
        was deleted.
        """
        deleted = bool(spec.get("delete_goal")) and conn.execute(
            "DELETE FROM training_goal WHERE user_id = %s", (user_id,)
        ).rowcount > 0
        if any(key in spec for key in _GOAL_KEYS):
            self._upsert_goal(conn, user_id, spec)
        if "weeks" in spec:
            self._merge_weeks(conn, user_id, spec["weeks"])
        return {"deleted_goal": deleted}

    @staticmethod
    def _upsert_goal(conn: Any, user_id: int, data: dict[str, Any]) -> None:
        """Patch the goal — only the fields present in ``data`` change —
        creating it (absent fields NULL) when the account has none."""
        fields = _normalize_goal({k: data[k] for k in _GOAL_KEYS if k in data}, partial=True)
        cols = list(fields)
        conn.execute(
            f"INSERT INTO training_goal (user_id, {', '.join(cols)}) "
            f"VALUES (%s{', %s' * len(cols)}) ON CONFLICT (user_id) DO UPDATE SET "
            + ", ".join(f"{c} = EXCLUDED.{c}" for c in cols),
            (user_id, *fields.values()),
        )

    @staticmethod
    def _merge_weeks(conn: Any, user_id: int, weeks: list[dict[str, Any]]) -> None:
        """Patch week targets keyed by ``week_start`` (snapped to its Monday).

        The only week writer (UI, REST and agent all use it). Only the weeks
        sent are written, so correcting one week never touches another. A
        bare ``{week_start}`` or a null ``duration_min`` removes that week's
        target; otherwise fields sent override and fields omitted keep their
        stored value. A new week needs its ``duration_min``.
        """
        if not isinstance(weeks, list):
            raise ValueError("'weeks' must be a list")
        seen: set[str] = set()
        for week in weeks:
            if not isinstance(week, dict):
                raise ValueError("each week must be a JSON object")
            w = _normalize_week(week)
            ws = _first_monday(date.fromisoformat(w["week_start"])).isoformat()
            if ws in seen:
                raise ValueError(f"the week of {ws} is listed twice")
            seen.add(ws)
            if len(w) == 1 or ("duration_min" in w and w["duration_min"] is None):
                conn.execute(
                    "DELETE FROM training_week WHERE user_id = %s AND week_start = %s",
                    (user_id, ws),
                )
                continue
            stored = conn.execute(
                "SELECT duration_min, is_deload FROM training_week "
                "WHERE user_id = %s AND week_start = %s",
                (user_id, ws),
            ).fetchone() or {"duration_min": None, "is_deload": False}
            duration = w.get("duration_min", stored["duration_min"])
            if duration is None:
                raise ValueError(f"the week of {ws} needs a duration_min target")
            conn.execute(
                "INSERT INTO training_week (user_id, week_start, duration_min, is_deload) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, week_start) DO UPDATE SET "
                "duration_min = EXCLUDED.duration_min, is_deload = EXCLUDED.is_deload",
                (user_id, ws, duration, w.get("is_deload", stored["is_deload"])),
            )
