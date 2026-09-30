"""The agent's windowed view of the season: goals, blocks and one entry per
calendar week with its targets, planned workouts and actual activities.

Pure read composition over ``TrainingAnchorStore`` / ``TrainingWorkoutStore``;
the goal/block/week resolution itself stays in ``TrainingAnchorStore``.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .training_rules import _first_monday, goal_activities, goal_workouts

#: Safety cap on the workouts returned in one agenda.
_MAX_WORKOUTS = 200

#: Description preview length when ``detail=False``: the text is kept (so no
#: second round-trip) but capped so a long description never bloats the context.
_DESC_PREVIEW_CHARS = 160

#: Bookkeeping columns the agent never needs (they only cost tokens).
_DROP = ("created_at", "updated_at")

#: Longest window (in weeks) one agenda may span.
_MAX_WEEKS = 26


def _slim(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop bookkeeping keys from one row (None passes through)."""
    if row is None:
        return None
    return {k: v for k, v in row.items() if k not in _DROP}


def _monday_iso(value: str) -> str:
    return _first_monday(date.fromisoformat(value)).isoformat()


def _block_overlaps(block: dict[str, Any], win_start: str, win_end: str) -> bool:
    """True when a block's dated range intersects the window; an undated or
    half-open block always counts, so a phase is never hidden before it is dated."""
    start, end = block.get("start_date"), block.get("end_date")
    return (not start or start <= win_end) and (not end or end >= win_start)


def _goal_header(
    anchor: Any, user_id: int, goal: dict[str, Any], win_start: str, win_end: str
) -> dict[str, Any]:
    """One goal's header plus only the blocks that overlap the window."""
    out = _slim(goal) or {}
    out["blocks"] = [
        {k: b.get(k) for k in
         ("id", "name", "focus", "start_date", "end_date", "target_weekly_km")}
        for b in anchor.list_blocks(user_id, goal["id"])
        if _block_overlaps(b, win_start, win_end)
    ]
    return out


def _full_goal(anchor: Any, user_id: int, goal_id: int, day: str) -> dict[str, Any] | None:
    """The whole resolved goal (every block/week + the current block), slimmed
    and with the weekly target flattened so the model never re-derives it."""
    resolved = anchor.resolve_goal(user_id, goal_id, day)
    if resolved is None:
        return None
    out = _slim(resolved["goal"]) or {}
    out["blocks"] = [
        {**(_slim(b) or {}), "weeks": [_slim(w) or {} for w in b.get("weeks", [])]}
        for b in resolved["blocks"]
    ]
    out["current_block"] = _slim(resolved["current_block"])
    out["week_number"] = resolved["week_number"]
    target = resolved["weekly_target"]
    out["weekly_target"] = None if target is None else {
        "block_id": target["block"]["id"],
        "block": target["block"]["name"],
        "week_number": target.get("week_number"),
        "week_count": target.get("week_count"),
        "week": _slim(target["week"]),
        "coverage": target.get("coverage"),
    }
    return out


def _window(
    day: str, weeks: int | None, date_start: str | None, date_end: str | None
) -> tuple[str, str]:
    """The inclusive agenda window: an explicit range, or ``weeks`` calendar
    weeks from Monday of ``day``'s week."""
    if (date_start is None) != (date_end is None):
        raise ValueError("pass both date_start and date_end, or neither")
    if date_start is not None:
        if weeks not in (None, 1):
            raise ValueError("pass either 'weeks' or a date_start/date_end range, not both")
        lo, hi = sorted((date.fromisoformat(date_start), date.fromisoformat(date_end)))
        return lo.isoformat(), hi.isoformat()
    span = max(1, min(int(weeks or 1), _MAX_WEEKS))
    start = _first_monday(date.fromisoformat(day))
    return start.isoformat(), (start + timedelta(days=7 * span - 1)).isoformat()


def _weeks(
    win_start: str,
    win_end: str,
    planned: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One entry per calendar week (Mon..Sun) of the window, grouping that
    week's per-goal targets, planned workouts and actual activities."""
    by_week: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def bucket(week_start: str) -> dict[str, list[dict[str, Any]]]:
        return by_week.setdefault(week_start, {"targets": [], "planned": [], "actual": []})

    for t in targets:
        bucket(t["week_start"])["targets"].append(t)
    for w in planned:
        bucket(_monday_iso(w["planned_date"]))["planned"].append(w)
    for a in actual:
        bucket(_monday_iso(a["start_date"]))["actual"].append(a)

    out: list[dict[str, Any]] = []
    cur, end = _first_monday(date.fromisoformat(win_start)), date.fromisoformat(win_end)
    while cur <= end:
        ws = cur.isoformat()
        out.append({
            "week_start": ws,
            "week_end": (cur + timedelta(days=6)).isoformat(),
            **by_week.get(ws, {"targets": [], "planned": [], "actual": []}),
        })
        cur += timedelta(days=7)
    return out


def build_agenda(
    store: Any,
    user_id: int,
    *,
    day: str | None = None,
    weeks: int | None = 1,
    date_start: str | None = None,
    date_end: str | None = None,
    goal_id: int | None = None,
    full: bool = False,
    detail: bool = False,
) -> dict[str, Any]:
    """The windowed season agenda for one user (see the agent's ``training``
    tool). ``store`` is a ``TrainingStore``. Raises ``ValueError`` on bad input."""
    anchor, workouts_store = store.anchor, store.workouts
    day = date.fromisoformat(day).isoformat() if day else date.today().isoformat()
    win_start, win_end = _window(day, weeks, date_start, date_end)

    goals = anchor.list_goals(user_id)
    if goal_id is not None:
        goals = [g for g in goals if g["id"] == int(goal_id)]
        if not goals:
            raise ValueError(f"goal {goal_id} not found")

    if full:
        out_goals = [
            g for g in (_full_goal(anchor, user_id, goal["id"], day) for goal in goals)
            if g is not None
        ]
    else:
        out_goals = [_goal_header(anchor, user_id, g, win_start, win_end) for g in goals]

    planned = workouts_store.list(user_id, win_start, win_end)
    actual = workouts_store.activities(user_id, win_start, win_end)
    if goal_id is not None:
        goal = goals[0]
        planned = goal_workouts(goal, anchor.list_blocks(user_id, goal["id"]), planned)
        actual = goal_activities(goal, actual)
    total = len(planned)
    planned = planned[:_MAX_WORKOUTS]
    if not detail:
        for w in planned:
            desc = w.get("description")
            if desc and len(desc) > _DESC_PREVIEW_CHARS:
                w["description"] = desc[:_DESC_PREVIEW_CHARS].rstrip() + "…"
    targets = anchor.coverage_range(user_id, win_start, win_end, goal_id)

    week_list = _weeks(win_start, win_end, planned, actual, targets)
    payload: dict[str, Any] = {
        "day": day,
        "window": {"from_date": win_start, "to_date": win_end, "weeks": len(week_list)},
        "can_undo": store.can_undo(user_id),
        "goals": out_goals,
        "weeks": week_list,
    }
    if total > _MAX_WORKOUTS:
        payload["workouts_truncated"] = {"total": total, "shown": _MAX_WORKOUTS}
    return payload
