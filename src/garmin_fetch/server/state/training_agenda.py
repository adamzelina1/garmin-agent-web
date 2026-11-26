"""The agent's windowed view of the season: the goal and its blocks, one
entry per calendar week with its target, planned workouts and actual
activities.

Pure read composition over ``TrainingAnchorStore`` / ``TrainingWorkoutStore``;
the block/week resolution itself stays in ``TrainingAnchorStore``.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .training_rules import _covering_block, _first_monday

#: Safety cap on the workouts returned in one agenda.
_MAX_WORKOUTS = 200

#: Keys the agent never needs: bookkeeping, the one goal's id on every block,
#: and ``completed`` (it restates ``status``).
_DROP = ("created_at", "updated_at", "goal_id", "completed")

#: Workout summary fields that its steps already carry.
_STEP_FIELDS = ("intensity", "target_pace_min_km", "target_hr_zone", "target_power_w")


def _slim(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """One row without bookkeeping keys or empty (None) values (None passes
    through)."""
    if row is None:
        return None
    return {k: v for k, v in row.items() if k not in _DROP and v is not None}


def _workout(w: dict[str, Any]) -> dict[str, Any]:
    """A planned workout; with steps, its targets live on the steps only,
    and a completed workout points at its activity (listed in the week's
    ``actual``)."""
    out = _slim(w) or {}
    out.pop("actual", None)
    if w.get("steps"):
        for key in _STEP_FIELDS:
            out.pop(key, None)
        out["steps"] = [_slim(step) for step in w["steps"]]
    return out


def _activity(a: dict[str, Any]) -> dict[str, Any]:
    """A synced activity, in the plan's units (minutes, km to 0.1)."""
    hours, km = a.get("duration_hours"), a.get("distance_km")
    return _slim({
        "activity_id": a["activity_id"],
        "start_date": a["start_date"],
        "activity_type": a["activity_type"],
        "duration_min": round(hours * 60) if hours else None,
        "distance_km": round(km, 1) if km else None,
    }) or {}


def _monday_iso(value: str) -> str:
    return _first_monday(date.fromisoformat(value)).isoformat()


def _goal(goal: dict[str, Any], blocks: list[dict[str, Any]], today: date) -> dict[str, Any]:
    """The goal with every block of the season and which one is current."""
    current = _covering_block(blocks, today)
    return {
        **(_slim(goal) or {}),
        "blocks": [
            {k: b[k] for k in ("id", "name", "focus", "start_date", "end_date") if b.get(k)}
            for b in blocks
        ],
        "current_block_id": current["id"] if current else None,
    }


def _window(today: date, date_start: str | None, date_end: str | None) -> tuple[str, str]:
    """The inclusive agenda window: the explicit range, or this week."""
    if (date_start is None) != (date_end is None):
        raise ValueError("pass both date_start and date_end, or neither")
    if date_start is not None:
        lo, hi = sorted((date.fromisoformat(date_start), date.fromisoformat(date_end)))
        return lo.isoformat(), hi.isoformat()
    start = _first_monday(today)
    return start.isoformat(), (start + timedelta(days=6)).isoformat()


def _weeks(
    win_start: str,
    win_end: str,
    planned: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One entry per calendar week (Mon..Sun) of the window, grouping that
    week's target, planned workouts and actual activities."""
    by_target = {t.pop("week_start"): t for t in targets}
    by_week: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for w in planned:
        by_week.setdefault(_monday_iso(w["planned_date"]), {"planned": [], "actual": []})["planned"].append(w)
    for a in actual:
        by_week.setdefault(_monday_iso(a["start_date"]), {"planned": [], "actual": []})["actual"].append(a)

    out: list[dict[str, Any]] = []
    cur, end = _first_monday(date.fromisoformat(win_start)), date.fromisoformat(win_end)
    while cur <= end:
        ws = cur.isoformat()
        group = by_week.get(ws, {"planned": [], "actual": []})
        target = by_target.get(ws)
        out.append({
            "week_start": ws,
            "target": _slim(target),
            "planned": [_workout(w) for w in group["planned"]],
            "actual": [_activity(a) for a in group["actual"]],
        })
        cur += timedelta(days=7)
    return out


def build_agenda(
    store: Any,
    user_id: int,
    *,
    today: date,
    date_start: str | None = None,
    date_end: str | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """The windowed season agenda for one user (see the agent's ``training``
    tool). ``store`` is a ``TrainingStore``; ``today`` is the user's date,
    which picks the default window (this week) and the current block.
    ``full`` adds ``season``: every week target of the season, in the same
    shape as ``weeks[].target``. Raises ``ValueError`` on bad input."""
    anchor, workouts_store = store.anchor, store.workouts
    win_start, win_end = _window(today, date_start, date_end)

    goals = anchor.list_goals(user_id)
    blocks = anchor.list_blocks(user_id, goals[0]["id"]) if goals else []
    planned = workouts_store.list(user_id, win_start, win_end)
    total = len(planned)
    payload: dict[str, Any] = {
        "today": today.isoformat(),
        "window": {"from_date": win_start, "to_date": win_end},
        "can_undo": store.can_undo(user_id),
        "goal": _goal(goals[0], blocks, today) if goals else None,
        "weeks": _weeks(
            win_start, win_end, planned[:_MAX_WORKOUTS],
            workouts_store.activities(user_id, win_start, win_end),
            anchor.coverage_range(user_id, win_start, win_end),
        ),
    }
    if total > _MAX_WORKOUTS:
        payload["workouts_truncated"] = {"total": total, "shown": _MAX_WORKOUTS}
    if full and blocks:
        season = anchor.coverage_range(
            user_id, min(b["start_date"] for b in blocks), max(b["end_date"] for b in blocks),
        )
        payload["season"] = [_slim(t) for t in season]
    return payload
