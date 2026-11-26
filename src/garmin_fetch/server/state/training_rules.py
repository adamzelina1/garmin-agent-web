"""Pure training-plan rules: calendar-week math and goal/activity matching.

No database access — the week/block resolution and the ONE goal-attribution
rule live here so the store, the REST API and the agent cannot disagree.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


#: Garmin activity typeKeys that satisfy each plan ``activity_type`` when
#: auto-matching completed activities and when summing a goal's actual weekly
#: volume. ``other`` matches any type and is intentionally absent here.
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


def _first_monday(day: date) -> date:
    """Monday of the calendar week containing ``day`` (weeks are Mon..Sun)."""
    return day - timedelta(days=day.weekday())


def _block_week_number(block_start: date, day: date) -> int:
    """1-based calendar week (Mon..Sun) of ``day`` within a block.

    A block's first week is the calendar week containing its start date, so a
    block that starts mid-week still aligns to Mon..Sun and its week boundary is
    exactly the plan tab's calendar week. Never returns less than 1.
    """
    return max(1, (day - _first_monday(block_start)).days // 7 + 1)


def _block_week_count(block: dict[str, Any]) -> int | None:
    """Number of calendar weeks (Mon..Sun) a dated block spans, or None.

    The count comes from the block's dates, not its stored week rows, so a
    partially-populated block still reports the full span.
    """
    start, end = block.get("start_date"), block.get("end_date")
    if not start or not end:
        return None
    base = _first_monday(date.fromisoformat(start))
    last = _first_monday(date.fromisoformat(end))
    return (last - base).days // 7 + 1


def _covering_block(
    blocks: list[dict[str, Any]], start: date, end: date | None = None
) -> dict[str, Any] | None:
    """The shortest *dated* block overlapping ``start``..``end`` (default: that
    one day), or None.

    When several blocks overlap, the most specific (shortest) phase wins.
    Undated blocks never cover anything.
    """
    lo = start.isoformat()
    hi = (end or start).isoformat()
    covering = [
        b for b in blocks
        if b.get("start_date") and b.get("end_date")
        and b["start_date"] <= hi and b["end_date"] >= lo
    ]
    if not covering:
        return None
    return min(
        covering,
        key=lambda b: (
            date.fromisoformat(b["end_date"]) - date.fromisoformat(b["start_date"])
        ).days,
    )


def _effective_week(stored: dict[str, Any] | None) -> dict[str, Any] | None:
    """The target of one calendar week: its stored row (``is_deload``
    defaulted to False), or None. Week targets are minutes only."""
    if stored is None:
        return None
    week = dict(stored)
    if week.get("is_deload") is None:
        week["is_deload"] = False
    return week


def _resolve_block_weeks(
    block: dict[str, Any], raw_weeks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every calendar-week target of a block: the stored row (or an empty
    week) for each week of a dated block's span, plus any stored week outside
    that span."""
    by_start = {w["week_start"]: w for w in raw_weeks if w.get("week_start")}
    resolved: list[dict[str, Any]] = []
    count = _block_week_count(block)
    if count is not None:
        base = _first_monday(date.fromisoformat(block["start_date"]))
        for i in range(count):
            ws = (base + timedelta(days=7 * i)).isoformat()
            resolved.append(
                _effective_week(by_start.get(ws))
                or {"week_start": ws, "duration_min": None, "is_deload": False}
            )
    in_span = {w["week_start"] for w in resolved}
    for w in raw_weeks:
        if w.get("week_start") not in in_span:
            resolved.append(_effective_week(w))
    return resolved


def _minutes_by_week(
    weeks_by_block: dict[int, list[dict[str, Any]]],
) -> dict[str, int]:
    """Every stored week target of a goal (all blocks), keyed by Monday."""
    return {
        w["week_start"]: w["duration_min"]
        for weeks in weeks_by_block.values() for w in weeks
        if w.get("week_start") and w.get("duration_min")
    }


def _change_pct(minutes_by_week: dict[str, int], week_start: str) -> int | None:
    """A week's target minutes vs the previous calendar week's (whole %,
    across block boundaries); None unless both weeks have a target."""
    prev = (date.fromisoformat(week_start) - timedelta(days=7)).isoformat()
    cur, before = minutes_by_week.get(week_start), minutes_by_week.get(prev)
    if not cur or not before:
        return None
    return round((cur - before) / before * 100)


def _coverage_entry(
    week: dict[str, Any], km: Any, mins: Any,
    actual_km: Any = 0, actual_min: Any = 0, change_pct: int | None = None,
) -> dict[str, Any]:
    """One week's target vs its planned and synced volume — the shape the
    plan tab, the agent's read and an edit result all use.

    ``target_min`` is the week's ``duration_min``; ``delta_min`` is
    scheduled minus target; ``change_pct`` is the target vs the previous
    week's (see :func:`_change_pct`). ``est_km`` (the planned runs' estimated
    km) and ``actual_km`` are for display only — km is never a target.
    Zero km/actuals (an unbuilt or future week) are None. ``km``/``mins``
    are the pre-summed planned totals and ``actual_km``/``actual_min`` the
    synced-activity totals, so callers can batch the SUMs.
    """
    target = week.get("duration_min")
    scheduled = int(round(float(mins or 0)))
    return {
        "target_min": target,
        "scheduled_min": scheduled,
        "delta_min": scheduled - target if target is not None else None,
        "change_pct": change_pct,
        "deload": True if week.get("is_deload") else None,
        "est_km": round(float(km or 0), 1) or None,
        "actual_min": int(round(float(actual_min or 0))) or None,
        "actual_km": round(float(actual_km or 0), 1) or None,
    }


def _activity_matches(plan_type: str, garmin_type: str | None) -> bool:
    """True when a Garmin activity typeKey satisfies a plan ``activity_type``."""
    if plan_type == "other":
        return True
    allowed = GARMIN_TYPE_MAP.get(plan_type)
    return allowed is not None and garmin_type in allowed


def _goal_matches(sport: str | None, garmin_type: str | None) -> bool:
    """True when a Garmin typeKey counts toward a goal's volume.

    A goal with no sport (or the catch-all ``other``) counts every activity;
    otherwise the sport must map through :data:`GARMIN_TYPE_MAP`.
    """
    if not sport or sport == "other":
        return True
    return _activity_matches(sport, garmin_type)


def _workout_matches_sport(sport: str | None, activity_type: str | None) -> bool:
    """True when a workout activity_type satisfies a goal's sport.

    Both values come from :data:`ACTIVITY_TYPES` ("run", "cycle", "swim",
    "strength", "other"). A goal with no sport (or "other") matches all
    workout types, and an "other" workout matches any sport. Otherwise the
    types must match (or map via :data:`GARMIN_TYPE_MAP`).
    """
    if not sport or sport == "other":
        return True
    if not activity_type or activity_type == "other":
        return True
    if sport == activity_type:
        return True
    allowed = GARMIN_TYPE_MAP.get(sport)
    return allowed is not None and activity_type in allowed


def goal_workouts(
    goal: dict[str, Any], workouts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The planned workouts that count toward the (one) goal's weekly
    volume: those whose type fits its sport. An account has one goal, so
    there is nothing else to attribute."""
    return [
        w for w in workouts
        if _workout_matches_sport(goal.get("sport"), w.get("activity_type"))
    ]


def goal_activities(
    goal: dict[str, Any], activities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The synced activities that count toward ``goal``'s sport."""
    return [
        a for a in activities
        if _goal_matches(goal.get("sport"), a.get("activity_type"))
    ]


def _closest_activity(
    candidates: list[dict[str, Any]], workout: dict[str, Any]
) -> dict[str, Any] | None:
    """Pick the candidate nearest the planned duration (the plan's unit;
    a run's planned distance is only an estimate)."""
    if not candidates:
        return None
    if workout.get("duration_min") is not None:
        with_duration = [a for a in candidates if a.get("duration_hours") is not None]
        if with_duration:
            return min(
                with_duration,
                key=lambda a: abs(workout["duration_min"] - a["duration_hours"] * 60),
            )
    return candidates[0]
