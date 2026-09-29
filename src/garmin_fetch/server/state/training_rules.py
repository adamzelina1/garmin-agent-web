"""Pure training-plan rules: calendar-week math and goal/activity matching.

No database access — the week/block resolution and the ONE goal-attribution
rule live here so the store, the REST API and the agent cannot disagree.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


#: Garmin activity typeKeys that satisfy each plan ``activity_type`` when
#: auto-matching completed activities and when summing a goal's actual weekly
#: volume. ``rest`` and ``other`` are handled specially (absence / any type) and
#: are intentionally absent here.
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


def _effective_week(
    block: dict[str, Any], monday: str | None, stored: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The target for one calendar week of a block.

    The ONE place the block's ``target_weekly_km`` baseline applies. A stored
    week row wins (a missing ``distance_km`` on it falls back to the baseline).
    With no row, a dated block that spans ``monday`` and has a baseline yields a
    synthetic week; otherwise None.
    """
    baseline = block.get("target_weekly_km")
    if stored is not None:
        week = dict(stored)
        if week.get("distance_km") is None and baseline is not None:
            week["distance_km"] = baseline
        if week.get("is_deload") is None:
            week["is_deload"] = False
        return week
    if baseline is None or not monday:
        return None
    if not (block.get("start_date") and block.get("end_date")):
        return None
    first = _first_monday(date.fromisoformat(block["start_date"])).isoformat()
    last = _first_monday(date.fromisoformat(block["end_date"])).isoformat()
    if first <= monday <= last:
        return {
            "week_start": monday,
            "distance_km": baseline,
            "duration_min": None,
            "is_deload": False,
        }
    return None


def _resolve_block_weeks(
    block: dict[str, Any], raw_weeks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every calendar-week target of a block: stored row or baseline for each
    week of a dated block's span, plus any stored week outside that span."""
    by_start = {w["week_start"]: w for w in raw_weeks if w.get("week_start")}
    resolved: list[dict[str, Any]] = []
    count = _block_week_count(block)
    if count is not None:
        base = _first_monday(date.fromisoformat(block["start_date"]))
        for i in range(count):
            ws = (base + timedelta(days=7 * i)).isoformat()
            resolved.append(
                _effective_week(block, ws, by_start.get(ws))
                or {"week_start": ws, "distance_km": None,
                    "duration_min": None, "is_deload": False}
            )
    in_span = {w["week_start"] for w in resolved}
    for w in raw_weeks:
        if w.get("week_start") not in in_span:
            resolved.append(_effective_week(block, w.get("week_start"), w))
    return resolved


def _coverage_entry(
    week: dict[str, Any], week_start: str, week_end: str,
    km: Any, mins: Any, actual_km: Any = 0, actual_min: Any = 0,
) -> dict[str, Any]:
    """Scheduled-vs-target coverage for one calendar week.

    The week's target unit is km when it carries ``distance_km``, else minutes
    when it carries ``duration_min`` (else no target). ``km``/``mins`` are the
    pre-summed planned totals and ``actual_km``/``actual_min`` the pre-summed
    synced-activity totals for the week, so callers can batch the SUMs.
    """
    if week.get("distance_km") is not None:
        unit, target = "km", week["distance_km"]
    elif week.get("duration_min") is not None:
        unit, target = "min", week["duration_min"]
    else:
        unit, target = None, None
    if unit == "km":
        scheduled = round(float(km or 0), 2)
        actual = round(float(actual_km or 0), 2)
    elif unit == "min":
        scheduled = int(mins or 0)
        actual = int(actual_min or 0)
    else:
        scheduled = 0
        actual = 0
    return {
        "week_start": week_start,
        "week_end": week_end,
        "unit": unit,
        "target": target,
        "scheduled": scheduled,
        "actual": actual,
        "delta": round(scheduled - target, 2) if target is not None else None,
    }


def _activity_matches(plan_type: str, garmin_type: str | None) -> bool:
    """True when a Garmin activity typeKey satisfies a plan ``activity_type``."""
    if plan_type == "other":
        return True
    allowed = GARMIN_TYPE_MAP.get(plan_type)
    return allowed is not None and garmin_type in allowed


def _goal_matches(sport: str | None, garmin_type: str | None) -> bool:
    """True when a Garmin typeKey counts toward a goal's volume.

    A goal with no sport (or the catch-all ``other``/``rest``) counts every
    activity; otherwise the sport must map through :data:`GARMIN_TYPE_MAP`.
    """
    if not sport or sport in ("other", "rest"):
        return True
    return _activity_matches(sport, garmin_type)


def _workout_matches_sport(sport: str | None, activity_type: str | None) -> bool:
    """True when a workout activity_type satisfies a goal's sport.

    Both values come from :data:`ACTIVITY_TYPES` ("run", "cycle", "swim",
    "strength", "rest", "other"). A goal with no sport (or "other"/"rest")
    matches all workout types. A workout with "rest" or "other" belongs to
    the active phase regardless of sport. Otherwise the types must match
    (or map via :data:`GARMIN_TYPE_MAP`).
    """
    if not sport or sport in ("other", "rest"):
        return True
    if not activity_type or activity_type in ("other", "rest"):
        return True
    if sport == activity_type:
        return True
    allowed = GARMIN_TYPE_MAP.get(sport)
    return allowed is not None and activity_type in allowed


def workout_in_goal(
    goal: dict[str, Any], blocks: list[dict[str, Any]], workout: dict[str, Any]
) -> bool:
    """The ONE rule for "does this planned workout belong to this goal".

    An explicit ``goal_id`` decides. Otherwise the workout's type must fit the
    goal's sport and its ``planned_date`` must fall inside one of the goal's
    dated blocks or, failing that, the goal's own start..target window (open
    bounds match). The store's coverage numbers and the agent's ``planned``
    lists both go through this, so they cannot disagree.
    """
    gid = workout.get("goal_id")
    if gid is not None:
        return int(gid) == int(goal["id"])
    if not _workout_matches_sport(goal.get("sport"), workout.get("activity_type")):
        return False
    day = workout.get("planned_date")
    if not day:
        return False
    if any(
        b.get("start_date") and b.get("end_date")
        and b["start_date"] <= day <= b["end_date"]
        for b in blocks
    ):
        return True
    start, target = goal.get("start_date"), goal.get("target_date")
    return (not start or start <= day) and (not target or day <= target)


def goal_workouts(
    goal: dict[str, Any], blocks: list[dict[str, Any]],
    workouts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The workouts that belong to ``goal`` (see :func:`workout_in_goal`)."""
    return [w for w in workouts if workout_in_goal(goal, blocks, w)]


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
