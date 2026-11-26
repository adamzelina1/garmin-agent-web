"""Pure training-plan rules: calendar-week math and goal/activity matching.

No database access — the season-week numbering and the ONE goal-attribution
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


def _season_week(goal: dict[str, Any] | None, monday: date) -> tuple[int | None, int | None]:
    """``(number, count)`` of the calendar week starting ``monday`` within the
    goal's season — the weeks from the one holding ``start_date`` to the one
    holding ``target_date``. ``number`` is None outside the season (or with no
    start date); ``count`` is None without both dates."""
    start = (goal or {}).get("start_date")
    if not start:
        return None, None
    base = _first_monday(date.fromisoformat(start))
    target = goal.get("target_date")
    count = (
        (_first_monday(date.fromisoformat(target)) - base).days // 7 + 1
        if target else None
    )
    number = (monday - base).days // 7 + 1
    if number < 1 or (count is not None and number > count):
        number = None
    return number, count if count and count > 0 else None


def _change_pct(weeks: dict[str, dict[str, Any]], week_start: str) -> int | None:
    """A week's target minutes vs the last build week before it (whole %).

    The comparison week is the nearest earlier calendar week with a target,
    stepping back over deload weeks: a week after a deload compares with the
    week before the deload (not the planned dip), and a deload week shows its
    cut against the build week before it. ``weeks`` holds the stored week
    targets by ``week_start``. None when there is no such week — a week
    without a target ends the look-back."""
    day = date.fromisoformat(week_start)
    while True:
        day -= timedelta(days=7)
        prev = weeks.get(day.isoformat())
        if prev is None or not prev["duration_min"]:
            return None
        if not prev["is_deload"]:
            break
    cur = weeks[week_start]["duration_min"]
    return round((cur - prev["duration_min"]) / prev["duration_min"] * 100)


def _coverage_entry(
    week: dict[str, Any], km: Any, mins: Any,
    actual_km: Any = 0, actual_min: Any = 0, change_pct: int | None = None,
) -> dict[str, Any]:
    """One week's target vs its planned and synced volume — the shape the
    plan tab, the agent's read and an edit result all use.

    ``target_min`` is the week's ``duration_min``; ``delta_min`` is
    scheduled minus target; ``change_pct`` is the target vs the last build
    week before it (see :func:`_change_pct`). ``est_km`` (the planned runs' estimated
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
    workout types, and an "other" workout matches any sport; otherwise the
    types must be equal.
    """
    if not sport or sport == "other":
        return True
    return not activity_type or activity_type in ("other", sport)


def goal_workouts(
    goal: dict[str, Any] | None, workouts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The planned workouts that count toward the weekly targets: those whose
    type fits the goal's sport (every workout without a goal)."""
    sport = (goal or {}).get("sport")
    return [w for w in workouts if _workout_matches_sport(sport, w.get("activity_type"))]


def goal_activities(
    goal: dict[str, Any] | None, activities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The synced activities that count toward the weekly targets (the goal's
    sport; every activity without a goal)."""
    sport = (goal or {}).get("sport")
    return [a for a in activities if _goal_matches(sport, a.get("activity_type"))]


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
