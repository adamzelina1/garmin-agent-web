"""Running-isolated metrics: foot-strike volume ACWR + gait (form) drift.

The general ACWR in ``workload.py`` is driven by ``training_load`` (a heart-rate
/ EPOC proxy), which is inflated by non-impact cross-training (cycling,
swimming). This module recomputes the same 7d/28d EMA ratio using only running
activities' distance, so it tracks bone / tendon / joint impact exposure that
cross-training masks.

    acute   = EMA_7d(daily running km)
    chronic = EMA_28d(daily running km)
    run_acwr = acute / chronic                 (null when chronic is too thin)

It also tracks gait degrading under fatigue as a separate, pace-normalised
signal: for each run day, cadence is compared to the user's own cadence-vs-pace
regression over the trailing ``GAIT_WINDOW_DAYS``, giving a z-score (negative =
fewer steps/min than expected at that pace = overstriding / worse mechanics).

    run_cadence_drift = (cadence - predicted)/resid_sd   (null until ≥8 prior runs)

Rest days (dates between the first and last run, up to today) count as zero
running, so breaks decay the chronic baseline naturally and a taper shows up as
a falling ratio. Like the other derived metrics these are recomputed from
``activity_summaries`` once per sync and stored in ``derived_metrics`` under the
``run_*`` metrics, so the UI and the AI agent read the same stored values.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .parser import _RUNNING_TYPES
from .workload import _ema, acwr_category

#: A running chronic baseline below this many km/day is treated as "insufficient
#: history" — the ratio is reported as null rather than an absurd spike, so a
#: runner returning from a break is not pegged at a huge ratio indefinitely.
#: Kept low (~2 km/week) so a genuine recreational runner still gets a signal.
RUN_CHRONIC_MIN_KM = 0.3

_ACUTE_SPAN = 7
_CHRONIC_SPAN = 28

#: Gait (cadence) drift: trailing window + minimum prior run-days before a
#: pace-normalised cadence residual can be produced.
GAIT_WINDOW_DAYS = 90
MIN_GAIT_SAMPLES = 8

#: Derived-metric names -> the key each value is pivoted to on the per-day row.
#: ACWR components only (these are emitted for every day of the continuous
#: running series). Gait metrics are emitted only on run days and are pivoted
#: explicitly in ``read_series``.
RUN_METRICS: dict[str, str] = {
    "run_acute_km": "run_acute_km",
    "run_chronic_km": "run_chronic_km",
}


def _daily_series(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Continuous per-day running-distance series (rest days zero).

    ``runs`` is the aggregated {calendar_date, run_km} rows. Returns an ordered
    list from the first run date to today, each day carrying ``run_km`` (0 on
    non-running days).
    """
    by_date: dict[str, float] = {r["calendar_date"]: r["run_km"] for r in runs}
    if not by_date:
        return []
    start = date.fromisoformat(min(by_date))
    end = max(date.today(), date.fromisoformat(max(by_date)))
    days: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        iso = cursor.isoformat()
        days.append({"calendar_date": iso, "run_km": by_date.get(iso, 0.0)})
        cursor += timedelta(days=1)
    return days


def compute_run_acwr(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute the running-volume ACWR series from per-day running km.

    ``runs`` is the {calendar_date, run_km} rows (one per date that had running;
    they are aggregated upstream). Returns a continuous daily series from the
    first run date to today, each day with ``run_km`` (0 on rest days),
    ``run_acute_km``, ``run_chronic_km`` and ``run_acwr`` + ``category`` (both
    null where chronic running is too thin to judge). Rows keep date order.
    """
    days = _daily_series(runs)
    acute = _ema([d["run_km"] for d in days], _ACUTE_SPAN)
    chronic = _ema([d["run_km"] for d in days], _CHRONIC_SPAN)
    for day, a, c in zip(days, acute, chronic):
        day["run_acute_km"] = round(a, 2)
        day["run_chronic_km"] = round(c, 2)
        if c >= RUN_CHRONIC_MIN_KM:
            ratio = a / c
            day["run_acwr"] = round(ratio, 2)
            day["category"] = acwr_category(ratio)
        else:
            day["run_acwr"] = None
            day["category"] = None
    return days


def to_metric_rows(day: dict[str, Any], fetched_at: str) -> list[dict[str, Any]]:
    """Convert one computed run-ACWR day into ``derived_metrics`` rows."""
    rows = [{
        "calendar_date": day["calendar_date"],
        "metric": "run_acwr",
        "value": day.get("run_acwr"),
        "qualifier": day.get("category"),
        "fetched_at": fetched_at,
    }]
    for metric, key in RUN_METRICS.items():
        rows.append({
            "calendar_date": day["calendar_date"],
            "metric": metric,
            "value": day.get(key),
            "qualifier": None,
            "fetched_at": fetched_at,
        })
    return rows


def read_series(conn: Any) -> list[dict[str, Any]]:
    """Read the stored running ACWR series from ``derived_metrics``, pivoted to
    one dict per day (same shape as ``compute_run_acwr`` output). ``conn`` must
    be RLS-bound to the user. Returns [] when nothing is stored yet.
    """
    rows = conn.execute(
        "SELECT calendar_date, metric, value, qualifier FROM derived_metrics "
        "WHERE metric LIKE 'run_%' ORDER BY calendar_date, metric"
    ).fetchall()
    days: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        cal = row["calendar_date"]
        day = days.get(cal)
        if day is None:
            day = {"calendar_date": cal}
            days[cal] = day
            order.append(cal)
        if row["metric"] == "run_acwr":
            day["run_acwr"] = row["value"]
            day["category"] = row["qualifier"]
        elif row["metric"] == "run_cadence_drift":
            day["run_cadence_drift"] = row["value"]
            day["gait_category"] = row["qualifier"]
        elif row["metric"] == "run_cadence":
            day["run_cadence"] = row["value"]
        elif row["metric"] == "run_gct_ms":
            day["run_gct_ms"] = row["value"]
        else:
            key = RUN_METRICS.get(row["metric"])
            if key is not None:
                day[key] = row["value"]
    return [days[cal] for cal in order]


def fetch_running_runs(conn: Any) -> list[dict[str, Any]]:
    """Per-day running distance from ``activity_summaries`` (RLS-bound).

    Only running-type activities contribute; distance is summed per start date.
    Returns the ordered {calendar_date, run_km} rows.
    """
    rows = conn.execute(
        "SELECT start_date, distance_km, activity_type "
        "FROM activity_summaries ORDER BY start_date"
    ).fetchall()
    by_date: dict[str, float] = {}
    for r in rows:
        if r["activity_type"] not in _RUNNING_TYPES:
            continue
        distance = r["distance_km"]
        if not isinstance(distance, (int, float)) or distance <= 0:
            continue
        d = r["start_date"]
        if d:
            by_date[d] = by_date.get(d, 0.0) + distance
    return [
        {"calendar_date": d, "run_km": round(v, 4)}
        for d, v in sorted(by_date.items())
    ]


# -- Gait (cadence) drift -----------------------------------------------------


def gait_category(drift: float | None) -> str | None:
    """Category for a pace-normalised cadence z-score (negative = slowing).

    ~±1 sd is normal run-to-run noise, so the "dropping" band starts at -1.0 and
    only a sustained/harder sag (-1.5) is called "Breaking Down".
    """
    if drift is None:
        return None
    if drift <= -1.5:
        return "Breaking Down"
    if drift <= -1.0:
        return "Form Dropping"
    if drift >= 0.6:
        return "Form Up"
    return "Stable"


def fetch_running_activities(conn: Any) -> list[dict[str, Any]]:
    """Running activities with the running-dynamics fields (RLS-bound).

    Returns the curated summary columns needed for the gait drift: cadence,
    speed (for pace), ground-contact time and distance, for running types only.
    """
    rows = conn.execute(
        "SELECT start_date, activity_type, distance_km, avg_speed_kmh, "
        "avg_cadence, avg_ground_contact_time_ms "
        "FROM activity_summaries ORDER BY start_date"
    ).fetchall()
    return [dict(r) for r in rows]


def _aggregate_gait(activities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-activity gait into a distance-weighted per-day point.

    For each day with running, keep the distance-weighted cadence, pace
    (min/km, from avg speed) and ground-contact time. Days with no usable
    speed or cadence are dropped — gait drift is impossible to pace-normalise
    without them. Returns the ordered daily rows.
    """
    by_date: dict[str, dict[str, Any]] = {}
    for r in activities:
        if r["activity_type"] not in _RUNNING_TYPES:
            continue
        d = r.get("start_date")
        speed = r.get("avg_speed_kmh")
        cadence = r.get("avg_cadence")
        distance = r.get("distance_km")
        if not d or not isinstance(speed, (int, float)) or speed <= 0:
            continue
        if not isinstance(cadence, (int, float)) or cadence <= 0:
            continue
        if not isinstance(distance, (int, float)) or distance <= 0:
            continue
        pace = 60.0 / speed
        acc = by_date.setdefault(d, {
            "calendar_date": d,
            "weight": 0.0,
            "cadence": 0.0,
            "pace": 0.0,
            "gct_ms": 0.0,
        })
        acc["cadence"] += cadence * distance
        acc["pace"] += pace * distance
        acc["gct_ms"] += (r.get("avg_ground_contact_time_ms") or 0.0) * distance
        acc["weight"] += distance
    out: list[dict[str, Any]] = []
    for d in sorted(by_date):
        acc = by_date[d]
        w = acc["weight"]
        gct_present = acc["gct_ms"] > 0
        out.append({
            "calendar_date": d,
            "cadence": round(acc["cadence"] / w, 1),
            "pace_min_km": round(acc["pace"] / w, 2),
            "gct_ms": round(acc["gct_ms"] / w, 1) if gct_present else None,
        })
    return out


def compute_cadence_drift(gait_days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add a pace-normalised cadence drift (z-score) to each run day.

    Each day's cadence is compared to a linear cadence-vs-pace regression fit on
    the user's own run days inside the trailing ``GAIT_WINDOW_DAYS`` before it
    (no future leakage). A negative value means fewer steps/min than the user
    produces at that pace — overstriding / worse mechanics. Null until there are
    ``MIN_GAIT_SAMPLES`` prior runs. Rows keep date order.
    """
    result: list[dict[str, Any]] = []
    for i, day in enumerate(gait_days):
        cutoff = date.fromisoformat(day["calendar_date"]) - timedelta(days=GAIT_WINDOW_DAYS)
        prior = [g for g in gait_days[:i] if date.fromisoformat(g["calendar_date"]) >= cutoff]
        drift = None
        if len(prior) >= MIN_GAIT_SAMPLES:
            slope, intercept, resid_sd = _cadence_fit(prior)
            if slope is not None and resid_sd is not None and resid_sd > 0:
                predicted = intercept + slope * day["pace_min_km"]
                drift = (day["cadence"] - predicted) / resid_sd
        out = dict(day)
        out["run_cadence_drift"] = round(drift, 2) if drift is not None else None
        out["gait_category"] = gait_category(drift)
        result.append(out)
    return result


def _cadence_fit(points: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None]:
    """Least-squares cadence = intercept + slope*pace over ``points``.

    ``points`` are prior run-day rows with ``cadence`` and ``pace_min_km``.
    Returns (slope, intercept, residual std-dev), or (None, None, None) when
    there is no cadence variance to score a drift against. When pace does not
    vary (no slope to fit) it falls back to mean-only: slope 0, intercept =
    mean cadence, residual sd = cadence std-dev — so a cadence sag at a constant
    pace is still detected.
    """
    xs = [p["pace_min_km"] for p in points]
    ys = [p["cadence"] for p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        sd_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
        return 0.0, mean_y, sd_y
    sxy = sum((xs[k] - mean_x) * (ys[k] - mean_y) for k in range(n))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [ys[k] - (intercept + slope * xs[k]) for k in range(n)]
    mean_r = sum(residuals) / n
    resid_sd = (sum((r - mean_r) ** 2 for r in residuals) / n) ** 0.5
    return slope, intercept, resid_sd


def to_gait_rows(day: dict[str, Any], fetched_at: str) -> list[dict[str, Any]]:
    """Convert one computed gait day into ``derived_metrics`` rows."""
    rows = [{
        "calendar_date": day["calendar_date"],
        "metric": "run_cadence_drift",
        "value": day.get("run_cadence_drift"),
        "qualifier": day.get("gait_category"),
        "fetched_at": fetched_at,
    }]
    for metric, key in (("run_cadence", "cadence"), ("run_gct_ms", "gct_ms")):
        rows.append({
            "calendar_date": day["calendar_date"],
            "metric": metric,
            "value": day.get(key),
            "qualifier": None,
            "fetched_at": fetched_at,
        })
    return rows


def compute_gait(conn: Any) -> list[dict[str, Any]]:
    """Full gait-drift series for a user-bound connection (run days only)."""
    return compute_cadence_drift(_aggregate_gait(fetch_running_activities(conn)))
