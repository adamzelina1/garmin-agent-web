"""Acute-to-Chronic Workload Ratio (ACWR) from daily training load.

ACWR compares short-term training fatigue against long-term fitness:
``acute`` is the 7-day exponential moving average of daily training load and
``chronic`` the 28-day EMA. Daily training load is the sum of the
``training_load`` field across all activities recorded on a calendar date.

    ACWR = EMA_7d(daily load) / EMA_28d(daily load)

Interpretation:
- 0.8-1.3    Sweet Spot — fitness gains with low injury risk
- >1.5       Danger — rapid workload spike; high injury/burnout risk
- <0.8       Detraining / excessive taper
- 1.3-1.5    Elevated (the gap between the published bands)

Like readiness, ACWR is fully derived: it is recomputed from
``activity_summaries`` and stored once per sync in ``derived_metrics``
(metric='acwr' plus acwr_acute_load / acwr_chronic_load / acwr_daily_load),
so the UI and the AI agent read the same stored values. Rest days (dates with
no activities between the first and last training day, and up to today) count
as zero load, which is what lets a taper show up as a falling ratio.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

ACUTE_SPAN = 7
CHRONIC_SPAN = 28

#: Derived-metric names -> the key each value is pivoted to per day.
ACWR_METRICS: dict[str, str] = {
    "acwr_daily_load": "daily_load",
    "acwr_acute_load": "acute_load",
    "acwr_chronic_load": "chronic_load",
}


def acwr_category(acwr: float) -> str:
    if acwr > 1.5:
        return "Danger"
    if acwr > 1.3:
        return "Elevated"
    if acwr >= 0.8:
        return "Sweet Spot"
    return "Detraining"


def _ema(series: list[float], span: int) -> list[float]:
    """Exponential moving average seeded with the first value (span-based)."""
    alpha = 2.0 / (span + 1.0)
    ema: float | None = None
    out: list[float] = []
    for value in series:
        ema = value if ema is None else alpha * value + (1.0 - alpha) * ema
        out.append(ema)
    return out


def compute_acwr(loads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute the ACWR series from per-day training loads.

    ``loads`` is the list of {calendar_date, daily_load} rows (one or more per
    date; they are summed per date). Returns a continuous daily series from
    the first load date to today, each day with ``daily_load`` (0 on rest
    days), ``acute_load``, ``chronic_load``, ``acwr`` and ``category`` (all
    null on days where chronic load is zero). Days keep chronological order.
    """
    by_date: dict[str, float] = {}
    for row in loads:
        cal = row.get("calendar_date")
        if not cal:
            continue
        try:
            value = float(row.get("daily_load") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        by_date[cal] = by_date.get(cal, 0.0) + value
    if not by_date:
        return []

    start = date.fromisoformat(min(by_date))
    end = max(date.today(), date.fromisoformat(max(by_date)))
    days: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        iso = cursor.isoformat()
        days.append({"calendar_date": iso, "daily_load": by_date.get(iso, 0.0)})
        cursor += timedelta(days=1)

    acute = _ema([d["daily_load"] for d in days], ACUTE_SPAN)
    chronic = _ema([d["daily_load"] for d in days], CHRONIC_SPAN)
    for day, a, c in zip(days, acute, chronic):
        day["acute_load"] = round(a, 2)
        day["chronic_load"] = round(c, 2)
        if c > 0:
            ratio = a / c
            day["acwr"] = round(ratio, 2)
            day["category"] = acwr_category(ratio)
        else:
            day["acwr"] = None
            day["category"] = None
    return days


def to_metric_rows(day: dict[str, Any], fetched_at: str) -> list[dict[str, Any]]:
    """Convert one computed ACWR day into ``derived_metrics`` rows."""
    rows = [{
        "calendar_date": day["calendar_date"],
        "metric": "acwr",
        "value": day["acwr"],
        "qualifier": day["category"],
        "fetched_at": fetched_at,
    }]
    for metric, key in ACWR_METRICS.items():
        rows.append({
            "calendar_date": day["calendar_date"],
            "metric": metric,
            "value": day.get(key),
            "qualifier": None,
            "fetched_at": fetched_at,
        })
    return rows


def read_series(conn: Any) -> list[dict[str, Any]]:
    """Read the stored ACWR series from ``derived_metrics``, pivoted to one
    dict per day (same shape as ``compute_acwr`` output). ``conn`` must be
    RLS-bound to the user. Returns [] when nothing is stored yet.
    """
    rows = conn.execute(
        "SELECT calendar_date, metric, value, qualifier FROM derived_metrics "
        "WHERE metric LIKE 'acwr%' ORDER BY calendar_date, metric"
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
        if row["metric"] == "acwr":
            day["acwr"] = row["value"]
            day["category"] = row["qualifier"]
        else:
            key = ACWR_METRICS.get(row["metric"])
            if key is not None:
                day[key] = row["value"]
    return [days[cal] for cal in order]


def fetch_activity_loads(conn: Any) -> list[dict[str, Any]]:
    """Per-day training loads from ``activity_summaries`` on an RLS-bound conn.

    Uses the activity's local start date; activities without a ``training_load``
    are skipped. Returns the ordered {calendar_date, daily_load} rows.
    """
    rows = conn.execute(
        "SELECT start_date AS calendar_date, training_load AS daily_load "
        "FROM activity_summaries WHERE training_load IS NOT NULL "
        "ORDER BY start_date"
    ).fetchall()
    return [dict(r) for r in rows]
