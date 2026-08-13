"""Custom training-readiness score derived from nightly HRV, RHR and sleep.

The score is recomputed once per sync (after the night's sleep/HRV/RHR have
landed in ``daily_metrics``) and stored in ``derived_metrics`` as the
``readiness`` metric plus ``readiness_*`` component rows, so the UI and the AI
agent read the same stored values — nothing is persisted beyond what the
sync pipeline recomputes from source data.

Method (28-day rolling baselines with ``min_samples=7``):

- For each day D the baseline is the trailing 28 calendar days before it
  (``[D-28, D-1]``); a metric's baseline is only usable when that window holds
  at least ``MIN_SAMPLES`` non-null readings of it.
- ``Z_HRV = (HRV_today - mu_HRV) / sigma_HRV``, clamped at ``+2.0`` max so a
  parasympathetic spike can't inflate the score.
- ``Z_RHR = (mu_RHR - RHR_today) / sigma_RHR`` (inverted: lower RHR = higher).
- ``Z_Sleep = (Sleep_today - mu_Sleep) / sigma_Sleep``.
- ``C = 0.50*Z_HRV + 0.30*Z_RHR + 0.20*Z_Sleep``.

The composite is then scaled to 0-100 against a rolling window of the user's
own recent composites instead of a fixed linear map: ``SCORE_ANCHORS``
specifies the target distribution as ``(cumulative percentile, score)``
anchor points, and each day's score is calibrated against the composites of
the trailing ``SCALE_WINDOW_DAYS`` before it (``[D-90, D-1]``). The score
spread therefore matches the requested shape relative to recent context, and
no day is ever calibrated against its own or future data (a historical score
no longer shifts when new nights are added). Calibration needs at least
``MIN_SCALE_SAMPLES`` composites in the window; scores stay monotone in C: a
better night is always a higher score.

All three of today's metrics and all three baselines must be available for a
day to receive a score, and its calibration window must hold enough prior
composites; otherwise that day's score is null (insufficient data). A zero
baseline sigma yields a z-score of 0 (no signal, no penalty).
"""

from __future__ import annotations

import math
from collections import deque
from datetime import date, timedelta
from statistics import fmean, pstdev
from typing import Any

WINDOW_DAYS = 28
MIN_SAMPLES = 7

SCALE_WINDOW_DAYS = 90
MIN_SCALE_SAMPLES = 7

HRV_Z_CLAMP = 2.0

WEIGHT_HRV = 0.50
WEIGHT_RHR = 0.30
WEIGHT_SLEEP = 0.20

#: Target score distribution, as (cumulative percentile, score) anchor points.
#: ``(p, s)`` means "the score that the p-th cumulative fraction of days
#: should sit at or below is s". Monotone increasing in both fields. Because
#: scores are monotone in the composite, a band edge at an anchor percentile
#: is exact on the user's history: with the defaults below, 15% of days land
#: at/below 40 (Depleted), 50% at/below 60 (so ~35% Low), 90% at/below 80
#: (so ~40% Moderate) and the top ~10% are Prime. Adjust these to reshape the
#: distribution — the scale follows automatically.
SCORE_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.00, 0.0),
    (0.15, 40.0),
    (0.50, 60.0),
    (0.90, 80.0),
    (1.00, 100.0),
)


def _baseline(samples: Any) -> tuple[float | None, float | None, int]:
    """(mean, population stdev, count) for a window, or (None, None, n) when
    fewer than ``MIN_SAMPLES`` numeric readings are present."""
    values = [v for v in samples if isinstance(v, (int, float))]
    if len(values) < MIN_SAMPLES:
        return None, None, len(values)
    return fmean(values), pstdev(values), len(values)


def _z_score(value: Any, mu: float | None, sigma: float | None) -> float | None:
    """Standard score of ``value`` against a baseline, or None when missing."""
    if value is None or mu is None or sigma is None:
        return None
    if sigma == 0:
        return 0.0
    return (value - mu) / sigma


def _category(score: float) -> str:
    if score >= 80:
        return "Prime"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Low"
    return "Depleted"


def _quantile(values: list[float], p: float) -> float:
    """Linear-interpolated quantile of a sorted list (numpy-style)."""
    n = len(values)
    if n == 1:
        return values[0]
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def _auto_cutoffs(composites: list[float]) -> list[tuple[float, float]]:
    """Turn the requested percentile anchors into (composite, score) cutoffs.

    ``composites`` is a calibration window's observed composite values; each
    anchor's percentile is evaluated against them, so the returned cutoffs are
    the composite scores that produce the requested score distribution.
    """
    values = sorted(composites)
    return [(_quantile(values, p), score) for p, score in SCORE_ANCHORS]


def _map_composite(composite: float, cutoffs: list[tuple[float, float]]) -> float:
    """Piecewise-linear map of a composite onto the anchored 0-100 scale.

    Clamps below the first anchor to its score and above the last to its
    score. Scores are monotone in the composite.
    """
    first_c, first_s = cutoffs[0]
    if composite <= first_c:
        return first_s
    last_c, last_s = cutoffs[-1]
    if composite >= last_c:
        return last_s
    for (c0, s0), (c1, s1) in zip(cutoffs, cutoffs[1:]):
        if c0 <= composite <= c1:
            if c1 == c0:
                return s0
            return s0 + (s1 - s0) * (composite - c0) / (c1 - c0)
    return last_s


def compute_readiness(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute the readiness score for every day in an ordered daily series.

    ``days`` is the ordered list of rows from ``daily_metrics`` (each with
    ``calendar_date``, and the ``hrv_last_night_avg`` / ``resting_hr`` /
    ``sleep_score`` columns when present). Returns one row per input day with
    the raw metrics, the baseline mean/std used, the per-metric z-scores, the
    sample counts, the weighted composite, and the 0-100 score + category
    (all null on days with insufficient data). Rows keep the input order.

    Scores are derived from the composite via ``SCORE_ANCHORS`` calibrated
    against the trailing ``SCALE_WINDOW_DAYS`` of composites before each day
    (never including the day itself), so the scaling is automatic, adapts to
    recent context, and no day's score depends on later data.
    """
    window: deque[dict[str, Any]] = deque()
    prelim: list[dict[str, Any]] = []
    for day in days:
        cal = day.get("calendar_date")
        try:
            day_date = date.fromisoformat(cal)
        except (TypeError, ValueError):
            continue
        cutoff = day_date - timedelta(days=WINDOW_DAYS)
        while window and window[0]["_date"] < cutoff:
            window.popleft()

        mu_hrv, sd_hrv, n_hrv = _baseline(r["hrv_last_night_avg"] for r in window)
        mu_rhr, sd_rhr, n_rhr = _baseline(r["resting_hr"] for r in window)
        mu_sleep, sd_sleep, n_sleep = _baseline(r["sleep_score"] for r in window)

        hrv = day.get("hrv_last_night_avg")
        rhr = day.get("resting_hr")
        sleep = day.get("sleep_score")

        z_hrv = _z_score(hrv, mu_hrv, sd_hrv)
        z_rhr = _z_score(rhr, mu_rhr, sd_rhr)
        z_sleep = _z_score(sleep, mu_sleep, sd_sleep)
        z_hrv_clamped = min(z_hrv, HRV_Z_CLAMP) if z_hrv is not None else None

        composite: float | None = None
        if z_hrv_clamped is not None and z_rhr is not None and z_sleep is not None:
            composite = (
                WEIGHT_HRV * z_hrv_clamped
                + WEIGHT_RHR * z_rhr
                + WEIGHT_SLEEP * z_sleep
            )

        prelim.append({
            "calendar_date": cal,
            "hrv": hrv,
            "rhr": rhr,
            "sleep_score": sleep,
            "hrv_mean": mu_hrv,
            "hrv_std": sd_hrv,
            "rhr_mean": mu_rhr,
            "rhr_std": sd_rhr,
            "sleep_mean": mu_sleep,
            "sleep_std": sd_sleep,
            "samples_hrv": n_hrv,
            "samples_rhr": n_rhr,
            "samples_sleep": n_sleep,
            "z_hrv": z_hrv_clamped,
            "z_rhr": z_rhr,
            "z_sleep": z_sleep,
            "composite": composite,
            "score": None,
            "category": None,
        })

        entry = dict(day)
        entry["_date"] = day_date
        window.append(entry)

    # Rolling calibration: each day is scored against the composites of the
    # trailing SCALE_WINDOW_DAYS before it. Days with no composite are skipped
    # entirely; days whose window holds too few composites stay unscored.
    cal_window: deque[tuple[date, float]] = deque()
    for row in prelim:
        if row["composite"] is None:
            continue
        row_date = date.fromisoformat(row["calendar_date"])
        cutoff = row_date - timedelta(days=SCALE_WINDOW_DAYS)
        while cal_window and cal_window[0][0] < cutoff:
            cal_window.popleft()
        if len(cal_window) >= MIN_SCALE_SAMPLES:
            cutoffs = _auto_cutoffs([composite for _, composite in cal_window])
            row["score"] = round(_map_composite(row["composite"], cutoffs), 1)
            row["category"] = _category(row["score"])
        cal_window.append((row_date, row["composite"]))
    return prelim


#: Derived-metric names (in ``derived_metrics.metric``) -> the key each value
#: is pivoted to on the per-day readiness row. ``readiness`` itself carries the
#: score with its category in ``qualifier``.
READINESS_METRICS: dict[str, str] = {
    "readiness_hrv": "hrv",
    "readiness_rhr": "rhr",
    "readiness_sleep": "sleep_score",
    "readiness_z_hrv": "z_hrv",
    "readiness_z_rhr": "z_rhr",
    "readiness_z_sleep": "z_sleep",
    "readiness_composite": "composite",
    "readiness_samples_hrv": "samples_hrv",
    "readiness_samples_rhr": "samples_rhr",
    "readiness_samples_sleep": "samples_sleep",
}


def to_metric_rows(day: dict[str, Any], fetched_at: str) -> list[dict[str, Any]]:
    """Convert one computed readiness day into ``derived_metrics`` rows.

    ``day`` is a scored output of ``compute_readiness``. The score + category
    become the ``readiness`` row; each component becomes its own
    ``readiness_*`` row so the agent and UI can query the full detail.
    """
    rows = [{
        "calendar_date": day["calendar_date"],
        "metric": "readiness",
        "value": day["score"],
        "qualifier": day["category"],
        "fetched_at": fetched_at,
    }]
    for metric, key in READINESS_METRICS.items():
        rows.append({
            "calendar_date": day["calendar_date"],
            "metric": metric,
            "value": day.get(key),
            "qualifier": None,
            "fetched_at": fetched_at,
        })
    return rows


def read_series(conn: Any) -> list[dict[str, Any]]:
    """Read the stored readiness series from ``derived_metrics``, pivoted to
    one dict per day (same shape as ``compute_readiness`` output). ``conn``
    must be RLS-bound to the user. Returns [] when nothing is stored yet.
    """
    rows = conn.execute(
        "SELECT calendar_date, metric, value, qualifier FROM derived_metrics "
        "WHERE metric LIKE 'readiness%' ORDER BY calendar_date, metric"
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
        if row["metric"] == "readiness":
            day["score"] = row["value"]
            day["category"] = row["qualifier"]
        else:
            key = READINESS_METRICS.get(row["metric"])
            if key is not None:
                day[key] = row["value"]
    return [days[cal] for cal in order]


def effective_cutoffs(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The calibration cutoffs applied to the most recent scored day.

    ``days`` is the stored series (``read_series`` output). Fits the anchors
    on the trailing ``SCALE_WINDOW_DAYS`` of composites before that day —
    the same rolling rule the build used — so the UI can show how the
    automatic scale was fit for today. Each entry carries the anchor
    percentile, the composite value it mapped to, and the score it produces.
    Returns [] when there is not yet enough calibration history.
    """
    scored = [d for d in days if d.get("composite") is not None]
    if not scored:
        return []
    last = scored[-1]
    window_start = date.fromisoformat(last["calendar_date"]) - timedelta(
        days=SCALE_WINDOW_DAYS
    )
    values = [
        d["composite"]
        for d in scored[:-1]
        if date.fromisoformat(d["calendar_date"]) >= window_start
    ]
    if len(values) < MIN_SCALE_SAMPLES:
        return []
    sorted_values = sorted(values)
    return [
        {
            "percentile": p,
            "composite": round(_quantile(sorted_values, p), 3),
            "score": round(score, 1),
        }
        for p, score in SCORE_ANCHORS
    ]


def fetch_daily_source(conn: Any) -> list[dict[str, Any]]:
    """Pull the readiness-relevant columns from ``daily_metrics`` on ``conn``.

    ``conn`` must be bound to a user (RLS scopes the rows). Missing columns on
    a fresh/legacy schema are skipped, so the query never fails on a table
    that has not seen a parse yet. Returns the ordered daily rows.
    """
    existing = {
        r["name"]
        for r in conn.execute(
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_name = 'daily_metrics'"
        ).fetchall()
    }
    cols = [
        c for c in ("hrv_last_night_avg", "resting_hr", "sleep_score")
        if c in existing
    ]
    if not cols:
        return []
    rows = conn.execute(
        "SELECT calendar_date, " + ", ".join(cols) + " "
        "FROM daily_metrics ORDER BY calendar_date"
    ).fetchall()
    return [dict(r) for r in rows]
