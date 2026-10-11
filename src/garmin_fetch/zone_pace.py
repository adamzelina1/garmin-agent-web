"""Learned running pace per heart-rate zone (Bayesian, recomputed each sync).

Each zone's pace is modelled as a ratio to lactate-threshold pace, so the
estimate follows fitness as Garmin re-detects the threshold:

    pace(zone) = ratio(zone) * current LT pace

The prior for each ratio is Joe Friel's run pace zone midpoint (TrainingPeaks,
"Joe Friel's Quick Guide to Setting Zones"); its strength comes from the width
of Friel's range (range = +/-2 sd). The evidence is one-minute windows of the
user's recent runs (``activity_detail_series``): each window's
``pace / LT pace on that day``, grouped by the Garmin running HR zone of its
average HR and weighted by recency (half-life ``HALF_LIFE_DAYS``). One-minute
windows beat 1 km laps in a walk-forward test (23.4 vs 26.2 s/km error on
held-out laps) because a lap averages over hills and surges that a minute
resolves. The posterior mean is the usual precision-weighted average of prior
and windows, with the window noise measured from the data each sync. The
reported sd counts ``_INDEPENDENT_S`` seconds of running as one independent
observation, since neighbouring minutes of one run are correlated. Zone order
is enforced (a higher zone is never slower) by pooling adjacent violators.

Filters keep the evidence comparable: outdoor runs only (treadmill pace is
footpod-derived), no runs hotter than ``MAX_TEMP_C`` (heat slows pace at the
same HR), never a run's first ``WARMUP_S`` (HR is still climbing), and only
windows that were mostly moving (stops and walks drop out). Stored in ``derived_metrics`` as ``zone_pace_z1..z5`` (value =
decimal min/km, qualifier = JSON details), replaced wholesale with the rest.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from typing import Any

from .parser import _RUNNING_TYPES

#: Friel run pace zones as % of threshold pace -> (midpoint ratio, prior sd).
#: Z1 is open-ended in Friel ("slower than 129%"); 1.35 is our choice. Garmin
#: Z5 starts at LTHR, so it maps to Friel 5a (97-100%), widened a little since
#: Garmin Z5 also spans 5b/5c.
PRIORS: dict[int, tuple[float, float]] = {
    1: (1.35, 0.05),
    2: (1.215, 0.0375),   # 114-129%
    3: (1.095, 0.0175),   # 106-113%
    4: (1.02, 0.015),     # 99-105%
    5: (0.985, 0.02),     # 97-100% (5a)
}

HALF_LIFE_DAYS = 30
WINDOW_DAYS = 180       # older runs weigh < 2% and are skipped
MAX_TEMP_C = 22.0
SAMPLE_S = 60           # window length
WARMUP_S = 600
MIN_SPEED_KMH = 6.0     # slower samples are stops/walking
_INDEPENDENT_S = 300    # seconds of running treated as one independent observation (sd only)
_DEFAULT_SIGMA = 0.10   # window noise (ratio units) when too few windows to measure

_INDOOR_TYPES = frozenset({
    "indoor_running", "treadmill_running", "virtual_run", "running_treadmill",
})
_OUTDOOR_RUNNING = tuple(sorted(_RUNNING_TYPES - _INDOOR_TYPES))

METRIC_PREFIX = "zone_pace_z"


def _has_column(conn: Any, table: str, column: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s",
        (table, column),
    ).fetchone() is not None


def _zone_floors(conn: Any) -> list[float] | None:
    """The running HR zone floors [z2_min, z3_min, z4_min, z5_min] (bpm)."""
    row = conn.execute(
        "SELECT zone2_min, zone3_min, zone4_min, zone5_min FROM hr_zones "
        "WHERE sport IN ('RUNNING', 'DEFAULT') "
        "ORDER BY sport = 'RUNNING' DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    floors = [row["zone2_min"], row["zone3_min"], row["zone4_min"], row["zone5_min"]]
    return None if any(f is None for f in floors) else [float(f) for f in floors]


def zone_of_hr(hr: float, floors: list[float]) -> int:
    """Garmin zone (1..5) of a heart rate given the zone floors."""
    return 1 + sum(1 for f in floors if hr >= f)


def _lt_pace_now(conn: Any) -> float | None:
    row = conn.execute(
        "SELECT lactate_threshold_speed_kmh AS v FROM daily_metrics "
        "WHERE lactate_threshold_speed_kmh > 0 "
        "ORDER BY calendar_date DESC LIMIT 1"
    ).fetchone()
    return 60.0 / row["v"] if row else None


def _windows(conn: Any) -> list[dict[str, Any]]:
    """One-minute windows of eligible runs: average moving pace and HR, plus
    the LT pace that applied on the run's date. A window counts only when it
    spans most of the minute and most of its samples were moving with HR."""
    return [dict(r) for r in conn.execute(
        "WITH runs AS ("
        "  SELECT a.activity_id, a.start_date, 60.0 / lt.v AS lt_pace "
        "  FROM activity_summaries a "
        "  CROSS JOIN LATERAL ("
        "    SELECT lactate_threshold_speed_kmh AS v FROM daily_metrics d "
        "    WHERE d.calendar_date <= a.start_date "
        "      AND d.lactate_threshold_speed_kmh > 0 "
        "    ORDER BY d.calendar_date DESC LIMIT 1) lt "
        "  WHERE a.activity_type = ANY(%(types)s) "
        "    AND (a.weather_temp_c IS NULL OR a.weather_temp_c <= %(max_t)s) "
        "    AND a.start_date >= %(since)s) "
        "SELECT r.start_date, r.lt_pace, "
        "  60.0 / avg(d.speed_kmh) FILTER (WHERE d.speed_kmh >= %(min_v)s) AS pace, "
        "  avg(d.heart_rate) FILTER (WHERE d.speed_kmh >= %(min_v)s) AS avg_hr "
        "FROM runs r JOIN activity_detail_series d USING (activity_id) "
        "WHERE d.elapsed_s >= %(warmup)s "
        "GROUP BY r.activity_id, r.start_date, r.lt_pace, "
        "  floor(d.elapsed_s / %(win)s) "
        "HAVING max(d.elapsed_s) - min(d.elapsed_s) >= 0.75 * %(win)s "
        "  AND count(*) FILTER (WHERE d.speed_kmh >= %(min_v)s "
        "    AND d.heart_rate IS NOT NULL) >= 0.6 * count(*)",
        {
            "types": list(_OUTDOOR_RUNNING), "max_t": MAX_TEMP_C,
            "since": date.fromordinal(date.today().toordinal() - WINDOW_DAYS).isoformat(),
            "min_v": MIN_SPEED_KMH, "warmup": WARMUP_S, "win": SAMPLE_S,
        },
    ).fetchall()]


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _sample_sigma(by_zone: dict[int, list[float]]) -> float:
    """Robust window noise: MAD of each window around its zone median, pooled
    over zones with enough windows to have a meaningful median."""
    resid = [
        abs(x - med)
        for xs in by_zone.values() if len(xs) >= 5
        for med in (_median(xs),)
        for x in xs
    ]
    if len(resid) < 10:
        return _DEFAULT_SIGMA
    return max(1.4826 * _median(resid), 0.03)


def _enforce_order(post: dict[int, dict[str, float]]) -> None:
    """Pool adjacent violators so the ratio never rises with the zone (a
    higher zone is never slower). Pools are precision-weighted; the
    correlation-adjusted ``honest`` precision pools alongside."""
    blocks: list[list[Any]] = []  # [mean, precision, honest, zones]
    for z in range(1, 6):
        blocks.append([post[z]["ratio"], post[z]["precision"], post[z]["honest"], [z]])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, p2, h2, z2 = blocks.pop()
            m1, p1, h1, z1 = blocks.pop()
            p = p1 + p2
            blocks.append([(m1 * p1 + m2 * p2) / p, p, h1 + h2, z1 + z2])
    for mean, precision, honest, zones in blocks:
        for z in zones:
            post[z].update(ratio=mean, precision=precision, honest=honest)


def compute_zone_paces(conn: Any) -> dict[str, Any] | None:
    """The learned pace per zone, or None without a threshold pace or zones.

    Returns {"lt_pace", "sigma", "zones": {z: {pace_min_km, sd_s, ratio,
    prior, minutes, eff_minutes}}} where ``minutes`` is the raw count of
    one-minute windows and ``eff_minutes`` the recency-weighted one.
    """
    if not _has_column(conn, "daily_metrics", "lactate_threshold_speed_kmh"):
        return None
    floors = _zone_floors(conn)
    lt_now = _lt_pace_now(conn)
    if floors is None or lt_now is None:
        return None
    today = date.today().toordinal()
    by_zone: dict[int, list[float]] = {z: [] for z in PRIORS}
    weights: dict[int, list[float]] = {z: [] for z in PRIORS}
    for win in _windows(conn):
        if not win["pace"] or win["avg_hr"] is None:
            continue
        z = zone_of_hr(float(win["avg_hr"]), floors)
        age = today - date.fromisoformat(win["start_date"][:10]).toordinal()
        by_zone[z].append(float(win["pace"]) / float(win["lt_pace"]))
        weights[z].append(0.5 ** (max(age, 0) / HALF_LIFE_DAYS))
    sigma = _sample_sigma(by_zone)
    per_obs = SAMPLE_S / _INDEPENDENT_S
    post: dict[int, dict[str, float]] = {}
    for z, (mu0, tau) in PRIORS.items():
        w = weights[z]
        precision = 1 / tau**2 + sum(w) / sigma**2
        ratio = (mu0 / tau**2 + sum(wi * x for wi, x in zip(w, by_zone[z])) / sigma**2) / precision
        # Full weight per window gives the best estimate (walk-forward test);
        # the reported sd uses the correlation-adjusted precision instead.
        honest = 1 / tau**2 + per_obs * sum(w) / sigma**2
        post[z] = {"ratio": ratio, "precision": precision, "honest": honest}
    _enforce_order(post)
    zones = {
        z: {
            "pace_min_km": round(p["ratio"] * lt_now, 3),
            "sd_s": round(lt_now * 60 / math.sqrt(p["honest"]), 1),
            "ratio": round(p["ratio"], 4),
            "prior": PRIORS[z][0],
            "minutes": len(by_zone[z]) * SAMPLE_S // 60,
            "eff_minutes": round(sum(weights[z]) * SAMPLE_S / 60, 1),
        }
        for z, p in post.items()
    }
    return {"lt_pace": round(lt_now, 3), "sigma": round(sigma, 4), "zones": zones}


def to_metric_rows(result: dict[str, Any] | None, fetched_at: str) -> list[dict[str, Any]]:
    """``derived_metrics`` rows (dated today), one per zone."""
    if not result:
        return []
    today = date.today().isoformat()
    return [
        {
            "calendar_date": today,
            "metric": f"{METRIC_PREFIX}{z}",
            "value": info["pace_min_km"],
            "qualifier": json.dumps({
                k: v for k, v in info.items() if k != "pace_min_km"
            } | {"lt_pace": result["lt_pace"], "sigma": result["sigma"]}),
            "fetched_at": fetched_at,
        }
        for z, info in result["zones"].items()
    ]


def read_zone_paces(conn: Any) -> dict[str, Any] | None:
    """The stored zone paces plus the HR zone bpm ranges, or None.

    ``conn`` must be RLS-bound to the user. Returns {"as_of", "lt_pace",
    "floors", "zones": {z: {pace_min_km, sd_s, ..., hr_min, hr_max}}}.
    """
    rows = conn.execute(
        "SELECT calendar_date, metric, value, qualifier FROM derived_metrics "
        "WHERE metric LIKE %s ORDER BY metric",
        (METRIC_PREFIX + "%",),
    ).fetchall()
    if not rows:
        return None
    floors = _zone_floors(conn)
    zones: dict[int, dict[str, Any]] = {}
    lt_pace = None
    for r in rows:
        z = int(r["metric"][len(METRIC_PREFIX):])
        info = json.loads(r["qualifier"] or "{}")
        lt_pace = info.pop("lt_pace", lt_pace)
        info.pop("sigma", None)
        info["pace_min_km"] = r["value"]
        if floors:
            bounds = [None, *floors, None]
            info["hr_min"], info["hr_max"] = bounds[z - 1], (
                bounds[z] - 1 if bounds[z] is not None else None
            )
        zones[z] = info
    return {
        "as_of": rows[0]["calendar_date"],
        "lt_pace": lt_pace,
        "floors": floors,
        "zones": zones,
    }


# -- planned workout totals ----------------------------------------------------

_ZONE_RE = re.compile(r"z(?:one)?\s*([1-5])(?:\s*[-–/]\s*z?(?:one)?\s*([1-5]))?", re.I)
_BPM_RE = re.compile(r"\d{2,3}")


def parse_hr_zone(text: Any, floors: list[float] | None) -> list[int]:
    """Zones named by a free-text HR target: 'Z2', 'zone 2', 'Z2-3', '2',
    or a bpm value/range ('144-159', '<150') mapped through the floors."""
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    m = _ZONE_RE.search(s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        return list(range(min(lo, hi), max(lo, hi) + 1))
    if re.fullmatch(r"[1-5]", s):
        return [int(s)]
    bpm = [int(b) for b in _BPM_RE.findall(s) if 60 <= int(b) <= 230]
    if bpm and floors:
        return [zone_of_hr(sum(bpm) / len(bpm), floors)]
    return []


def is_hr_target(text: Any) -> bool:
    """True when free text names an HR zone ('Z2', 'zone 2-3', '2') or bpm
    ('144-159', '<150') that :func:`parse_hr_zone` can resolve."""
    if parse_hr_zone(text, None):
        return True
    return any(60 <= int(b) <= 230 for b in _BPM_RE.findall(str(text or "")))


def _step_pace(step: dict[str, Any], paces: dict[str, Any] | None) -> float | None:
    """A step's pace: its explicit pace, else its HR zone's learned pace
    (a zone range like 'Z1-Z2' averages the zones)."""
    pace = step.get("target_pace_min_km")
    if pace:
        return float(pace)
    if not paces:
        return None
    zones = [
        z for z in parse_hr_zone(step.get("target_hr_zone"), paces.get("floors"))
        if z in paces["zones"]
    ]
    if not zones:
        return None
    return sum(paces["zones"][z]["pace_min_km"] for z in zones) / len(zones)


def steps_minutes(steps: list[dict[str, Any]]) -> int | None:
    """Total planned time of the steps (repeats included), whole minutes."""
    if not steps:
        return None
    return round(sum(s["duration_sec"] * (s.get("repeat") or 1) for s in steps) / 60)


def estimate_run_km(
    steps: list[dict[str, Any]], paces: dict[str, Any] | None,
) -> float | None:
    """Distance of a planned run from its steps: sum of time / pace.

    Each step is priced by its own pace or HR zone (validation guarantees one
    on every non-rest run step); ``rest`` steps cover no distance. None when a
    step can't be priced, i.e. no zone paces are stored yet.
    """
    km = 0.0
    for step in steps:
        if step.get("kind") == "rest":
            continue
        pace = _step_pace(step, paces)
        if not pace:
            return None
        km += step["duration_sec"] * (step.get("repeat") or 1) / 60.0 / pace
    return round(km, 1) if steps else None
