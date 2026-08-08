"""Project scalar values from raw Garmin payloads into a wide ``daily_metrics`` row.

The raw layer stores exactly what Garmin returns; parsing is a separate step.
This module reads the raw ``metrics``/``activities`` stores and projects the
*useful scalar* values into typed rows. Intraday series (heart-rate ticks,
stress/respiration arrays, sleep levels, body-battery charge curves)
intentionally stay in the raw store — they are not collapsed here.

Design rules for every extractor:
- Returns only leaf scalars (ints/floats/str/bool), never containers.
- Missing ``None`` values are omitted (skipped at merge time).
- Values keep the shape Garmin returns; nothing is validated beyond type.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable


def _get(obj: Any, *path: str, default: Any = None) -> Any:
    """Nested key lookup through a dict path, or ``default`` on any gap."""
    for key in path:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
    return obj if obj is not None else default


def _leaf(d: dict[str, Any], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Pick present values from ``d``, keyed by snake_case output name."""
    return {
        out: value
        for out, src in pairs
        if (value := d.get(src)) is not None
    }


def _hours(seconds: Any) -> float | None:
    """Seconds as hours (4dp), or None when the source isn't a number."""
    if isinstance(seconds, (int, float)):
        return round(seconds / 3600, 4)
    return None


def _local_time(epoch_ms: Any) -> str | None:
    """Epoch-millis timestamp as a local ``HH:MM`` time, or None."""
    if isinstance(epoch_ms, (int, float)):
        return datetime.fromtimestamp(epoch_ms / 1000).strftime("%H:%M")
    return None


# --- Per-type extractors --------------------------------------------------

def parse_heart_rate(payload: dict[str, Any]) -> dict[str, Any]:
    out = _leaf(payload, [
        ("resting_hr", "restingHeartRate"),
        ("min_hr", "minHeartRate"),
        ("max_hr", "maxHeartRate"),
        ("last_7d_avg_resting_hr", "lastSevenDaysAvgRestingHeartRate"),
    ])
    # The daily payload carries each zone's lower/upper bound (bpm) and the
    # time spent in it: {"zoneNumber": 2, "min": 125, "max": 140, ...}.
    for zone in payload.get("heartRateZones") or []:
        n = zone.get("zoneNumber")
        if n is None:
            continue
        lo, hi = zone.get("min"), zone.get("max")
        if isinstance(lo, (int, float)):
            out[f"hr_zone_{n}_min"] = lo
        if isinstance(hi, (int, float)):
            out[f"hr_zone_{n}_max"] = hi
        h = _hours(zone.get("secondsInZone"))
        if h is not None:
            out[f"hr_zone_{n}_hours"] = h
    return out


def parse_steps(payload: dict[str, Any]) -> dict[str, Any]:
    """Steps arrive as a list of 15-min buckets (wrapped in ``value``)."""
    buckets = payload.get("value")
    if not isinstance(buckets, list):
        return {}
    out: dict[str, Any] = {}
    total_steps = sum(b.get("steps") or 0 for b in buckets)
    total_pushes = sum(b.get("pushes") or 0 for b in buckets)
    if total_steps:
        out["total_steps"] = total_steps
    if total_pushes:
        out["pushes"] = total_pushes
    return out


def parse_sleep(payload: dict[str, Any]) -> dict[str, Any]:
    dto = payload.get("dailySleepDTO") or {}
    out: dict[str, Any] = {}
    # Durations arrive in seconds; project them into hours.
    for src, name in (
        ("sleepTimeSeconds", "sleep_time_hours"),
        ("napTimeSeconds", "nap_time_hours"),
        ("deepSleepSeconds", "deep_sleep_hours"),
        ("lightSleepSeconds", "light_sleep_hours"),
        ("remSleepSeconds", "rem_sleep_hours"),
        ("awakeSleepSeconds", "awake_sleep_hours"),
        ("unmeasurableSleepSeconds", "unmeasurable_sleep_hours"),
    ):
        h = _hours(dto.get(src))
        if h is not None:
            out[name] = h
    out.update(_leaf(dto, [
        ("average_respiration", "averageRespirationValue"),
        ("awake_count", "awakeCount"),
        ("avg_sleep_stress", "avgSleepStress"),
    ]))
    # Sleep window as local wall-clock times (epoch millis -> HH:MM).
    start = _local_time(dto.get("sleepStartTimestampLocal"))
    if start is not None:
        out["sleep_start_local"] = start
    end = _local_time(dto.get("sleepEndTimestampLocal"))
    if end is not None:
        out["sleep_end_local"] = end
    # Top-level sleep extras (not on the DTO).
    out.update(_leaf(payload, [
        ("resting_hr", "restingHeartRate"),
        ("hrv_status", "hrvStatus"),
        ("body_battery_change", "bodyBatteryChange"),
        ("restless_moments_count", "restlessMomentsCount"),
    ]))
    # Sleep score: nested under dailySleepDTO.sleepScores.overall.
    overall = _get(dto, "sleepScores", "overall")
    if isinstance(overall, dict):
        score = overall.get("value")
        if score is not None:
            out["sleep_score"] = score
        qualifier = overall.get("qualifierKey")
        if qualifier is not None:
            out["sleep_score_qualifier"] = qualifier
    return out


def parse_hrv(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(_get(payload, "hrvSummary", default={}), [
        ("hrv_last_night_avg", "lastNightAvg"),
        ("hrv_last_night_5min_high", "lastNight5MinHigh"),
        ("hrv_status", "status"),
    ])


def parse_stress(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("avg_stress", "avgStressLevel"),
        ("max_stress", "maxStressLevel"),
    ])


def parse_respiration(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("respiration_waking_avg", "avgWakingRespirationValue"),
        ("respiration_sleep_avg", "avgSleepRespirationValue"),
        ("respiration_lowest", "lowestRespirationValue"),
        ("respiration_highest", "highestRespirationValue"),
    ])


def parse_spo2(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("spo2_avg", "averageSpO2"),
        ("spo2_lowest", "lowestSpO2"),
        ("spo2_latest", "latestSpO2"),
        ("spo2_avg_sleep", "avgSleepSpO2"),
        ("spo2_last_7d_avg", "lastSevenDaysAvgSpO2"),
    ])


def parse_rhr(payload: dict[str, Any]) -> dict[str, Any]:
    """RHR is nested under allMetrics.metricsMap.WELLNESS_RESTING_HEART_RATE."""
    metrics_map = _get(payload, "allMetrics", "metricsMap", default={})
    entries = metrics_map.get("WELLNESS_RESTING_HEART_RATE") or []
    if not entries:
        return {}
    value = entries[0].get("value") if isinstance(entries[0], dict) else None
    return {"resting_hr": value} if value is not None else {}


def parse_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """The daily summary payload — rich set of scalar health/activity values."""
    out = _leaf(payload, [
        ("total_steps", "totalSteps"),
        ("total_distance_m", "totalDistanceMeters"),
        ("total_kcal", "totalKilocalories"),
        ("active_kcal", "activeKilocalories"),
        ("bmr_kcal", "bmrKilocalories"),
        ("burned_kcal", "burnedKilocalories"),
        ("consumed_kcal", "consumedKilocalories"),
        ("remaining_kcal", "remainingKilocalories"),
        ("resting_hr", "restingHeartRate"),
        ("min_hr", "minHeartRate"),
        ("max_hr", "maxHeartRate"),
        ("last_7d_avg_resting_hr", "lastSevenDaysAvgRestingHeartRate"),
        ("avg_stress", "averageStressLevel"),
        ("max_stress", "maxStressLevel"),
        ("moderate_intensity_minutes", "moderateIntensityMinutes"),
        ("vigorous_intensity_minutes", "vigorousIntensityMinutes"),
        ("floors_ascended", "floorsAscended"),
        ("floors_descended", "floorsDescended"),
        ("body_battery_charged", "bodyBatteryChargedValue"),
        ("body_battery_drained", "bodyBatteryDrainedValue"),
        ("body_battery_highest", "bodyBatteryHighestValue"),
        ("body_battery_lowest", "bodyBatteryLowestValue"),
        ("body_battery_most_recent", "bodyBatteryMostRecentValue"),
        ("body_battery_at_wake", "bodyBatteryAtWakeTime"),
        ("average_spo2", "averageSpo2"),
        ("lowest_spo2", "lowestSpo2"),
        ("latest_spo2", "latestSpo2"),
        ("highest_respiration", "highestRespirationValue"),
        ("lowest_respiration", "lowestRespirationValue"),
        ("avg_waking_respiration", "avgWakingRespirationValue"),
    ])
    # Activity intensity buckets arrive in seconds; project into hours.
    for src, name in (
        ("highlyActiveSeconds", "highly_active_hours"),
        ("activeSeconds", "active_hours"),
        ("sedentarySeconds", "sedentary_hours"),
    ):
        h = _hours(payload.get(src))
        if h is not None:
            out[name] = h
    return out


def parse_body_battery(payload: dict[str, Any]) -> dict[str, Any]:
    """Body battery is a list of events; collapse to daily charge dynamics."""
    events = payload.get("value")
    if not isinstance(events, list):
        return {}
    impact = 0.0
    slept = False
    for ev in events:
        event = ev.get("event") or {}
        if event.get("eventType") == "SLEEP":
            slept = True
        b = event.get("bodyBatteryImpact")
        if isinstance(b, (int, float)):
            impact += b
    out: dict[str, Any] = {}
    if impact:
        out["body_battery_net_change"] = round(impact, 2)
    out["body_battery_slept"] = slept
    return out


def parse_max_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Arrives as a list wrapping a single element with a ``generic`` bucket."""
    values = payload.get("value")
    if not isinstance(values, list) or not values:
        return {}
    bucket = values[0] if isinstance(values[0], dict) else {}
    generic = bucket.get("generic") or {}
    return _leaf(generic, [
        ("vo2max", "vo2MaxValue"),
        ("vo2max_precise", "vo2MaxPreciseValue"),
        ("fitness_age", "fitnessAge"),
    ])


def parse_fitnessage(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("fitness_age", "fitnessAge"),
        ("chronological_age", "chronologicalAge"),
        ("achievable_fitness_age", "achievableFitnessAge"),
        ("previous_fitness_age", "previousFitnessAge"),
    ])


def parse_hydration(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("hydration_sweat_loss_ml", "sweatLossInML"),
    ])


def parse_intensity_minutes(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("moderate_intensity_minutes", "moderateMinutes"),
        ("vigorous_intensity_minutes", "vigorousMinutes"),
        ("weekly_moderate", "weeklyModerate"),
        ("weekly_vigorous", "weeklyVigorous"),
        ("weekly_total", "weeklyTotal"),
        ("week_goal", "weekGoal"),
        ("day_of_goal_met", "dayOfGoalMet"),
    ])


def parse_floors(payload: dict[str, Any]) -> dict[str, Any]:
    """Floors arrive as a per-interval array; we report the raw count."""
    values = payload.get("floorValuesArray")
    if not isinstance(values, list):
        return {}
    total = sum(v for v in values if isinstance(v, (int, float)))
    return {"floors": total} if total else {}


def _first_bucket(mapping: Any) -> dict[str, Any] | None:
    """First dict value in a device-keyed map, or None."""
    if not isinstance(mapping, dict):
        return None
    for bucket in mapping.values():
        if isinstance(bucket, dict):
            return bucket
    return None


def parse_training_status(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # VO2Max appears only when data exists for the day.
    vo2 = _get(payload, "mostRecentVO2Max", "generic", default={})
    out.update(_leaf(vo2, [
        ("vo2max", "vo2MaxValue"),
        ("vo2max_precise", "vo2MaxPreciseValue"),
    ]))
    # Load balance / status are keyed by device id (unknown at parse time).
    load = _first_bucket(_get(
        payload, "mostRecentTrainingLoadBalance", "metricsTrainingLoadBalanceDTOMap",
        default={},
    ))
    if load:
        out.update(_leaf(load, [
            ("training_balance_feedback", "trainingBalanceFeedbackPhrase"),
        ]))
    status = _first_bucket(_get(
        payload, "mostRecentTrainingStatus", "latestTrainingStatusData", default={},
    ))
    if status:
        out.update(_leaf(status, [
            ("training_status", "trainingStatus"),
            ("weekly_training_load", "weeklyTrainingLoad"),
            ("training_status_feedback", "trainingStatusFeedbackPhrase"),
        ]))
    return out


def parse_lactate_threshold(payload: dict[str, Any]) -> dict[str, Any]:
    """Running lactate threshold (HR/speed) plus FTP, from per-day range arrays."""
    out: dict[str, Any] = {}
    for key, column in (
        ("heart_rate", "lactate_threshold_hr"),
        ("speed", "lactate_threshold_speed"),
        ("power", "ftp_watts"),
    ):
        entries = payload.get(key)
        if isinstance(entries, dict):
            entries = [entries]
        if entries:
            value = entries[0].get("value")
            if value is not None:
                out[column] = value
    return out


#: Registry: data-type name -> extractor.
PARSERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "heart_rate": parse_heart_rate,
    "steps": parse_steps,
    "sleep": parse_sleep,
    "hrv": parse_hrv,
    "stress": parse_stress,
    "respiration": parse_respiration,
    "spo2": parse_spo2,
    "rhr": parse_rhr,
    "stats": parse_stats,
    "body_battery": parse_body_battery,
    "max_metrics": parse_max_metrics,
    "fitnessage": parse_fitnessage,
    "hydration": parse_hydration,
    "intensity_minutes": parse_intensity_minutes,
    "floors": parse_floors,
    "training_status": parse_training_status,
    "lactate_threshold": parse_lactate_threshold,
}


#: Types whose stored values persist until superseded (sparse, e.g. lactate
#: threshold only updates when a session recalculates it). Days with no data
#: keep the last known value instead of going NULL.
_FFILL_TYPES = frozenset({"lactate_threshold"})


def _forward_fill(parsed: dict[str, Any], carry: dict[str, Any]) -> dict[str, Any]:
    """Merge today's parsed values into the running carry, then return the row."""
    carry.update(parsed)
    return dict(carry)


def build_daily_rows(db: Any, types: list[str] | None = None) -> dict[str, int]:
    """Parse all stored raw metrics into ``daily_metrics``.

    Reads the raw ``metrics`` store, applies the matching parser for each
    (data_type, date) row, and merges the scalars into the daily row.
    Returns {data_type: dates_parsed}.
    """
    # Some registered types (e.g. training_readiness) are fetch-only and have
    # no projection; skip them rather than crash.
    names = [n for n in (types or list(PARSERS)) if n in PARSERS]
    counts: dict[str, int] = {}
    for name in names:
        rows = db.conn.execute(
            "SELECT calendar_date, raw_json, fetched_at FROM metrics "
            "WHERE data_type = ? ORDER BY calendar_date",
            (name,),
        ).fetchall()
        carry: dict[str, Any] = {}
        for row in rows:
            parsed = PARSERS[name](json.loads(row["raw_json"]))
            if name in _FFILL_TYPES:
                parsed = _forward_fill(parsed, carry)
            if not parsed:
                continue
            db.merge_daily(row["calendar_date"], parsed, row["fetched_at"])
            counts[name] = counts.get(name, 0) + 1
    return counts


# --- Activity summaries -------------------------------------------------------
#
# Activities are stored atomically in ``activities`` (summary + details). The
# parsed projection is one row per ``activityId`` in ``activity_summaries``,
# holding only the curated, shared summary fields (durations, HR, time in
# zones, distance, calories, intensity). Fields that a given activity type
# lacks (e.g. distance on indoor cycling) are stored as NULL.
#
# Durations are stored in **hours**, distances in **km**, HR time in zones in
# **hours** (converted from the summary's seconds). The details payload is
# intentionally not parsed here.


def parse_activity_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the flat summary payload into the curated shared columns."""
    out = _leaf(payload, [
        ("activity_name", "activityName"),
        ("start_time_local", "startTimeLocal"),
        ("start_time_gmt", "startTimeGMT"),
        ("distance_km", "distance"),
        ("avg_hr", "averageHR"),
        ("max_hr", "maxHR"),
        ("calories", "calories"),
        ("training_load", "activityTrainingLoad"),
        ("aerobic_training_effect", "aerobicTrainingEffect"),
        ("anaerobic_training_effect", "anaerobicTrainingEffect"),
        ("moderate_intensity_minutes", "moderateIntensityMinutes"),
        ("vigorous_intensity_minutes", "vigorousIntensityMinutes"),
        ("elevation_gain_m", "elevationGain"),
        ("elevation_loss_m", "elevationLoss"),
        ("min_elevation_m", "minElevation"),
        ("max_elevation_m", "maxElevation"),
        ("avg_speed_kmh", "averageSpeed"),
        ("max_speed_kmh", "maxSpeed"),
        ("min_respiration_rate", "minRespirationRate"),
        ("avg_respiration_rate", "avgRespirationRate"),
        ("max_respiration_rate", "maxRespirationRate"),
        ("body_battery_change", "differenceBodyBattery"),
        ("water_estimated_ml", "waterEstimated"),
        ("is_pr", "isPR"),
    ])
    # activityType is a dict; pull the typeKey directly.
    activity_type = _get(payload, "activityType", "typeKey")
    if activity_type:
        out["activity_type"] = activity_type
    # Durations: summary gives seconds -> hours.
    for src, key in (
        ("duration", "duration_hours"),
        ("elapsedDuration", "elapsed_hours"),
        ("movingDuration", "moving_hours"),
    ):
        h = _hours(payload.get(src))
        if h is not None:
            out[key] = h
    # Distance is metres -> km; average/max speed is m/s -> km/h.
    if isinstance(out.get("distance_km"), (int, float)):
        out["distance_km"] = round(out["distance_km"] / 1000, 4)
    if isinstance(out.get("avg_speed_kmh"), (int, float)):
        out["avg_speed_kmh"] = round(out["avg_speed_kmh"] * 3.6, 2)
    if isinstance(out.get("max_speed_kmh"), (int, float)):
        out["max_speed_kmh"] = round(out["max_speed_kmh"] * 3.6, 2)
    # Heart-rate zones 1..5: each as a percentage of total activity duration.
    duration_s = payload.get("duration")
    if isinstance(duration_s, (int, float)) and duration_s > 0:
        for zone in range(1, 6):
            secs = payload.get(f"hrTimeInZone_{zone}")
            if isinstance(secs, (int, float)) and secs > 0:
                out[f"hr_time_zone_{zone}_pct"] = round(secs / duration_s * 100, 1)
    # start_date: local start date (YYYY-MM-DD).
    start_date = str(out.get("start_time_local") or "")[:10]
    if start_date:
        out["start_date"] = start_date
    return out


def build_activity_summaries(db: Any) -> dict[str, int]:
    """Parse stored activity summaries into ``activity_summaries``.

    Reads the raw ``activities`` store and projects each summary into one
    curated row keyed by ``activityId``. Returns {"activities": rows_parsed}.
    """
    rows = db.conn.execute(
        "SELECT activity_id, raw_json, fetched_at FROM activities"
    ).fetchall()
    count = 0
    for row in rows:
        parsed = parse_activity_summary(json.loads(row["raw_json"]))
        if not parsed:
            continue
        db.upsert_activity_summary(row["activity_id"], parsed, row["fetched_at"])
        count += 1
    return {"activities": count}


# --- Activity detail series --------------------------------------------------
#
# The per-activity details payload holds the intra-activity time series. Each
# tick row is a ``{"metrics": [...]}`` array aligned positionally with
# ``metricDescriptors``; each descriptor names one series (``directTimestamp``,
# ``directHeartRate``, ``sumDistance``, ...). The series is projected into a
# wide, fixed ``activity_detail_series`` table (one row per tick; metrics a
# sport's device doesn't record are NULL) plus a few scalar aggregates - avg/max
# cadence and power - merged into the curated activity summary row.


#: Cadence descriptors in order of preference - Garmin uses a different call
#: per sport (run vs swim vs cycle). ``directDoubleCadence`` streams 2x the
#: real cadence (spm), so its values are halved; ``directFractionalCadence`` is
#: a last-resort leftover only used when nothing else exists.
_CADENCE_DESCRIPTORS = (
    ("directRunCadence", 1.0),
    ("directSwimCadence", 1.0),
    ("directCyclingCadence", 1.0),
    ("directDoubleCadence", 0.5),
    ("directFractionalCadence", 1.0),
)

#: Series descriptor -> projected column name (wide table columns).
_SERIES_COLUMNS = {
    "directTimestamp": "ts_ms",
    "directHeartRate": "heart_rate",
    "directPower": "power_w",
    "directSpeed": "speed_mps",
    "directElevation": "elevation_m",
    "sumDistance": "distance_m",
    "directLatitude": "latitude",
    "directLongitude": "longitude",
}


def _pick_cadence(descriptors: list[dict[str, Any]]) -> tuple[int | None, float]:
    """Return (metrics index, value scale) of the best cadence, or (None, 1)."""
    present = {(desc or {}).get("key") for desc in descriptors}
    for key, scale in _CADENCE_DESCRIPTORS:
        if key in present:
            return _descriptor_index(descriptors, key), scale
    return None, 1.0


def _descriptor_index(descriptors: list[dict[str, Any]], key: str) -> int | None:
    for i, desc in enumerate(descriptors):
        if (desc or {}).get("key") == key:
            return i
    return None


def _series_index(details: dict[str, Any]) -> dict[str, int]:
    """Map each series column name to its descriptor index."""
    columns: dict[str, int] = {}
    for i, desc in enumerate(details.get("metricDescriptors") or []):
        key = (desc or {}).get("key")
        if key in _SERIES_COLUMNS:
            columns[_SERIES_COLUMNS[key]] = i
    return columns


def parse_activity_details(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the details payload into scalar aggregates.

    Reads the intra-activity series and reduces it to the handful of scalar
    values that belong on the curated summary row: average/max cadence and
    average/max power (watts). HR avg/max already come from the flat summary.
    Never raises on missing descriptors or empty series.

    Returns {} on a payload with no usable series.
    """
    columns = _series_index(payload)
    descriptors = payload.get("metricDescriptors") or []
    cadence_idx, cadence_scale = _pick_cadence(descriptors)
    power_idx = columns.get("power_w")
    if cadence_idx is None and power_idx is None:
        return {}
    cadence: list[float] = []
    power: list[float] = []
    for row in payload.get("activityDetailMetrics") or []:
        metrics = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(metrics, (list, tuple)):
            continue
        if cadence_idx is not None and cadence_idx < len(metrics):
            v = metrics[cadence_idx]
            if isinstance(v, (int, float)) and v == v and v > 0:  # rejects NaN/0
                cadence.append(float(v) * cadence_scale)
        if power_idx is not None and power_idx < len(metrics):
            v = metrics[power_idx]
            if isinstance(v, (int, float)) and v == v and v > 0:
                power.append(float(v))
    out: dict[str, Any] = {}
    if cadence:
        out["avg_cadence"] = round(sum(cadence) / len(cadence), 1)
        out["max_cadence"] = round(max(cadence), 1)
    if power:
        out["avg_power_w"] = round(sum(power) / len(power), 1)
        out["max_power_w"] = round(max(power), 1)
    return out


def parse_activity_detail_series(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Project the details payload into one wide per-tick row per data point.

    Each row carries the fields the activity recorded (heart rate, cadence,
    power, speed, elevation, cumulative distance, lat/lon); metrics the sport
    didn't record are left out (NULL at insert). A ``tick`` column gives the
    0-based index within the activity so ordering survives aggregation.
    """
    columns = _series_index(payload)
    cadence_idx, cadence_scale = _pick_cadence(payload.get("metricDescriptors") or [])
    if not columns and cadence_idx is None:
        return []
    by_index = dict(columns)
    if cadence_idx is not None:
        by_index["cadence"] = cadence_idx
    rows: list[dict[str, Any]] = []
    for tick, row in enumerate(payload.get("activityDetailMetrics") or []):
        metrics = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(metrics, (list, tuple)):
            continue
        projected: dict[str, Any] = {"tick": tick}
        for col_name, idx in by_index.items():
            if idx >= len(metrics):
                continue
            v = metrics[idx]
            if isinstance(v, (int, float)) and v == v:  # reject NaN only
                if col_name == "cadence":
                    projected[col_name] = float(v) * cadence_scale
                else:
                    projected[col_name] = float(v)
        if len(projected) > 1:
            rows.append(projected)
    return rows


def build_activity_details(db: Any) -> dict[str, int]:
    """Parse stored activity detail payloads into the series + aggregates.

    For each stored activity with a detail payload: (1) merges the derived
    avg/max cadence + avg/max power into the activity summary row, and (2)
    replaces the per-tick row in ``activity_detail_series`` with a fresh
    projection. Returns {"activities": parsed, "series": ticks_written}.
    """
    rows = db.conn.execute(
        "SELECT activity_id, details_json, details_fetched_at FROM activities "
        "WHERE details_json IS NOT NULL AND details_json != ''"
    ).fetchall()
    activities = 0
    series = 0
    for row in rows:
        payload = json.loads(row["details_json"])
        agg = parse_activity_details(payload)
        if agg:
            db.upsert_activity_summary(row["activity_id"], agg, row["details_fetched_at"])
            activities += 1
        ticks = parse_activity_detail_series(payload)
        if ticks:
            series += db.replace_activity_series(row["activity_id"], ticks)
    return {"activities": activities, "series": series}


# --- Heart-rate zone profiles ------------------------------------------------
#
# HR zone boundaries are a *user profile* setting (per sport), not daily data:
# the device stores each sport's zone floors (bpm) plus the max HR used to
# compute them. Stored raw in ``user_profile`` (profile_type='hr_zones'); the
# projection below replaces the whole ``hr_zones`` table with one row per sport
# that carries each zone's derived [min, max) range. Zone N spans
# ``zoneN_floor`` up to (but not including) ``zoneN+1_floor``; zone 5 runs to
# the max HR used. A floor of None (unset sport) yields no zone columns.


def parse_hr_zones(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the heartRateZones payload into per-sport zone-range rows."""
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        sport = item.get("sport")
        if not sport:
            continue
        floors = [item.get(f"zone{n}Floor") for n in range(1, 6)]
        if any(not isinstance(f, (int, float)) for f in floors):
            continue
        max_hr = item.get("maxHeartRateUsed")
        if not isinstance(max_hr, (int, float)) and max_hr is not None:
            max_hr = None
        row: dict[str, Any] = {
            "sport": sport,
            "training_method": item.get("trainingMethod"),
            "max_hr_used": max_hr,
            "resting_hr_used": item.get("restingHeartRateUsed"),
            "lactate_threshold_hr_used": item.get("lactateThresholdHeartRateUsed"),
            "resting_hr_auto_update": item.get("restingHrAutoUpdateUsed"),
        }
        # Zone N = [floor_N, floor_(N+1)) for N 1..4; zone 5 = [floor, max_hr].
        for n in range(1, 6):
            lo = floors[n - 1]
            hi = max_hr if n == 5 else (
                floors[n] - 1
                if isinstance(floors[n], (int, float)) and n < 5
                else None
            )
            row[f"zone{n}_min"] = lo
            row[f"zone{n}_max"] = hi
        rows.append(row)
    return rows


def build_hr_zones(db: Any) -> dict[str, int]:
    """Project the stored HR-zone profile into the ``hr_zones`` table.

    Replaces the whole table from the latest stored profile payload (there is
    exactly one such payload, keyed 'hr_zones'). Returns {"hr_zones": rows}.
    """
    profile = db.get_profile("hr_zones")
    if not profile:
        return {"hr_zones": 0}
    payload = json.loads(profile["raw_json"])
    if not isinstance(payload, list):
        return {"hr_zones": 0}
    rows = parse_hr_zones(payload)
    for row in rows:
        row["fetched_at"] = profile["fetched_at"]
    return {"hr_zones": db.replace_hr_zones(rows)}