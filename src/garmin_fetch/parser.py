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
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Garmin activity typeKeys for which ``pace_min_km`` (min/km) is meaningful.
_RUNNING_TYPES = frozenset({
    "running", "track_running", "trail_running", "indoor_running",
    "treadmill_running", "virtual_run", "street_running", "ultra_run",
    "running_treadmill", "running_street", "running_track", "running_trail",
})


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


def _num(value: Any) -> float | None:
    """A numeric value, or None for anything else (incl. NaN)."""
    if isinstance(value, (int, float)) and value == value:
        return float(value)
    return None


def _hours(seconds: Any) -> float | None:
    """Seconds as hours (4dp), or None when the source isn't a number."""
    if isinstance(seconds, (int, float)):
        return round(seconds / 3600, 4)
    return None


def _local_time(epoch_ms: Any) -> str | None:
    """Epoch-millis 'Local' timestamp as a wall-clock ``HH:MM``, or None.

    Garmin's ``*TimestampLocal`` fields are epoch millis with the UTC offset
    already applied (``sleepStartTimestampLocal`` == the local wall-clock time
    encoded as if it were UTC). Interpreting the value in UTC recovers that
    wall clock; ``datetime.fromtimestamp`` without a tz would apply the host
    timezone's offset on top, shifting it by the UTC offset again.
    """
    if isinstance(epoch_ms, (int, float)):
        return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).strftime("%H:%M")
    return None


# --- Per-type extractors --------------------------------------------------

def parse_heart_rate(payload: dict[str, Any]) -> dict[str, Any]:
    # The daily heart-rate payload carries only the summary scalars (plus an
    # intraday ``heartRateValues`` series); it has no per-day zone block. Zone
    # boundaries are a profile setting projected separately into ``hr_zones``.
    return _leaf(payload, [
        ("resting_hr", "restingHeartRate"),
        ("min_hr", "minHeartRate"),
        ("max_hr", "maxHeartRate"),
    ])


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


def parse_daily_summary(payload: dict[str, Any]) -> dict[str, Any]:
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
        ("avg_stress", "averageStressLevel"),
        ("max_stress", "maxStressLevel"),
        ("moderate_intensity_minutes", "moderateIntensityMinutes"),
        ("vigorous_intensity_minutes", "vigorousIntensityMinutes"),
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
    for ev in events:
        event = ev.get("event") or {}
        b = event.get("bodyBatteryImpact")
        if isinstance(b, (int, float)):
            impact += b
    if impact:
        return {"body_battery_net_change": round(impact, 2)}
    return {}


def parse_max_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Arrives as a list wrapping a single element with a ``generic`` bucket."""
    values = payload.get("value")
    if not isinstance(values, list) or not values:
        return {}
    bucket = values[0] if isinstance(values[0], dict) else {}
    generic = bucket.get("generic") or {}
    return _leaf(generic, [
        ("vo2max", "vo2MaxPreciseValue"),
        ("fitness_age", "fitnessAge"),
    ])


def parse_fitnessage(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("fitness_age", "fitnessAge"),
    ])


def parse_intensity_minutes(payload: dict[str, Any]) -> dict[str, Any]:
    return _leaf(payload, [
        ("moderate_intensity_minutes", "moderateMinutes"),
        ("vigorous_intensity_minutes", "vigorousMinutes"),
        ("weekly_moderate", "weeklyModerate"),
        ("weekly_vigorous", "weeklyVigorous"),
        ("weekly_total", "weeklyTotal"),
        ("day_of_goal_met", "dayOfGoalMet"),
    ])


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
        ("vo2max", "vo2MaxPreciseValue"),
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
            ("weekly_training_load", "weeklyTrainingLoad"),
            ("training_status_feedback", "trainingStatusFeedbackPhrase"),
        ]))
    return out


def parse_lactate_threshold(payload: dict[str, Any]) -> dict[str, Any]:
    """Running lactate threshold (HR/speed) plus running power, from per-day
    range arrays. Garmin detects lactate threshold from running sessions, so
    the ``power`` entry is *running* power (running FTP), not cycling FTP."""
    out: dict[str, Any] = {}
    for key, column in (
        ("heart_rate", "lactate_threshold_hr"),
        ("speed", "lactate_threshold_speed_kmh"),
        ("power", "running_ftp_watts"),
    ):
        entries = payload.get(key)
        if isinstance(entries, dict):
            entries = [entries]
        if entries:
            value = entries[0].get("value")
            if value is not None:
                if column == "lactate_threshold_speed_kmh":
                    # Garmin reports LT speed in 0.1 m/s -> km/h (x36).
                    out[column] = round(value * 36, 2)
                else:
                    out[column] = value
    return out


def _first_entry(entries: Any) -> dict[str, Any] | None:
    """First dict in a list (or a dict itself), or None."""
    if isinstance(entries, dict):
        return entries
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        return entries[0]
    return None


def parse_sweat_loss(payload: dict[str, Any]) -> dict[str, Any]:
    """Estimated sweat loss from the daily hydration payload, in ml."""
    return _leaf(payload, [
        ("sweat_loss_ml", "sweatLossInML"),
    ])


def parse_weight(payload: dict[str, Any]) -> dict[str, Any]:
    entry = _first_entry(payload.get("dateWeightList"))
    if entry is None:
        entry = payload.get("totalAverage") or {}
    if not isinstance(entry, dict):
        return {}
    return _leaf(entry, [
        ("weight_kg", "weight"),
        ("bmi", "bmi"),
        ("body_fat_pct", "bodyFat"),
    ])


def parse_body_composition(payload: dict[str, Any]) -> dict[str, Any]:
    entry = _first_entry(payload.get("dateWeightList"))
    if entry is None:
        entry = payload.get("totalAverage") or {}
    if not isinstance(entry, dict):
        return {}
    return _leaf(entry, [
        ("weight_kg", "weight"),
        ("bmi", "bmi"),
        ("body_fat_pct", "bodyFat"),
        ("body_water_pct", "bodyWater"),
        ("bone_mass_kg", "boneMass"),
        ("muscle_mass_kg", "muscleMass"),
        ("physique_rating", "physiqueRating"),
        ("visceral_fat", "visceralFat"),
        ("metabolic_age", "metabolicAge"),
    ])


def parse_blood_pressure(payload: dict[str, Any]) -> dict[str, Any]:
    entry = _first_entry(payload.get("measurementSummaries"))
    if entry is None:
        return {}
    return _leaf(entry, [
        ("systolic_bp", "systolic"),
        ("diastolic_bp", "diastolic"),
        ("pulse_bpm", "pulse"),
    ])


def parse_endurance_score(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dto = payload.get("enduranceScoreDTO")
    if isinstance(dto, dict):
        out.update(_leaf(dto, [
            ("endurance_score", "enduranceScore"),
            ("endurance_score_level", "level"),
            ("endurance_score_vo2max", "vo2MaxValue"),
        ]))
    if not out:
        out.update(_leaf(payload, [("endurance_score", "avg")]))
    return out


def parse_hill_score(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    entries = payload.get("hillScoreDTOList")
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        out.update(_leaf(entries[0], [("hill_score", "hillScore")]))
    if not out:
        for value in (payload.get("periodAvgScore") or {}).values():
            if value is not None:
                out["hill_score"] = value
                break
    return out


def parse_running_tolerance(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, list):
        payload = {"value": payload}
    entry = _first_entry(payload.get("value"))
    if entry is None:
        entry = _first_entry(payload)
    if entry is None:
        return {}
    return _leaf(entry, [
        ("running_tolerance", "runningTolerance"),
        ("running_tolerance_value", "value"),
    ])


def parse_cycling_ftp(payload: dict[str, Any]) -> dict[str, Any]:
    """Cycling functional threshold power (watts) from the FTP endpoint."""
    if isinstance(payload, list):
        payload = {"value": payload}
    entry = _first_entry(payload.get("value"))
    if entry is None:
        entry = _first_entry(payload)
    if entry is None:
        return {}
    out = _leaf(entry, [
        ("cycling_ftp_watts", "ftp"),
        ("cycling_ftp_watts", "functionalThresholdPower"),
        ("cycling_ftp_watts", "value"),
    ])
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
    "daily_summary": parse_daily_summary,
    "body_battery": parse_body_battery,
    "max_metrics": parse_max_metrics,
    "fitnessage": parse_fitnessage,
    "intensity_minutes": parse_intensity_minutes,
    "training_status": parse_training_status,
    "lactate_threshold": parse_lactate_threshold,
    "sweat_loss": parse_sweat_loss,
    "weight": parse_weight,
    "body_composition": parse_body_composition,
    "blood_pressure": parse_blood_pressure,
    "endurance_score": parse_endurance_score,
    "hill_score": parse_hill_score,
    "running_tolerance": parse_running_tolerance,
    "cycling_ftp": parse_cycling_ftp,
}

#: daily_metrics columns each daily type's extractor can write. Kept in sync
#: with the extractors above; used to prune a newly-excluded type's columns
#: (a column is only NULLed when no *other enabled* type also writes it).
TYPE_COLUMNS: dict[str, set[str]] = {
    "heart_rate": {"resting_hr", "min_hr", "max_hr"},
    "steps": {"total_steps", "pushes"},
    "sleep": {
        "sleep_time_hours", "nap_time_hours", "deep_sleep_hours",
        "light_sleep_hours", "rem_sleep_hours", "awake_sleep_hours",
        "unmeasurable_sleep_hours", "average_respiration", "awake_count",
        "avg_sleep_stress", "sleep_start_local", "sleep_end_local",
        "resting_hr", "hrv_status", "body_battery_change",
        "restless_moments_count", "sleep_score", "sleep_score_qualifier",
    },
    "hrv": {"hrv_last_night_avg", "hrv_last_night_5min_high", "hrv_status"},
    "stress": {"avg_stress", "max_stress"},
    "respiration": {
        "respiration_waking_avg", "respiration_sleep_avg",
        "respiration_lowest", "respiration_highest",
    },
    "spo2": {
        "spo2_avg", "spo2_lowest", "spo2_latest",
        "spo2_avg_sleep", "spo2_last_7d_avg",
    },
    "rhr": {"resting_hr"},
    "daily_summary": {
        "total_steps", "total_distance_m", "total_kcal", "active_kcal",
        "bmr_kcal", "burned_kcal", "consumed_kcal", "remaining_kcal",
        "resting_hr", "min_hr", "max_hr",
        "avg_stress", "max_stress", "moderate_intensity_minutes",
        "vigorous_intensity_minutes",
        "body_battery_charged", "body_battery_drained", "body_battery_highest",
        "body_battery_lowest", "body_battery_most_recent",
        "body_battery_at_wake", "average_spo2", "lowest_spo2", "latest_spo2",
        "highest_respiration", "lowest_respiration", "avg_waking_respiration",
        "highly_active_hours", "active_hours", "sedentary_hours",
    },
    "body_battery": {"body_battery_net_change"},
    "max_metrics": {"vo2max", "fitness_age"},
    "fitnessage": {"fitness_age"},
    "intensity_minutes": {
        "moderate_intensity_minutes", "vigorous_intensity_minutes",
        "weekly_moderate", "weekly_vigorous", "weekly_total", "day_of_goal_met",
    },
    "training_status": {
        "vo2max", "training_balance_feedback",
        "weekly_training_load", "training_status_feedback",
    },
    "lactate_threshold": {
        "lactate_threshold_hr", "lactate_threshold_speed_kmh", "running_ftp_watts",
    },
    "sweat_loss": {"sweat_loss_ml"},
    "weight": {"weight_kg", "bmi", "body_fat_pct"},
    "body_composition": {
        "weight_kg", "bmi", "body_fat_pct", "body_water_pct", "bone_mass_kg",
        "muscle_mass_kg", "physique_rating", "visceral_fat", "metabolic_age",
    },
    "blood_pressure": {"systolic_bp", "diastolic_bp", "pulse_bpm"},
    "endurance_score": {
        "endurance_score", "endurance_score_level", "endurance_score_vo2max",
    },
    "hill_score": {"hill_score"},
    "running_tolerance": {"running_tolerance", "running_tolerance_value"},
    "cycling_ftp": {"cycling_ftp_watts"},
}


#: Types whose stored values persist until superseded (sparse, e.g. lactate
#: threshold only updates when a session recalculates it). Days with no data
#: keep the last known value instead of going NULL.
_FFILL_TYPES = frozenset({"lactate_threshold"})


def _forward_fill(parsed: dict[str, Any], carry: dict[str, Any]) -> dict[str, Any]:
    """Merge today's parsed values into the running carry, then return the row."""
    carry.update(parsed)
    return dict(carry)


def build_daily_rows(
    db: Any, types: list[str] | None = None, force: bool = False
) -> dict[str, int]:
    """Parse stored raw metrics into ``daily_metrics``.

    Reads the raw ``metrics`` store, applies the matching parser for each
    (data_type, date) row, and merges the scalars into the daily row. By
    default only rows whose raw payload changed since last parsed are touched
    (tracked via ``metrics.parsed_at``); pass ``force=True`` to re-parse
    everything. Returns {data_type: dates_parsed}.
    """
    # Some registered types (e.g. training_readiness) are fetch-only and have
    # no projection; skip them rather than crash.
    names = [n for n in (types or list(PARSERS)) if n in PARSERS]
    # NOTE: a full re-parse deliberately does NOT drop and rebuild the
    # daily_metrics columns anymore. daily_metrics is one shared, RLS-scoped
    # table: dropping a column deletes every account's projected values, not
    # just the syncing user's, and they only come back for dates that user
    # re-parses. Stale columns from renamed/removed metrics simply linger as
    # NULLs — a safe trade-off for the multi-user server.
    counts: dict[str, int] = {}
    for name in names:
        rows = db.metrics_rows(name)
        parsed_at = {} if force else db.metric_parsed_at(name)
        if force:
            dirty: set[str] | None = None
        else:
            changed = {
                r["calendar_date"]
                for r in rows
                if parsed_at.get(r["calendar_date"]) != r["fetched_at"]
            }
            if name in _FFILL_TYPES:
                # Carry semantics: a change anywhere recasts the whole chain, so
                # re-parse the entire (sparse) type from scratch — cheap and exact.
                if changed:
                    dirty = {r["calendar_date"] for r in rows}
                else:
                    dirty = set()
            else:
                dirty = changed
        carry: dict[str, Any] = {}
        for row in rows:
            if not force and row["calendar_date"] not in dirty:
                continue
            parsed = PARSERS[name](json.loads(row["raw_json"]))
            if name in _FFILL_TYPES:
                parsed = _forward_fill(parsed, carry)
            db.mark_metric_parsed(name, row["calendar_date"])
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
        ("norm_power_w", "normPower"),
        ("vo2max", "vO2MaxValue"),
        ("avg_stride_length_cm", "avgStrideLength"),
        ("avg_vertical_oscillation_cm", "avgVerticalOscillation"),
        ("avg_ground_contact_time_ms", "avgGroundContactTime"),
        ("avg_vertical_ratio_pct", "avgVerticalRatio"),
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
    # Running pace in min/km (decimal minutes), only for running-type
    # activities — the same number flipped from km/h, in the units runners
    # actually talk in. Non-running activities get NULL, not a meaningless
    # min/km.
    if (
        out.get("activity_type") in _RUNNING_TYPES
        and isinstance(out.get("avg_speed_kmh"), (int, float))
        and out["avg_speed_kmh"] > 0
    ):
        out["pace_min_km"] = round(60 / out["avg_speed_kmh"], 2)
    # Heart-rate zones 1..5: each as a percentage of total activity duration.
    duration_s = payload.get("duration")
    if isinstance(duration_s, (int, float)) and duration_s > 0:
        for zone in range(1, 6):
            secs = payload.get(f"hrTimeInZone_{zone}")
            if isinstance(secs, (int, float)) and secs > 0:
                out[f"hr_time_zone_{zone}_pct"] = round(secs / duration_s * 100, 1)
    # Power zones 1..5: same layout (seconds in zone -> % of duration).
    if isinstance(duration_s, (int, float)) and duration_s > 0:
        for zone in range(1, 6):
            secs = payload.get(f"powerTimeInZone_{zone}")
            if isinstance(secs, (int, float)) and secs > 0:
                out[f"power_time_zone_{zone}_pct"] = round(secs / duration_s * 100, 1)
    # start_date: local start date (YYYY-MM-DD).
    start_date = str(out.get("start_time_local") or "")[:10]
    if start_date:
        out["start_date"] = start_date
    # End time (local wall clock): Garmin only exposes endTimeGMT, so derive
    # the local end from the local start plus elapsed duration (this matches
    # the app's displayed end time, since local and GMT shift together).
    start_local = out.get("start_time_local")
    elapsed_s = payload.get("elapsedDuration") or payload.get("duration")
    if isinstance(start_local, str) and isinstance(elapsed_s, (int, float)):
        try:
            end = datetime.strptime(
                start_local, "%Y-%m-%d %H:%M:%S"
            ) + timedelta(seconds=elapsed_s)
            out["end_time_local"] = end.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return out


def build_activity_summaries(db: Any, force: bool = False) -> dict[str, int]:
    """Parse stored activity summaries into ``activity_summaries``.

    Reads the raw ``activities`` store and projects each summary into one
    curated row keyed by ``activityId``. Only activities whose summary payload
    changed since last parse are touched unless ``force``. Returns
    {"activities": rows_parsed}.
    """
    rows = db.activity_rows()
    marker = "summary_parsed_at"
    stamped = {} if force else db.activity_parsed_at(marker)
    count = 0
    for row in rows:
        if not force and stamped.get(row["activity_id"]) == row["fetched_at"]:
            continue
        parsed = parse_activity_summary(json.loads(row["raw_json"]))
        if not parsed:
            continue
        db.upsert_activity_summary(row["activity_id"], parsed, row["fetched_at"])
        db.mark_activity_parsed(row["activity_id"], marker, "fetched_at")
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
    "directSpeed": "speed_kmh",
    "directElevation": "elevation_m",
    "sumDistance": "distance_m",
    "directRespirationRate": "respiration_rate",
    "sumAccumulatedPower": "accumulated_power_w",
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
                elif col_name == "speed_kmh":
                    # Garmin records m/s -> km/h (x3.6).
                    projected[col_name] = round(float(v) * 3.6, 2)
                else:
                    projected[col_name] = float(v)
        if len(projected) > 1:
            rows.append(projected)
    return rows


def build_activity_details(db: Any, force: bool = False) -> dict[str, int]:
    """Parse stored activity detail payloads into the series + aggregates.

    For each stored activity with a detail payload: (1) merges the derived
    avg/max cadence + avg/max power into the activity summary row, and (2)
    replaces the per-tick row in ``activity_detail_series`` with a fresh
    projection. Returns {"activities": parsed, "series": ticks_written}.
    Only activities whose details changed since last parse are touched unless
    ``force``.
    """
    rows = db.activity_detail_rows()
    marker = "details_parsed_at"
    stamped = {} if force else db.activity_parsed_at(marker)
    activities = 0
    series = 0
    for row in rows:
        if (
            not force
            and row["details_fetched_at"] is not None
            and stamped.get(row["activity_id"]) == row["details_fetched_at"]
        ):
            continue
        payload = json.loads(row["details_json"])
        agg = parse_activity_details(payload)
        if agg:
            db.upsert_activity_summary(row["activity_id"], agg, row["details_fetched_at"])
            activities += 1
        ticks = parse_activity_detail_series(payload)
        if ticks:
            series += db.replace_activity_series(row["activity_id"], ticks)
        db.mark_activity_parsed(row["activity_id"], marker, "details_fetched_at")
    return {"activities": activities, "series": series}


# --- Activity splits ---------------------------------------------------------
#
# Each activity's splits payload (per-lap work/rest chunks from
# ``get_activity_splits``) is stored raw on the ``activities`` row
# (``splits_json``) and projected below into the ``activity_splits`` table.
# Pace is derived from distance+duration so it is comparable regardless of the
# units Garmin happens to use in the raw field.


def _split_type(item: dict[str, Any]) -> str:
    """Map Garmin's lap ``intensityType`` to the documented ``split_type``.

    Garmin tags a lap as ACTIVE (a work chunk, e.g. a per-km distance split),
    REST or INACTIVE (a recovery chunk). Anything else is passed through
    lowercased, falling back to "split".
    """
    raw = (item.get("intensityType") or item.get("splitType") or "").strip().upper()
    if raw == "ACTIVE":
        return "distance"
    if raw in ("REST", "INACTIVE"):
        return "rest"
    return raw.lower() or "split"


def parse_activity_splits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Project an activity's splits payload into typed per-split rows.

    Garmin's ``get_activity_splits`` payload carries the per-lap list under
    ``lapDTOs`` (some payload shapes use ``splits``); each lap is one
    work/rest chunk with distance, duration and (for most sports) HR/power/
    cadence. ``start_time_s`` is the lap's offset into the activity (seconds,
    accumulated from preceding laps' elapsed durations). Returns [] for a
    payload with no usable lap list. Values that the sport didn't record are
    omitted (NULL at insert).
    """
    raw_splits = payload.get("lapDTOs")
    if raw_splits is None:
        raw_splits = payload.get("splits")
    if not isinstance(raw_splits, list) or not raw_splits:
        return []
    rows: list[dict[str, Any]] = []
    offset = 0.0
    for index, item in enumerate(raw_splits):
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {
            "split_number": index,
            "split_type": _split_type(item),
            "start_time_s": offset,
        }
        distance = _num(item.get("distance"))
        duration = _num(item.get("duration"))
        if distance is not None:
            row["distance_m"] = round(distance, 1)
        if duration is not None:
            row["duration_s"] = round(duration, 1)
        if distance and duration:
            row["pace_sec_per_km"] = round(duration / (distance / 1000.0), 1)
        for src, col in (
            ("averageHR", "avg_hr"),
            ("avgHr", "avg_hr"),
            ("maxHR", "max_hr"),
            ("maxHr", "max_hr"),
            ("averagePower", "avg_power"),
            ("avgPower", "avg_power"),
            ("maxPower", "max_power"),
            ("averageRunCadence", "avg_cadence"),
            ("averageBikeCadence", "avg_cadence"),
            ("avgCadence", "avg_cadence"),
            ("maxRunCadence", "max_cadence"),
            ("maxBikeCadence", "max_cadence"),
            ("maxCadence", "max_cadence"),
            ("elevationGain", "elevation_gain_m"),
        ):
            value = _num(item.get(src))
            if value is not None:
                row[col] = value
        elapsed = _num(item.get("elapsedDuration"))
        offset += elapsed if elapsed is not None else (duration or 0.0)
        rows.append(row)
    return rows


def build_activity_splits(db: Any, force: bool = False) -> dict[str, int]:
    """Project stored activity splits payloads into the ``activity_splits`` table.

    For each stored activity with a splits payload, replaces the per-split
    projection. Only activities whose splits payload changed since last parse
    are touched unless ``force``. Returns {"activities": parsed,
    "splits": splits_written}.
    """
    rows = db.activity_split_rows()
    marker = "splits_parsed_at"
    stamped = {} if force else db.activity_parsed_at(marker)
    activities = 0
    splits = 0
    for row in rows:
        if (
            not force
            and row["splits_fetched_at"] is not None
            and stamped.get(row["activity_id"]) == row["splits_fetched_at"]
        ):
            continue
        payload = json.loads(row["splits_json"])
        parsed = parse_activity_splits(payload)
        if parsed:
            splits += db.replace_activity_splits(row["activity_id"], parsed)
            activities += 1
        if row["splits_fetched_at"] is not None:
            db.mark_activity_parsed(row["activity_id"], marker, "splits_fetched_at")
    return {"activities": activities, "splits": splits}


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


# --- Power-zone profiles -----------------------------------------------------
#
# Cycling power-zone boundaries are a *user profile* setting (per sport), like
# HR zones: the device stores each sport's zone floors (watts) plus the
# functional threshold power used to compute them. Stored raw in
# ``user_profile`` (profile_type='power_zones'); the projection below replaces
# the whole ``power_zones`` table with one row per sport. Zone N spans
# ``zoneN_floor`` up to (but not including) ``zoneN+1_floor``; the last zone
# runs to the functional threshold power.


def parse_power_zones(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the power-zones payload into per-sport zone-range rows.

    Mirrors ``parse_hr_zones``: one row per sport; zone N = [floor_N,
    floor_(N+1)) for N < 7, zone 7 = [floor, functional_threshold_power].
    A sport with no numeric floors yields no row.
    """
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        sport = item.get("sportType") or item.get("sport")
        if not sport:
            continue
        floors = [item.get(f"zone{n}Floor") for n in range(1, 8)]
        if not any(isinstance(f, (int, float)) for f in floors):
            continue
        ftp = item.get("functionalThresholdPower")
        if not isinstance(ftp, (int, float)):
            ftp = None
        row: dict[str, Any] = {
            "sport": sport,
            "functional_threshold_power": ftp,
        }
        for n in range(1, 8):
            lo = floors[n - 1]
            if not isinstance(lo, (int, float)):
                continue
            hi = ftp if n == 7 else (
                floors[n] - 1
                if isinstance(floors[n], (int, float)) and n < 7
                else None
            )
            row[f"zone{n}_min"] = lo
            row[f"zone{n}_max"] = hi
        rows.append(row)
    return rows


def build_power_zones(db: Any) -> dict[str, int]:
    """Project the stored power-zone profile into the ``power_zones`` table.

    Replaces the whole table from the latest stored payload (user_profile key
    'power_zones'). Returns {"power_zones": rows}.
    """
    profile = db.get_profile("power_zones")
    if not profile:
        return {"power_zones": 0}
    payload = json.loads(profile["raw_json"])
    if not isinstance(payload, list):
        return {"power_zones": 0}
    rows = parse_power_zones(payload)
    for row in rows:
        row["fetched_at"] = profile["fetched_at"]
    return {"power_zones": db.replace_power_zones(rows)}


# --- Race predictions ---------------------------------------------------------
#
# Garmin's race predictor is a current-fitness snapshot (predicted finish times
# for 5k/10k/half/full), not day-keyed history. It's fetched as a profile-style
# payload under user_profile('race_predictions') and projected into the single
# ``race_predictions`` row. Times arrive in seconds; they're stored as minutes.


def parse_race_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the race-predictor payload into minute-based predicted times.

    Returns ``{}`` when none of the four predicted times is present.
    """
    out: dict[str, Any] = {}
    for src, name in (
        ("time5K", "time_5k_min"),
        ("time10K", "time_10k_min"),
        ("timeHalfMarathon", "time_half_marathon_min"),
        ("timeMarathon", "time_marathon_min"),
    ):
        seconds = payload.get(src)
        if isinstance(seconds, (int, float)):
            out[name] = round(seconds / 60, 2)
    if not out:
        return {}
    calendar_date = payload.get("calendarDate")
    if isinstance(calendar_date, str) and calendar_date:
        out["calendar_date"] = calendar_date
    return out


def build_race_predictions(db: Any) -> dict[str, int]:
    """Project the stored race-predictor snapshot into ``race_predictions``.

    Replaces the single row from the latest stored payload (user_profile key
    'race_predictions'). Returns {"race_predictions": rows}.
    """
    profile = db.get_profile("race_predictions")
    if not profile:
        return {"race_predictions": 0}
    payload = json.loads(profile["raw_json"])
    if not isinstance(payload, dict):
        return {"race_predictions": 0}
    row = parse_race_predictions(payload)
    if not row:
        return {"race_predictions": 0}
    row["fetched_at"] = profile["fetched_at"]
    return {"race_predictions": db.replace_race_predictions(row)}


# --- Gear ---------------------------------------------------------------------
#
# Garmin gear (bikes, shoes, ...) is a current "what's in your garage" snapshot,
# like hr_zones: one entry per item with its cumulative stats (distance ridden,
# max speed, activity count, last use). Stored raw in ``user_profile``
# (profile_type='gear') and projected below into one ``gear`` row per item,
# replacing the whole table each sync (no history).


def _gear_value(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """First present non-None, non-empty value among candidate keys."""
    for key in keys:
        if isinstance(item, dict):
            value = item.get(key)
            if value is not None and value != "":
                return value
    return default


def _iso_date(value: Any) -> str | None:
    """Normalise a Garmin datetime string to ``YYYY-MM-DD`` (or None)."""
    if not isinstance(value, str):
        return None
    return value[:10] or None


def parse_gear(payload: Any) -> list[dict[str, Any]]:
    """Project the gear payload into per-item rows (most recent stats).

    The gear list may come back as a bare list or wrapped (e.g.
    ``{"gearList": [...]}``). Only a minimal, high-signal set is kept per item:
    type, name, cumulative distance, activity count, last use and retired
    status. ``name`` is the user's nickname (``displayName``) when set,
    otherwise the real product name (``customMakeModel``, e.g. "novablast 5").
    Cumulative stats (total distance, activity count, last activity date)
    arrive per item from the stats endpoint merged into the same dict.
    Distance is meters -> km.
    """
    if isinstance(payload, dict):
        for key in ("gearList", "gear", "items"):
            wrapped = payload.get(key)
            if isinstance(wrapped, list):
                payload = wrapped
                break
    if not isinstance(payload, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        gear_uuid = _gear_value(item, "uuid", "gearUUID", "gearUuid")
        if not gear_uuid:
            continue
        row: dict[str, Any] = {
            "gear_uuid": str(gear_uuid),
            "gear_type": _gear_value(item, "gearTypeName", "typeKey", "type", "gearType"),
            "name": _gear_value(item, "displayName", "nickname")
            or _gear_value(item, "customMakeModel", "customModel", "model"),
            "last_activity_date": _iso_date(
                _gear_value(item, "lastActivityDate", "lastUsedDate")
            ),
        }
        distance_m = _gear_value(item, "totalDistance", "totalDistanceMeters", "distance")
        if isinstance(distance_m, (int, float)):
            row["total_distance_km"] = round(distance_m / 1000, 3)
        activity_count = _gear_value(
            item, "activityCount", "totalActivities", "numberOfActivities", "activity_count"
        )
        if isinstance(activity_count, (int, float)):
            row["activity_count"] = int(activity_count)
        status = _gear_value(item, "gearStatusName", "status")
        if isinstance(status, str) and status:
            row["retired"] = status.lower() == "retired"
        else:
            retired = _gear_value(item, "retired", "isRetired")
            if isinstance(retired, bool):
                row["retired"] = retired
            elif isinstance(retired, (int, float)):
                row["retired"] = bool(retired)
        rows.append(row)
    return rows


def build_gear(db: Any) -> dict[str, int]:
    """Project the stored gear snapshot into the ``gear`` table.

    Replaces the whole table from the latest stored payload (user_profile key
    'gear'). Returns {"gear": rows}.
    """
    profile = db.get_profile("gear")
    if not profile:
        return {"gear": 0}
    payload = json.loads(profile["raw_json"])
    rows = parse_gear(payload)
    if not rows:
        return {"gear": 0}
    for row in rows:
        row["fetched_at"] = profile["fetched_at"]
    return {"gear": db.replace_gear(rows)}


# --- Devices -----------------------------------------------------------------
#
# The current device list (from ``get_devices``) is stored raw in
# ``user_profile`` (profile_type='devices'). Only the identity is kept — the
# agent needs the model name to say what the user records on; battery/firmware/
# last-used are noise and the per-device settings calls that would populate them
# were dropped. The projection replaces the whole ``devices`` table with one row
# per device.


def parse_devices(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the device snapshot into one row per device.

    Returns [] for a payload with no usable device rows.
    """
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        device_id = item.get("deviceId")
        if device_id is None:
            continue
        rows.append(
            {
                "device_id": device_id,
                "model_name": item.get("modelName"),
                "display_name": item.get("displayName") or item.get("deviceName"),
            }
        )
    return rows


def build_devices(db: Any) -> dict[str, int]:
    """Project the stored device snapshot into the ``devices`` table.

    Replaces the whole table from the latest stored payload (user_profile key
    'devices'). Returns {"devices": rows}.
    """
    profile = db.get_profile("devices")
    if not profile:
        return {"devices": 0}
    payload = json.loads(profile["raw_json"])
    if not isinstance(payload, list):
        return {"devices": 0}
    rows = parse_devices(payload)
    for row in rows:
        row["fetched_at"] = profile["fetched_at"]
    return {"devices": db.replace_devices(rows)}


# --- Activity weather ---------------------------------------------------------
#
# Each activity's weather payload (observed conditions at the activity site)
# is stored raw on the ``activities`` row (``weather_json``) and projected into
# the curated ``activity_summaries`` columns below. Values arrive in imperial
# units (degF, mph); they're converted to metric (degC, km/h) to match the rest
# of the projection.


def _f_to_c(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round((value - 32) * 5 / 9, 2)
    return None


def _mph_to_kmh(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(value * 1.609344, 2)
    return None


def parse_activity_weather(payload: dict[str, Any]) -> dict[str, Any]:
    """Project one activity weather payload into metric summary columns.

    Returns ``{}`` for a payload with no usable data.
    """
    out: dict[str, Any] = {}
    temp = _f_to_c(payload.get("temp"))
    if temp is not None:
        out["weather_temp_c"] = temp
    apparent = _f_to_c(payload.get("apparentTemp"))
    if apparent is not None:
        out["weather_apparent_c"] = apparent
    humidity = payload.get("relativeHumidity")
    if isinstance(humidity, (int, float)):
        out["weather_humidity"] = humidity
    wind = _mph_to_kmh(payload.get("windSpeed"))
    if wind is not None:
        out["weather_wind_kmh"] = wind
    description = _get(payload, "weatherTypeDTO", "desc")
    if isinstance(description, str) and description:
        out["weather_description"] = description
    return out


def build_activity_weather(db: Any, force: bool = False) -> dict[str, int]:
    """Project each stored activity's weather payload into its summary row.

    Reads the raw ``weather_json`` column on ``activities`` and merges the
    derived weather scalars into ``activity_summaries``. Only activities whose
    weather payload changed since last parse are touched unless ``force``.
    Returns {"activities": rows_parsed}.
    """
    rows = db.activity_weather_rows()
    marker = "weather_parsed_at"
    stamped = {} if force else db.activity_parsed_at(marker)
    count = 0
    for row in rows:
        if (
            not force
            and row["weather_fetched_at"] is not None
            and stamped.get(row["activity_id"]) == row["weather_fetched_at"]
        ):
            continue
        parsed = parse_activity_weather(json.loads(row["weather_json"]))
        if not parsed:
            continue
        db.upsert_activity_summary(
            row["activity_id"], parsed, row["weather_fetched_at"]
        )
        db.mark_activity_parsed(row["activity_id"], marker, "weather_fetched_at")
        count += 1
    return {"activities": count}