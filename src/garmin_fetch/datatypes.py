from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DataType:
    """A single type of daily Garmin data that the fetcher can sync.

    Only describes how to *fetch* raw data. Parsing into typed tables is a
    separate, later step. ``label``/``description`` are human-facing metadata
    for the settings UI (e.g. "Cycling FTP — needs a power meter"), so users
    can toggle off the types they don't have or want.
    """

    name: str
    fetch: Callable[[Any, str], Any]
    label: str = ""
    description: str = ""

    def __str__(self) -> str:
        return self.label or self.name

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "label": self.label or self.name,
            "description": self.description,
        }


# --- Fetch adapters -------------------------------------------------------


def fetch_heart_rate(client: Any, cdate: str) -> Any:
    return client.get_heart_rates(cdate)


def fetch_steps(client: Any, cdate: str) -> Any:
    return client.get_steps_data(cdate)


def fetch_sleep(client: Any, cdate: str) -> Any:
    return client.get_sleep_data(cdate)


def fetch_body_battery(client: Any, cdate: str) -> Any:
    return client.get_body_battery_events(cdate)


def fetch_hrv(client: Any, cdate: str) -> Any:
    return client.get_hrv_data(cdate)


def fetch_stress(client: Any, cdate: str) -> Any:
    return client.get_all_day_stress(cdate)


def fetch_respiration(client: Any, cdate: str) -> Any:
    return client.get_respiration_data(cdate)


def fetch_spo2(client: Any, cdate: str) -> Any:
    return client.get_spo2_data(cdate)


def fetch_rhr(client: Any, cdate: str) -> Any:
    return client.get_rhr_day(cdate)


def fetch_daily_summary(client: Any, cdate: str) -> Any:
    return client.get_stats(cdate)


def fetch_intensity_minutes(client: Any, cdate: str) -> Any:
    return client.get_intensity_minutes_data(cdate)


def fetch_max_metrics(client: Any, cdate: str) -> Any:
    return client.get_max_metrics(cdate)


def fetch_training_readiness(client: Any, cdate: str) -> Any:
    return client.get_training_readiness(cdate)


def fetch_morning_training_readiness(client: Any, cdate: str) -> Any:
    return client.get_morning_training_readiness(cdate)


def fetch_fitnessage(client: Any, cdate: str) -> Any:
    return client.get_fitnessage_data(cdate)


def fetch_training_status(client: Any, cdate: str) -> Any:
    return client.get_training_status(cdate)


def fetch_lactate_threshold(client: Any, cdate: str) -> Any:
    return client.get_lactate_threshold(
        latest=False, start_date=cdate, end_date=cdate
    )


def fetch_sweat_loss(client: Any, cdate: str) -> Any:
    return client.get_hydration_data(cdate)


def fetch_weight(client: Any, cdate: str) -> Any:
    return client.get_daily_weigh_ins(cdate)


def fetch_body_composition(client: Any, cdate: str) -> Any:
    return client.get_body_composition(cdate, cdate)


def fetch_blood_pressure(client: Any, cdate: str) -> Any:
    return client.get_blood_pressure(cdate, cdate)


def fetch_endurance_score(client: Any, cdate: str) -> Any:
    return client.get_endurance_score(cdate, cdate)


def fetch_hill_score(client: Any, cdate: str) -> Any:
    return client.get_hill_score(cdate, cdate)


def fetch_running_tolerance(client: Any, cdate: str) -> Any:
    return client.get_running_tolerance(cdate, cdate, aggregation="daily")


def fetch_cycling_ftp(client: Any, cdate: str) -> Any:
    """Cycling functional threshold power (watts), from the cycling power meter."""
    return client.get_functional_threshold_power_range(
        cdate, cdate, sport="CYCLING", aggregation="daily"
    )


DATA_TYPES: dict[str, DataType] = {
    "heart_rate": DataType(
        "heart_rate", fetch_heart_rate, "Heart rate", "All-day heart-rate curve"
    ),
    "steps": DataType("steps", fetch_steps, "Steps", "Daily step count"),
    "sleep": DataType(
        "sleep", fetch_sleep, "Sleep", "Sleep stages, score and duration"
    ),
    "body_battery": DataType(
        "body_battery",
        fetch_body_battery,
        "Body Battery",
        "Energy reserve throughout the day",
    ),
    "hrv": DataType(
        "hrv", fetch_hrv, "HRV", "Nightly heart-rate variability"
    ),
    "stress": DataType("stress", fetch_stress, "Stress", "All-day stress level"),
    "respiration": DataType(
        "respiration", fetch_respiration, "Respiration", "Breathing rate during the day"
    ),
    "spo2": DataType(
        "spo2",
        fetch_spo2,
        "Pulse Ox (SpO2)",
        "Blood-oxygen saturation (needs a SpO2-capable watch)",
    ),
    "rhr": DataType(
        "rhr", fetch_rhr, "Resting HR", "Daily resting heart rate"
    ),
    "daily_summary": DataType(
        "daily_summary",
        fetch_daily_summary,
        "Daily summary",
        "Aggregate daily stats (calories, distance, intensity)",
    ),
    "intensity_minutes": DataType(
        "intensity_minutes",
        fetch_intensity_minutes,
        "Intensity minutes",
        "Moderate/vigorous minutes toward your weekly goal",
    ),
    "max_metrics": DataType(
        "max_metrics",
        fetch_max_metrics,
        "Max metrics",
        "Personal bests / max effort values",
    ),
    "training_readiness": DataType(
        "training_readiness",
        fetch_training_readiness,
        "Training readiness",
        "Readiness score combining recovery signals",
    ),
    "morning_training_readiness": DataType(
        "morning_training_readiness",
        fetch_morning_training_readiness,
        "Morning training readiness",
        "Readiness score measured on waking",
    ),
    "fitnessage": DataType(
        "fitnessage", fetch_fitnessage, "Fitness age", "Estimated fitness vs. biological age"
    ),
    "training_status": DataType(
        "training_status",
        fetch_training_status,
        "Training status",
        "VO2max, training load and status feedback",
    ),
    "lactate_threshold": DataType(
        "lactate_threshold",
        fetch_lactate_threshold,
        "Lactate threshold (running)",
        "Running LT heart rate, pace and running power",
    ),
    "sweat_loss": DataType(
        "sweat_loss", fetch_sweat_loss, "Sweat loss", "Estimated sweat loss during exercise"
    ),
    "weight": DataType(
        "weight", fetch_weight, "Weight", "Daily weigh-ins, BMI and body fat"
    ),
    "body_composition": DataType(
        "body_composition",
        fetch_body_composition,
        "Body composition",
        "Muscle/bone/water mass and body-fat percentage",
    ),
    "blood_pressure": DataType(
        "blood_pressure",
        fetch_blood_pressure,
        "Blood pressure",
        "Measured systolic / diastolic / pulse",
    ),
    "endurance_score": DataType(
        "endurance_score",
        fetch_endurance_score,
        "Endurance score",
        "Long-distance endurance rating",
    ),
    "hill_score": DataType(
        "hill_score", fetch_hill_score, "Hill score", "Climbing performance rating"
    ),
    "running_tolerance": DataType(
        "running_tolerance",
        fetch_running_tolerance,
        "Running tolerance",
        "How your body tolerates running load",
    ),
    "cycling_ftp": DataType(
        "cycling_ftp",
        fetch_cycling_ftp,
        "Cycling FTP",
        "Functional threshold power in watts (needs a cycling power meter)",
    ),
}

DEFAULT_TYPES: tuple[DataType, ...] = tuple(DATA_TYPES.values())

#: Every collectible daily metric type (fetchable + parseable), in registry
#: order. This is the exhaustive "what can be synced" set: per-user config
#: (and the server .env) exclude *from* this list, so anything not here is
#: never silently collected or silently missed. Activities, profile snapshots
#: (hr_zones / race_predictions) and activity weather are separate sync steps
#: and are not part of the daily-metrics list.
DAILY_TYPES: tuple[str, ...] = tuple(DATA_TYPES.keys())


def resolve_types(names: list[str] | None) -> list[DataType]:
    """Map data-type names to their registered DataType objects.

    Unknown names raise a ValueError listing valid options.
    """
    if not names:
        return list(DEFAULT_TYPES)

    resolved: list[DataType] = []
    for name in names:
        name = name.strip().lower()
        if name not in DATA_TYPES:
            raise ValueError(
                f"Unknown data type '{name}'. Valid: {', '.join(sorted(DATA_TYPES))}"
            )
        resolved.append(DATA_TYPES[name])
    return resolved