from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DataType:
    """A single type of daily Garmin data that the fetcher can sync.

    Only describes how to *fetch* raw data. Parsing into typed tables is a
    separate, later step.
    """

    name: str
    fetch: Callable[[Any, str], Any]

    def __str__(self) -> str:
        return self.name


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


def fetch_stats(client: Any, cdate: str) -> Any:
    return client.get_stats(cdate)


def fetch_intensity_minutes(client: Any, cdate: str) -> Any:
    return client.get_intensity_minutes_data(cdate)


def fetch_floors(client: Any, cdate: str) -> Any:
    return client.get_floors(cdate)


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


DATA_TYPES: dict[str, DataType] = {
    "heart_rate": DataType("heart_rate", fetch_heart_rate),
    "steps": DataType("steps", fetch_steps),
    "sleep": DataType("sleep", fetch_sleep),
    "body_battery": DataType("body_battery", fetch_body_battery),
    "hrv": DataType("hrv", fetch_hrv),
    "stress": DataType("stress", fetch_stress),
    "respiration": DataType("respiration", fetch_respiration),
    "spo2": DataType("spo2", fetch_spo2),
    "rhr": DataType("rhr", fetch_rhr),
    "stats": DataType("stats", fetch_stats),
    "intensity_minutes": DataType("intensity_minutes", fetch_intensity_minutes),
    "floors": DataType("floors", fetch_floors),
    "max_metrics": DataType("max_metrics", fetch_max_metrics),
    "training_readiness": DataType("training_readiness", fetch_training_readiness),
    "morning_training_readiness": DataType(
        "morning_training_readiness", fetch_morning_training_readiness
    ),
    "fitnessage": DataType("fitnessage", fetch_fitnessage),
    "training_status": DataType("training_status", fetch_training_status),
    "lactate_threshold": DataType("lactate_threshold", fetch_lactate_threshold),
}

DEFAULT_TYPES: tuple[DataType, ...] = tuple(DATA_TYPES.values())


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