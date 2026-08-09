from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict[str, str]:
    """Load configuration from the .env file and environment.

    Environment variables take precedence over values in .env.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    email = os.getenv("GARMIN_EMAIL", "")
    password = os.getenv("GARMIN_PASSWORD", "")
    db_path = os.getenv("GARMIN_DB_PATH", str(PROJECT_ROOT / "garmin.db"))
    if not Path(db_path).is_absolute():
        db_path = str(PROJECT_ROOT / db_path)
    tokens_path = os.getenv("GARMIN_TOKENS_PATH", "")
    start_date = os.getenv("GARMIN_START_DATE", "")
    activity_freeze_days = os.getenv("GARMIN_ACTIVITY_FREEZE_DAYS", "7")
    excluded_data_types = os.getenv("GARMIN_EXCLUDED_DATA_TYPES", "")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url = os.getenv("LLM_BASE_URL", "")
    llm_api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "high")

    weather_home_lat = os.getenv("GARMIN_HOME_LAT", "")
    weather_home_lon = os.getenv("GARMIN_HOME_LON", "")

    def _optional_float(name: str, value: str) -> str:
        if value:
            try:
                float(value)
            except ValueError as exc:
                raise RuntimeError(
                    f"{name} must be a decimal number (or empty) in .env"
                ) from exc
        return value

    weather_home_lat = _optional_float("GARMIN_HOME_LAT", weather_home_lat)
    weather_home_lon = _optional_float("GARMIN_HOME_LON", weather_home_lon)

    memory_file = os.getenv(
        "GARMIN_MEMORY_FILE", str(PROJECT_ROOT / "agent_memory.json")
    )
    web_session_file = os.getenv(
        "GARMIN_WEB_SESSION_FILE", str(PROJECT_ROOT / "web_session.json")
    )
    trace_file = os.getenv("GARMIN_TRACE_FILE", str(PROJECT_ROOT / "ask_trace.jsonl"))

    if not email or not password:
        raise RuntimeError(
            "GARMIN_EMAIL and GARMIN_PASSWORD must be set in .env"
        )

    try:
        freeze_days = int(activity_freeze_days)
    except ValueError as exc:
        raise RuntimeError(
            "GARMIN_ACTIVITY_FREEZE_DAYS must be an integer in .env"
        ) from exc
    if freeze_days < 0:
        raise RuntimeError("GARMIN_ACTIVITY_FREEZE_DAYS must be >= 0 in .env")

    return {
        "email": email,
        "password": password,
        "db_path": db_path,
        "tokens_path": tokens_path,
        "start_date": start_date,
        "activity_freeze_days": freeze_days,
        "excluded_data_types": excluded_data_types,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "llm_reasoning_effort": llm_reasoning_effort,
        "memory_file": memory_file,
        "weather_home_lat": weather_home_lat,
        "weather_home_lon": weather_home_lon,
        "web_session_file": web_session_file,
        "trace_file": trace_file,
    }