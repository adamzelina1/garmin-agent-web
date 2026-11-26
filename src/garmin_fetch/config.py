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

    db_url = os.getenv("GARMIN_DB_URL", "")
    start_date = os.getenv("GARMIN_START_DATE", "")
    activity_freeze_days = os.getenv("GARMIN_ACTIVITY_FREEZE_DAYS", "7")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url = os.getenv("LLM_BASE_URL", "")
    llm_api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "high")

    # Per-account settings (Garmin login, home location, start date override,
    # excluded types) live in the ``users`` table; the CLI tools read the
    # GARMIN_LOCAL_USER_ID account's row like the server does.
    readonly_db_url = os.getenv("GARMIN_READONLY_DB_URL", "")
    admin_db_url = os.getenv("GARMIN_ADMIN_DB_URL", "")
    enc_key = os.getenv("GARMIN_ENC_KEY", "")
    jwt_secret = os.getenv("GARMIN_JWT_SECRET", "")
    jwt_ttl_hours = os.getenv("GARMIN_JWT_TTL_HOURS", "24")
    cron_token = os.getenv("GARMIN_CRON_TOKEN", "")
    sync_interval_min = os.getenv("GARMIN_SYNC_INTERVAL_MIN", "60")
    sync_max_workers = os.getenv("GARMIN_SYNC_MAX_WORKERS", "4")
    fetch_sleep_sec = os.getenv("GARMIN_FETCH_SLEEP_SEC", "0")
    auto_sync = os.getenv("GARMIN_AUTO_SYNC", "0")
    local_user_id = os.getenv("GARMIN_LOCAL_USER_ID", "1")

    try:
        jwt_ttl = int(jwt_ttl_hours)
        sync_interval = int(sync_interval_min)
        max_workers = int(sync_max_workers)
        local_uid = int(local_user_id)
    except ValueError as exc:
        raise RuntimeError(
            "GARMIN_JWT_TTL_HOURS / GARMIN_SYNC_INTERVAL_MIN / "
            "GARMIN_SYNC_MAX_WORKERS / GARMIN_LOCAL_USER_ID must be integers "
            "in .env"
        ) from exc
    if max_workers < 1:
        raise RuntimeError("GARMIN_SYNC_MAX_WORKERS must be >= 1 in .env")

    try:
        fetch_sleep = float(fetch_sleep_sec)
    except ValueError as exc:
        raise RuntimeError("GARMIN_FETCH_SLEEP_SEC must be a number (seconds) in .env") from exc
    if fetch_sleep < 0:
        raise RuntimeError("GARMIN_FETCH_SLEEP_SEC must be >= 0 in .env")

    auto_sync_enabled = auto_sync.strip().lower() in ("1", "true", "yes", "on")
    # Open by default; set to 0 once the people who should have accounts do.
    allow_signup = os.getenv("GARMIN_ALLOW_SIGNUP", "1").strip().lower() in (
        "1", "true", "yes", "on",
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
        "db_url": db_url,
        "start_date": start_date,
        "activity_freeze_days": freeze_days,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "llm_provider": llm_provider,
        "llm_reasoning_effort": llm_reasoning_effort,
        "readonly_db_url": readonly_db_url,
        "admin_db_url": admin_db_url,
        "enc_key": enc_key,
        "jwt_secret": jwt_secret,
        "jwt_ttl_hours": jwt_ttl,
        "cron_token": cron_token,
        "sync_interval_min": sync_interval,
        "sync_max_workers": max_workers,
        "fetch_sleep_sec": fetch_sleep,
        "auto_sync": auto_sync_enabled,
        "allow_signup": allow_signup,
        "local_user_id": local_uid,
    }