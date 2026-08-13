"""Multi-user server layer: auth (JWT), background sync, read-only ask API.

This package turns the single-user Garmin agent into a self-hosted,
multi-account service:

- ``auth``          — users table, bcrypt hashing, JWT, signup + MFA confirm
- ``crypto``        — AES-GCM encryption for per-user Garmin credentials/tokens
- ``sync_worker``   — per-user background sync (thread pool + APScheduler cron)
- ``app``           — FastAPI app + small JS frontend
- ``setup_db``      — bootstrap of the ``garmin_app`` / ``garmin_readonly`` roles
"""

from .app import create_app

__all__ = ["create_app"]
