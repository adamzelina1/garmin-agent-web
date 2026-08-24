"""Background per-user sync: a bounded thread pool + APScheduler cron loop.

One user's long Garmin sync must not block another's, so every ``POST /sync``
(or the periodic cron) enqueues a job that runs in its own worker thread. A
user's job is deduplicated (only one queued/running at a time).

Phase 4 hardening lives here too: per-account rate limiting with exponential
backoff on the worker. Because the Garmin API is informal and tos-risky, any
sync failure backs the account off before it is retried (``sync_fail_count``
grows, ``rate_limit_until`` moves out) so a flapping account can't hammer the
endpoint. Credentials/tokens are decrypted only inside the worker, never
logged.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from garminconnect import GarminConnectAuthenticationError

from ..db import Database
from ..fetcher import DataFetcher, refresh_weather_forecast, sync_data

from .auth import UserStore, now_iso
from .crypto import Encryptor
from .state import TrainingPlanStore

logger = logging.getLogger(__name__)

_BACKOFF_BASE_MIN = 30
_BACKOFF_CAP_MIN = 480  # 8 hours


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class SyncManager:
    """Drives per-user syncs on a bounded thread pool with a cron loop."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.store = UserStore(cfg["db_url"])
        self.plan = TrainingPlanStore(cfg["db_url"])
        self.encryptor = Encryptor(cfg["enc_key"])
        self._pool = ThreadPoolExecutor(
            max_workers=cfg["sync_max_workers"], thread_name_prefix="garmin-sync"
        )
        self._lock = threading.Lock()
        self._running: set[int] = set()
        self._queued: set[int] = set()
        self._scheduler = BackgroundScheduler(daemon=True)

    def start(self) -> None:
        if not self.cfg.get("auto_sync"):
            logger.info(
                "sync worker started: auto-sync disabled (GARMIN_AUTO_SYNC=0); "
                "manual syncs still work via POST /sync"
            )
            return
        self._scheduler.add_job(
            self.cron_sync,
            "interval",
            minutes=self.cfg["sync_interval_min"],
            id="cron_sync",
            coalesce=True,
            max_instances=1,
        )
        self._scheduler.start()
        logger.info(
            "sync worker started: %d workers, cron every %d min",
            self.cfg["sync_max_workers"], self.cfg["sync_interval_min"],
        )

    def shutdown(self) -> None:
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001 - scheduler may already be stopped
            pass
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.store.close()
        self.plan.close()

    def is_running(self, user_id: int) -> bool:
        with self._lock:
            return user_id in self._running or user_id in self._queued

    def enqueue(self, user_id: int, force_reparse: bool = False) -> bool:
        """Queue a sync for ``user_id``; no-op (False) if already queued/running.

        ``force_reparse`` rebuilds the typed tables from raw on the next sync,
        ignoring incremental parse markers.
        """
        with self._lock:
            if user_id in self._running or user_id in self._queued:
                return False
            self._queued.add(user_id)
        self._pool.submit(self._run, user_id, force_reparse)
        return True

    def _run(self, user_id: int, force_reparse: bool = False) -> None:
        with self._lock:
            self._queued.discard(user_id)
            self._running.add(user_id)
        try:
            self.sync_user(user_id, force_reparse=force_reparse)
        finally:
            with self._lock:
                self._running.discard(user_id)

    def sync_user(
        self, user_id: int, *, force_reparse: bool = False
    ) -> dict[str, int] | None:
        """Run a full sync for one account using its stored, decrypted creds."""
        user = self.store.get(user_id)
        if not user or not user["active"]:
            logger.info("skipping sync for inactive user %s", user_id)
            return None
        if self._backoff_active(user):
            logger.info("skipping sync for user %s (rate-limited)", user_id)
            return None

        garmin_password = self.encryptor.decrypt(user["garmin_cred_enc"])
        token_string = (
            self.encryptor.decrypt(user["tokens_enc"]) if user.get("tokens_enc") else ""
        )
        fetcher = DataFetcher(
            user["garmin_email"], garmin_password, tokens_path=token_string,
            sleep_sec=self.cfg.get("fetch_sleep_sec", 0.0),
        )
        db = Database.from_url(self.cfg["db_url"], user_id=user_id)
        sync_cfg = dict(self.cfg)
        if user.get("excluded_data_types"):
            sync_cfg["excluded_data_types"] = user["excluded_data_types"]
        if user.get("sync_start_date"):
            # Per-user backfill start: overrides GARMIN_START_DATE for this
            # account. Empty -> the server-level default (or resume-from-last).
            sync_cfg["start_date"] = user["sync_start_date"]
        try:
            counts = sync_data(
                config=sync_cfg, db=db, fetcher=fetcher,
                full_reparse=force_reparse,
            )
            fresh_tokens = (
                fetcher.get_token_string() if fetcher.is_authenticated() else ""
            )
            if fresh_tokens and fresh_tokens != token_string:
                self.store.set_garmin_creds(
                    user_id, user["garmin_cred_enc"],
                    self.encryptor.encrypt(fresh_tokens),
                )
            self.store.confirm(user_id)  # Garmin bind now known-good
            self.store.set_sync_status(
                user_id,
                last_sync_at=now_iso(),
                error=None,
                fail_count=0,
                rate_limit_until=None,
            )
            logger.info("synced user %s: %s", user_id, counts)
            self._refresh_weather(user_id, user, db)
            self._refine_start(user_id, current_start=user.get("sync_start_date"))
            self._autocomplete_plan(user_id)
            return counts
        except GarminConnectAuthenticationError as exc:
            # Wrong Garmin credentials: a clear error, and no point hammering
            # with exponential backoff — retry on the normal cron cadence.
            logger.warning("invalid Garmin credentials for user %s: %s", user_id, exc)
            self.store.set_sync_status(
                user_id,
                error=f"invalid Garmin credentials: {exc}"[:500],
                fail_count=int(user.get("sync_fail_count") or 0) + 1,
                rate_limit_until=None,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - one account must not kill others
            self._record_failure(user_id, exc)
            return None
        finally:
            db.close()

    def _backoff_active(self, user: dict[str, Any]) -> bool:
        until = _parse_iso(user.get("rate_limit_until"))
        return until is not None and until > datetime.now(timezone.utc)

    def _refresh_weather(self, user_id: int, user: dict[str, Any], db: Any) -> None:
        """Store the next 16-day forecast on the already-open sync connection.

        Best-effort tied to a sync so the training-plan calendar's forecast only
        changes when a sync runs; a failure just logs and keeps the last stored
        forecast rather than failing the sync.
        """
        lat, lon = user.get("home_lat"), user.get("home_lon")
        if not lat or not lon:
            logger.info(
                "skipping weather-forecast refresh for user %s (no home location)",
                user_id,
            )
            return
        try:
            count = refresh_weather_forecast(db, float(lat), float(lon))
            if count:
                logger.info(
                    "refreshed %d weather-forecast day(s) for user %s",
                    count, user_id,
                )
        except Exception as exc:  # noqa: BLE001 - forecast must never break a sync
            logger.warning(
                "weather-forecast refresh failed for user %s: %s", user_id, exc
            )

    def _autocomplete_plan(self, user_id: int) -> None:
        """Mark planned workouts completed against the freshly synced activities.

        Best-effort: a failure only logs a warning and is retried on the next
        sync (the matcher is idempotent and only ever completes, never un-does).
        """
        try:
            result = self.plan.autocomplete(user_id)
        except Exception as exc:  # noqa: BLE001 - must never break a sync
            logger.warning(
                "training-plan autocomplete failed for user %s: %s", user_id, exc
            )
            return
        if result.get("completed"):
            logger.info(
                "autocompleted %d workout(s) for user %s",
                result["completed"], user_id,
            )

    def _refine_start(self, user_id: int, current_start: str | None) -> None:
        """Auto-refine the account's start date after a successful sync.

        Determines the first day that is followed by sustained good data (a
        sparse first day is confirmed as the start by the days after it), then
        deletes the raw + projected daily rows before the refined date and
        persists it as the account's ``sync_start_date`` so future backfills
        skip the junk prefix. The start is never moved earlier than the user's
        own configured choice, and a lone sparse day (with good data after it)
        is kept, not trimmed. Best-effort: a failure only logs a warning and is
        re-evaluated on the next sync.
        """
        from ..db import refine_user_start

        try:
            result = refine_user_start(
                self.cfg["db_url"], user_id, current_start=current_start
            )
        except Exception as exc:  # noqa: BLE001 - refine must never break a sync
            logger.warning(
                "start-date refine failed for user %s: %s", user_id, exc
            )
            return
        if not result:
            return
        recommended, pruned = result
        self.store.set_config(user_id, sync_start_date=recommended)
        logger.info(
            "refined start date for user %s to %s (pruned %d row(s))",
            user_id, recommended, pruned,
        )

    def _record_failure(self, user_id: int, exc: Exception) -> None:
        user = self.store.get(user_id) or {}
        count = int(user.get("sync_fail_count") or 0)
        msg = f"{type(exc).__name__}: {exc}"
        # A Garmin/Cloudflare SSO ban is IP-level and slow to lift; every extra
        # attempt just extends it. Detect the signature and back off straight to
        # the cap so the worker stops hammering a banned IP.
        lowered = msg.lower()
        ban = any(
            hint in lowered
            for hint in ("sso.garmin.com", "cloudflare", "banned", "429")
        )
        if ban:
            backoff_min = _BACKOFF_CAP_MIN
            error = (
                "Garmin is temporarily blocking this server's IP (Cloudflare/SSO "
                f"ban); will retry automatically in ~{backoff_min // 60}h"
            )
        else:
            backoff_min = min(_BACKOFF_BASE_MIN * (2 ** count), _BACKOFF_CAP_MIN)
            error = msg[:500]
        until = (datetime.now(timezone.utc) + timedelta(minutes=backoff_min)).isoformat()
        logger.warning(
            "sync failed for user %s (%d failures, backoff %d min): %s",
            user_id, count + 1, backoff_min, msg,
        )
        self.store.set_sync_status(
            user_id,
            error=error,
            fail_count=count + 1,
            rate_limit_until=until,
        )

    def cron_sync(self) -> int:
        """Enqueue one sync per active user, spaced across the cron interval.

        Enqueuing every opted-in account at the same tick makes the worker hit
        Garmin with a burst of logins/fetches from one IP — the fastest way to
        trigger a 429/Cloudflare block (Phase 4 hardening is a backstop, not a
        substitute). Instead, this walks the users who turned on auto-sync and
        enqueues each one on a stagger: the gap between accounts is
        ``interval_min / n_users`` (at least 1 minute), so with 2 users and a
        60-min interval they sync ~30 min apart rather than back-to-back. The
        stagger runs on a background daemon thread so the scheduler isn't
        blocked. Returns the number of auto-sync users scheduled.
        """
        users = self.store.list_auto_sync()
        if not users:
            return 0
        gap_min = max(1, self.cfg["sync_interval_min"] // len(users))
        threading.Thread(
            target=self._staggered_enqueue, args=(users, gap_min), daemon=True
        ).start()
        logger.info(
            "cron sync: %d auto-sync user(s), enqueuing them spread over ~%d min "
            "(gap %d min)",
            len(users), gap_min * len(users), gap_min,
        )
        return len(users)

    def _staggered_enqueue(self, users: list[dict[str, Any]], gap_min: int) -> None:
        """Enqueue each user one at a time, sleeping ``gap_min`` between them."""
        for i, user in enumerate(users):
            if i:
                time.sleep(gap_min * 60)
            self.enqueue(user["id"])
