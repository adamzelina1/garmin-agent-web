"""User identity + Garmin-account binding for the multi-user server.

Holds the ``users`` table (identity, separate from Garmin data), password
hashing (bcrypt), JWT issue/verify, and the signup flow that binds a Garmin
account by logging in once and encrypting the resulting OAuth tokens (AES-GCM
via :mod:`.crypto`).

Signup supports Garmin's two-step verification: when the Garmin login requires
a verification code, ``register`` keeps the (unconfirmed) account and returns an
MFA challenge token; the browser asks the user for the code and submits it to
``confirm_mfa``, which resumes the login and finishes the bind. The in-flight
Garmin client is held in memory (single-process server, keyed by the challenge,
with a short TTL), so nothing MFA-related is persisted unencrypted.
"""

from __future__ import annotations

import logging
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from garminconnect import Garmin, GarminConnectAuthenticationError

from ..db import open_pg_pool
from .crypto import Encryptor

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: How long a pending MFA challenge stays valid (seconds) before the user must
#: start registration again.
_MFA_TTL_S = 600


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_jwt(user_id: int, email: str, secret: str, ttl_hours: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + timedelta(hours=ttl_hours),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_jwt(token: str, secret: str) -> dict[str, Any]:
    return jwt.decode(token, secret, algorithms=["HS256"])


class UserStore:
    """Read/write access to the ``users`` identity table (no RLS on users)."""

    def __init__(self, url: str) -> None:
        self._pool = open_pg_pool(url, min_size=1, max_size=4)

    def _row(self, conn: Any, row: Any | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def create(
        self,
        *,
        email: str,
        password_hash: str,
        garmin_email: str,
        garmin_cred_enc: str,
    ) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO users (email, password_hash, garmin_email, "
                "garmin_cred_enc, created_at) VALUES (%s, %s, %s, %s, %s) "
                "RETURNING id",
                (email, password_hash, garmin_email, garmin_cred_enc, now_iso()),
            ).fetchone()
            return row["id"]

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = %s", (email,)
            ).fetchone()
            return self._row(conn, row)

    def get(self, user_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = %s", (user_id,)
            ).fetchone()
            return self._row(conn, row)

    def delete(self, user_id: int) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM users WHERE id = %s", (user_id,))

    def set_garmin_creds(
        self, user_id: int, garmin_cred_enc: str, tokens_enc: str
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE users SET garmin_cred_enc = %s, tokens_enc = %s "
                "WHERE id = %s",
                (garmin_cred_enc, tokens_enc, user_id),
            )

    def confirm(self, user_id: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE users SET confirmed = TRUE WHERE id = %s",
                (user_id,),
            )

    def reset_signup(
        self,
        user_id: int,
        *,
        email: str,
        password_hash: str,
        garmin_email: str,
        garmin_cred_enc: str,
    ) -> None:
        """Re-bind an unconfirmed account so a failed/expired MFA attempt can
        be retried (used by re-registration)."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE users SET email = %s, password_hash = %s, garmin_email = %s, "
                "garmin_cred_enc = %s, tokens_enc = NULL, confirmed = FALSE, "
                "sync_error = NULL, sync_fail_count = 0, rate_limit_until = NULL "
                "WHERE id = %s",
                (email, password_hash, garmin_email, garmin_cred_enc, user_id),
            )

    def set_sync_status(
        self,
        user_id: int,
        *,
        last_sync_at: str | None = None,
        error: str | None = None,
        fail_count: int | None = None,
        rate_limit_until: str | None = None,
    ) -> None:
        with self._pool.connection() as conn:
            sets = []
            params: list[Any] = []
            for col, val in (
                ("last_sync_at", last_sync_at),
                ("sync_error", error),
                ("sync_fail_count", fail_count),
                ("rate_limit_until", rate_limit_until),
            ):
                if val is not None:
                    sets.append(f"{col} = %s")
                    params.append(val)
            if not sets:
                return
            params.append(user_id)
            conn.execute(
                f"UPDATE users SET {', '.join(sets)} WHERE id = %s", params
            )

    def set_config(
        self,
        user_id: int,
        *,
        llm_api_key_enc: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        home_lat: str | None = None,
        home_lon: str | None = None,
        excluded_data_types: str | None = None,
        sync_start_date: str | None = None,
        auto_sync: bool | None = None,
    ) -> None:
        with self._pool.connection() as conn:
            sets = []
            params: list[Any] = []
            for col, val in (
                ("llm_api_key_enc", llm_api_key_enc),
                ("llm_base_url", llm_base_url),
                ("llm_model", llm_model),
                ("home_lat", home_lat),
                ("home_lon", home_lon),
                ("excluded_data_types", excluded_data_types),
                ("sync_start_date", sync_start_date),
                ("auto_sync", auto_sync),
            ):
                if val is not None:
                    sets.append(f"{col} = %s")
                    params.append(val)
            if not sets:
                return
            params.append(user_id)
            conn.execute(
                f"UPDATE users SET {', '.join(sets)} WHERE id = %s", params
            )

    def list_active(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM users WHERE active = TRUE ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    def list_auto_sync(self) -> list[dict[str, Any]]:
        """Users who opted into scheduled auto-sync (per-account setting)."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM users WHERE active = TRUE AND auto_sync = TRUE "
                "ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        self._pool.close()


class AuthError(Exception):
    """A user-facing auth/signup error with an HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class AuthService:
    """Register / login / token-verify for one server config."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.store = UserStore(cfg["db_url"])
        self.encryptor = Encryptor(cfg["enc_key"])
        self._secret = cfg["jwt_secret"]
        if not self._secret:
            raise RuntimeError("GARMIN_JWT_SECRET must be set in .env")
        self._ttl = cfg["jwt_ttl_hours"]
        # In-flight Garmin logins awaiting a verification code. Keyed by a
        # random challenge token; values hold the live Garmin client (whose MFA
        # session lives on the reusable requests.Session), the user/email to
        # finish binding, and a creation timestamp for the TTL.
        self._pending_mfa: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self.store.close()

    # -- MFA ------------------------------------------------------------------

    def _prune_pending_mfa(self) -> None:
        now = time.time()
        for key in [
            k for k, v in self._pending_mfa.items()
            if now - v["created_at"] > _MFA_TTL_S
        ]:
            self._pending_mfa.pop(key, None)

    # -- tokens ---------------------------------------------------------------

    def _token(self, user_id: int, email: str) -> str:
        return create_jwt(user_id, email, self._secret, self._ttl)

    def current_user(self, token: str) -> dict[str, Any]:
        try:
            payload = decode_jwt(token, self._secret)
        except jwt.InvalidTokenError as exc:
            raise AuthError(401, "invalid or expired token") from exc
        user = self.store.get(int(payload["sub"]))
        if not user or not user["active"]:
            raise AuthError(401, "account not found or disabled")
        return user

    # -- signup ---------------------------------------------------------------

    def register(
        self, *, email: str, password: str, garmin_email: str, garmin_password: str
    ) -> dict[str, Any]:
        """Create an account and bind a Garmin account.

        The account is created and the Garmin login is attempted immediately as
        part of signup. Invalid Garmin credentials
        (``GarminConnectAuthenticationError``) or a transient Garmin block
        (rate-limit/Cloudflare/429) both delete the account and fail signup
        with a clear message. If Garmin requires a verification code (two-step
        verification), the account is kept as unconfirmed and an ``mfa_required``
        response with a one-time ``challenge`` token is returned — the caller
        must then submit the code to :meth:`confirm_mfa` to finish the bind.
        """
        email = (email or "").strip().lower()
        if not _EMAIL_RE.match(email):
            raise AuthError(400, "a valid email address is required")
        if not password or len(password) < 8:
            raise AuthError(400, "password must be at least 8 characters")
        if not garmin_email or not garmin_password:
            raise AuthError(400, "Garmin credentials are required to bind an account")

        existing = self.store.get_by_email(email)
        if existing and existing["confirmed"]:
            raise AuthError(409, "an account with this email already exists")

        cred_enc = self.encryptor.encrypt(garmin_password)
        if existing:
            # Legacy unconfirmed account (abandoned MFA attempt): adopt it.
            user_id = existing["id"]
            self.store.reset_signup(
                user_id,
                email=email,
                password_hash=hash_password(password),
                garmin_email=garmin_email.strip(),
                garmin_cred_enc=cred_enc,
            )
        else:
            user_id = self.store.create(
                email=email,
                password_hash=hash_password(password),
                garmin_email=garmin_email.strip(),
                garmin_cred_enc=cred_enc,
            )

        client = Garmin(
            email=garmin_email,
            password=garmin_password,
            is_cn=False,
            return_on_mfa=True,
        )
        # Skip the widget+cffi strategy: it scrapes an HTML widget and can
        # falsely report "MFA required" for accounts without two-step
        # verification (see fetcher._SKIP_GARMIN_STRATEGIES).
        client.client.skip_strategies = ["widget+cffi"]
        try:
            mfa_status, _ = client.login()
        except GarminConnectAuthenticationError as exc:
            # Wrong Garmin credentials (or a locked account): fail signup.
            logger.warning("invalid Garmin credentials at signup for %s: %s", email, exc)
            self.store.delete(user_id)
            raise AuthError(400, f"invalid Garmin credentials: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - transient Garmin rate-limit / Cloudflare
            logger.warning("Garmin login failed at signup for %s: %s", email, exc)
            self.store.delete(user_id)
            raise AuthError(
                502,
                "Couldn't log in to Garmin right now — Garmin is likely "
                "rate-limiting or Cloudflare-blocking this server (common from "
                f"a server IP): {type(exc).__name__}. Nothing was created — "
                "please try again in a few minutes.",
            ) from exc

        if mfa_status == "needs_mfa":
            # Hold the in-flight client so the verification code can complete
            # the login in a follow-up request. The account stays unconfirmed.
            self._prune_pending_mfa()
            challenge = secrets.token_urlsafe(24)
            self._pending_mfa[challenge] = {
                "client": client,
                "user_id": user_id,
                "email": email,
                "cred_enc": cred_enc,
                "created_at": time.time(),
            }
            return {"status": "mfa_required", "challenge": challenge}

        self._finish_login(user_id, email, client, cred_enc)
        return {"status": "ok", "token": self._token(user_id, email)}

    def confirm_mfa(self, *, challenge: str, code: str) -> dict[str, Any]:
        """Complete a Garmin two-step login started by :meth:`register`.

        Looks up the in-flight client by the challenge token, submits the
        user's verification code, then finishes binding the account (encrypts
        and stores the resulting tokens, marks it confirmed) and returns a JWT.
        """
        code = (code or "").strip()
        if not challenge or not code:
            raise AuthError(400, "an MFA challenge and verification code are required")
        self._prune_pending_mfa()
        entry = self._pending_mfa.get(challenge)
        if not entry:
            raise AuthError(410, "the MFA session expired or is invalid — please register again")
        client = entry["client"]
        try:
            # The client_state returned at MFA time is None; resume_login ignores it.
            client.resume_login(None, code)
        except GarminConnectAuthenticationError as exc:
            # Wrong/expired code: keep the session so the user can retry.
            raise AuthError(400, f"verification code rejected by Garmin: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - transient Garmin rate-limit / Cloudflare
            raise AuthError(
                502,
                "couldn't verify the code right now — Garmin is likely "
                f"rate-limiting this server: {type(exc).__name__}. Please try again.",
            ) from exc
        self._pending_mfa.pop(challenge, None)
        self._finish_login(entry["user_id"], entry["email"], client, entry["cred_enc"])
        return {"status": "ok", "token": self._token(entry["user_id"], entry["email"])}

    def _finish_login(
        self, user_id: int, email: str, client: Any, cred_enc: str
    ) -> None:
        tokens = client.client.dumps()
        self.store.set_garmin_creds(user_id, cred_enc, self.encryptor.encrypt(tokens))
        self.store.confirm(user_id)

    # -- login ----------------------------------------------------------------

    def login(self, *, email: str, password: str) -> dict[str, Any]:
        user = self.store.get_by_email((email or "").strip().lower())
        if not user or not verify_password(password or "", user["password_hash"]):
            raise AuthError(401, "invalid email or password")
        if not user["active"]:
            raise AuthError(403, "account disabled")
        return {"status": "ok", "token": self._token(user["id"], user["email"])}

    # -- per-user config ------------------------------------------------------

    def get_user_config(self, user_id: int) -> dict[str, Any]:
        from ..datatypes import DATA_TYPES

        user = self.store.get(user_id) or {}
        return {
            "llm_api_key_set": bool(user.get("llm_api_key_enc")),
            "llm_base_url": user.get("llm_base_url") or "",
            "llm_model": user.get("llm_model") or "",
            "home_lat": user.get("home_lat") or "",
            "home_lon": user.get("home_lon") or "",
            "excluded_data_types": [
                n.strip()
                for n in (user.get("excluded_data_types") or "").split(",")
                if n.strip()
            ],
            "auto_sync": bool(user.get("auto_sync")),
            "sync_start_date": user.get("sync_start_date") or "",
            "available_data_types": [t.as_dict() for t in DATA_TYPES.values()],
        }

    def save_user_config(
        self,
        user_id: int,
        *,
        api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        home_lat: str | None = None,
        home_lon: str | None = None,
        excluded_data_types: list[str] | None = None,
        clear_api_key: bool = False,
        auto_sync: bool | None = None,
        sync_start_date: str | None = None,
    ) -> None:
        # A non-empty api_key replaces the stored key; an empty/absent value
        # leaves it untouched (the key is never echoed back to the browser, so
        # a blank save must not wipe it). Explicitly clear via clear_api_key.
        api_key_enc = "" if clear_api_key else None
        if api_key:
            api_key_enc = self.encryptor.encrypt(api_key)
        from ..datatypes import DAILY_TYPES

        valid_types = set(DAILY_TYPES)
        if excluded_data_types is not None:
            unknown = sorted(t for t in excluded_data_types if t not in valid_types)
            if unknown:
                raise AuthError(
                    400,
                    f"unknown data type(s) to exclude: {', '.join(unknown)}. "
                    f"Valid daily types: {', '.join(DAILY_TYPES)}",
                )
        # An absent start date leaves the stored one untouched; a blank value
        # clears it; a non-empty value must be a real YYYY-MM-DD date.
        effective_start: str | None = None
        if sync_start_date is not None:
            cleaned = sync_start_date.strip()
            if cleaned:
                from datetime import date as _date

                try:
                    _date.fromisoformat(cleaned)
                except ValueError as exc:
                    raise AuthError(
                        400, "sync start date must be YYYY-MM-DD (or blank)"
                    ) from exc
                effective_start = cleaned
            else:
                effective_start = ""
        self.store.set_config(
            user_id,
            llm_api_key_enc=api_key_enc,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            home_lat=home_lat,
            home_lon=home_lon,
            excluded_data_types=(
                ",".join(excluded_data_types) if excluded_data_types is not None else None
            ),
            auto_sync=auto_sync,
            sync_start_date=effective_start,
        )

    def user_llm_configured(self, user: dict[str, Any]) -> bool:
        if user.get("llm_api_key_enc"):
            return True
        base = user.get("llm_base_url") or ""
        return bool(base) and (
            "localhost" in base or "127.0.0.1" in base or "host.docker" in base
        )
