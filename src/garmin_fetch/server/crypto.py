"""Symmetric encryption for per-user Garmin credentials and OAuth tokens.

AES-256-GCM with a random 12-byte nonce per encryption. The key is a base64
encoded 32-byte secret supplied via ``GARMIN_ENC_KEY`` (env / secret manager)
— never committed, never logged. Every stored blob is ``base64(nonce || ct)``;
an empty/missing key makes encryption refuse to start rather than silently
store plaintext.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_BYTES = 12


def _load_key(enc_key: str | None) -> bytes:
    value = enc_key or os.getenv("GARMIN_ENC_KEY", "")
    if not value:
        raise RuntimeError(
            "GARMIN_ENC_KEY must be set (base64 of a 32-byte key) to encrypt "
            "Garmin credentials/tokens"
        )
    try:
        key = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("GARMIN_ENC_KEY is not valid base64") from exc
    if len(key) != 32:
        raise RuntimeError("GARMIN_ENC_KEY must decode to exactly 32 bytes")
    return key


def generate_key() -> str:
    """Return a fresh base64 32-byte key for GARMIN_ENC_KEY."""
    return base64.b64encode(os.urandom(32)).decode()


class Encryptor:
    """AES-256-GCM encrypt/decrypt helper for one key."""

    def __init__(self, enc_key: str | None) -> None:
        self._key = _load_key(enc_key)

    def encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(NONCE_BYTES)
        ct = AESGCM(self._key).encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ct).decode()

    def decrypt(self, token: str) -> str:
        try:
            blob = base64.b64decode(token, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("ciphertext is not valid base64") from exc
        if len(blob) <= NONCE_BYTES:
            raise ValueError("ciphertext is truncated")
        nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
        try:
            return AESGCM(self._key).decrypt(nonce, ct, None).decode("utf-8")
        except InvalidTag as exc:
            raise ValueError("decryption failed (wrong key or tampered data)") from exc
