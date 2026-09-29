"""Per-user agent state (session, memory, principles, trace) in ``user_state``.

The web agent keeps its conversation history, long-term memory and tool-call
trace as rows in the ``user_state`` table (``user_id`` + ``key`` + ``value``),
scoped by the same Row-Level Security as every data table, so each account
reads and writes only its own rows. Nothing user-facing is stored on disk.

Connections come from a shared pool and set ``app.user_id`` per transaction
(the ``true`` flag makes ``set_config`` transaction-scoped), exactly like the
read-only agent does, so RLS applies to every statement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

from ...db import open_pg_pool


_KEY_MEMORY = "memory"
_KEY_PRINCIPLES = "training_principles"
_KEY_SESSION = "web_session"
_KEY_TRACE = "trace"

_MAX_KEY = 80
_MAX_VALUE = 2000
#: Soft cap on the number of facts in the long-term profile. The whole profile
#: is injected into every turn, so growth is bounded: adding a new key past the
#: cap is rejected until the agent consolidates or forgets.
_MAX_FACTS = 40
#: Training principles are free-form markdown the athlete writes, so they get a
#: larger cap than a single memory value.
_MAX_PRINCIPLES = 4000


class UserState:
    """Per-user key/value rows in ``user_state`` (RLS-scoped)."""

    def __init__(self, url: str) -> None:
        self._pool = open_pg_pool(url, min_size=1, max_size=4)

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    def get(self, user_id: int, key: str) -> str | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "SELECT value FROM user_state WHERE user_id = %s AND key = %s",
                (user_id, key),
            ).fetchone()
        return row["value"] if row else None

    def set(self, user_id: int, key: str, value: str) -> None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "INSERT INTO user_state (user_id, key, value, updated_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (user_id, key) DO UPDATE SET "
                "value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
                (user_id, key, value, datetime.now(timezone.utc).isoformat()),
            )

    def delete(self, user_id: int, key: str) -> None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "DELETE FROM user_state WHERE user_id = %s AND key = %s",
                (user_id, key),
            )

    def get_session_messages(self, user_id: int) -> list[Any] | None:
        """Load the persisted conversation as pydantic-ai messages (or None)."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        raw = self.get(user_id, _KEY_SESSION)
        if not raw:
            return None
        return ModelMessagesTypeAdapter.validate_json(raw)

    def set_session_messages(self, user_id: int, messages: list[Any]) -> None:
        """Persist the conversation as pydantic-ai message JSON under a stable key."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        self.set(
            user_id,
            _KEY_SESSION,
            ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8"),
        )

    def clear_session(self, user_id: int) -> None:
        """Drop the stored conversation so the next turn starts a fresh session."""
        self.delete(user_id, _KEY_SESSION)

    def append_trace(self, user_id: int, record: dict[str, Any]) -> None:
        """Append one trace record to the user's stored trace list (JSON)."""
        raw = self.get(user_id, _KEY_TRACE)
        rows: list[dict[str, Any]] = []
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    rows = parsed
            except json.JSONDecodeError:
                rows = []
        rows.append(record)
        self.set(user_id, _KEY_TRACE, json.dumps(rows, ensure_ascii=False))

    def close(self) -> None:
        self._pool.close()


class PgMemory:
    """DB-backed long-term memory implementing the agent's memory interface.

    Facts are stored as one JSON dict under the ``memory`` key of the user's
    ``user_state`` row. Kept deliberately small: the whole profile is injected
    into every turn, so a soft cap (``max_facts``) forces consolidation rather
    than endless accumulation.
    """

    #: Exposed so the agent (and its prompt) can gauge how full memory is and
    #: nudge consolidation before the hard cap turns into a tool error.
    max_facts = _MAX_FACTS

    def __init__(self, state: UserState, user_id: int) -> None:
        self._state = state
        self._user_id = user_id

    def _read(self) -> dict[str, str]:
        raw = self._state.get(self._user_id, _KEY_MEMORY)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)
        }

    def _write(self, data: dict[str, str]) -> None:
        self._state.set(
            self._user_id,
            _KEY_MEMORY,
            json.dumps(data, indent=2, ensure_ascii=False),
        )

    @staticmethod
    def _clean_facts(facts: dict[str, Any]) -> dict[str, str]:
        """Strip and validate a ``{key: value}`` batch; raises ``ValueError``."""
        if not isinstance(facts, dict):
            raise ValueError("facts must be an object of {key: value}")
        clean: dict[str, str] = {}
        for key, value in facts.items():
            key, value = str(key).strip(), str(value).strip()
            if not key or len(key) > _MAX_KEY:
                raise ValueError(f"key must be 1..{_MAX_KEY} characters")
            if len(value) > _MAX_VALUE:
                raise ValueError(f"value must be at most {_MAX_VALUE} characters")
            clean[key] = value
        return clean

    @staticmethod
    def _cap_error(keys: Iterable[str]) -> ValueError:
        listed = ", ".join(sorted(keys)) or "(none)"
        return ValueError(
            f"memory is full ({_MAX_FACTS} facts max); consolidate overlapping "
            f"keys or forget stale ones before adding. Current keys: {listed}"
        )

    def get(self) -> dict[str, str]:
        return self._read()

    def remember(self, facts: dict[str, Any]) -> int:
        """Upsert a batch of facts; returns the new total count.

        An existing key is overwritten (never duplicated). Adding a *new* key
        when the profile is already at ``max_facts`` is rejected so the agent
        has to consolidate first.
        """
        clean = self._clean_facts(facts)
        data = self._read()
        existing = set(data)
        data.update(clean)
        if len(data) > _MAX_FACTS:
            raise self._cap_error(existing)
        self._write(data)
        return len(data)

    def forget(self, keys: Iterable[str]) -> list[str]:
        """Delete the given keys; returns the keys actually removed."""
        data = self._read()
        removed = [k for k in (str(k).strip() for k in keys) if k in data]
        if not removed:
            return []
        for key in removed:
            del data[key]
        self._write(data)
        return removed

    def replace(self, facts: dict[str, Any]) -> int:
        """Overwrite the whole profile with ``facts``; returns the new count.

        The consolidation path: the agent sends the complete intended fact set
        (overlapping keys merged, stale ones dropped). Rejected if it would
        exceed the cap.
        """
        clean = self._clean_facts(facts)
        if len(clean) > _MAX_FACTS:
            raise self._cap_error(clean)
        self._write(clean)
        return len(clean)


class PgPrinciples:
    """DB-backed training principles: the athlete's own standing directives.

    Deliberately *not* the same thing as memory. Memory holds facts the agent
    records itself and injects as background context; principles are authored by
    the athlete in Settings and injected as authoritative coaching guidance the
    agent must follow. Stored as plain text under the ``training_principles`` key
    of the user's ``user_state`` row; an empty value deletes the row.
    """

    def __init__(self, state: UserState, user_id: int) -> None:
        self._state = state
        self._user_id = user_id

    def get(self) -> str:
        return (self._state.get(self._user_id, _KEY_PRINCIPLES) or "").strip()

    def set(self, text: str) -> None:
        text = (text or "").strip()
        if len(text) > _MAX_PRINCIPLES:
            raise ValueError(
                f"training principles must be at most {_MAX_PRINCIPLES} characters"
            )
        if text:
            self._state.set(self._user_id, _KEY_PRINCIPLES, text)
        else:
            self._state.delete(self._user_id, _KEY_PRINCIPLES)
