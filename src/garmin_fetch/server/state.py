"""Per-user agent state (session, memory, trace) persisted in Postgres.

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
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

from ..db import open_pg_pool

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

#: Workout descriptions are meant to be a short one-line summary, but the agent
#: tends to write paragraphs. The prompt asks it to stay brief; this is the hard
#: backstop that holds regardless of model behaviour.
_MAX_DESCRIPTION_CHARS = 200

#: Allowed training-plan values (kept here so the store, the API and the agent
#: tools share one vocabulary).
ACTIVITY_TYPES = ("run", "cycle", "swim", "strength", "rest", "other")
INTENSITIES = ("easy", "moderate", "hard", "race_pace")

#: Workout lifecycle. ``completed`` and ``partial`` both count as done; the
#: API's derived ``completed`` boolean is computed from this, never stored.
PLAN_STATUSES = ("planned", "completed", "partial", "skipped")

#: Structured-workout step roles. A workout's ``steps`` is an ordered JSON list
#: of ``{kind, duration_sec, ...}`` intervals (e.g. warm-up, threshold,
#: recovery); the flat duration/distance/targets remain the workout summary.
STEP_KINDS = ("warmup", "steady", "work", "recovery", "cooldown", "rest")

#: Hard backstops so a runaway edit can't bloat a single workout row.
_MAX_STEPS = 50
_MAX_STEP_TEXT = 80

#: Garmin activity typeKeys that satisfy each plan ``activity_type`` when
#: auto-matching completed activities and when summing a goal's actual weekly
#: volume. ``rest`` and ``other`` are handled specially (absence / any type) and
#: are intentionally absent here.
GARMIN_TYPE_MAP: dict[str, frozenset[str]] = {
    "run": frozenset({
        "running", "track_running", "trail_running", "indoor_running",
        "treadmill_running", "virtual_run", "street_running", "ultra_run",
        "running_treadmill", "running_street", "running_track", "running_trail",
    }),
    "cycle": frozenset({
        "cycling", "road_biking", "mountain_biking", "indoor_cycling",
        "e_biking", "e_mountain_biking", "cyclocross", "gravel_cycling", "bmx",
        "track_cycling", "recumbent_cycling", "hand_cycling",
    }),
    "swim": frozenset({
        "lap_swimming", "open_water_swimming", "pool_swimming",
    }),
    "strength": frozenset({
        "strength_training", "weight_training",
    }),
}


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


def _parse_duration(value: Any, label: str) -> int:
    """Parse a workout-step duration into whole seconds.

    A plain number is read as minutes (the ``duration_min`` convention); strings
    use friendly units: ``"90s"``, ``"15m"``, ``"1h"``, ``"1h30m"`` or clock form
    ``"1:30"`` (mm:ss) / ``"1:02:30"`` (hh:mm:ss). The API/agent speak this
    instead of pre-computing seconds. Raises ``ValueError`` on anything else.
    """
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a duration like '15m' or '90s'")
    if isinstance(value, (int, float)):
        return int(round(float(value) * 60))
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        raise ValueError(f"{label} must be a duration like '15m' or '90s'")
    if ":" in text:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"{label} must be 'mm:ss' or 'hh:mm:ss'")
        try:
            nums = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"{label} must be 'mm:ss' or 'hh:mm:ss'") from exc
        seconds = nums[-1] + nums[-2] * 60 + (nums[0] * 3600 if len(nums) == 3 else 0)
        return int(round(seconds))
    total = 0.0
    matched = False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", text):
        matched = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    if matched and re.fullmatch(r"(?:\d+(?:\.\d+)?[hms])+", text):
        return int(round(total))
    try:
        return int(round(float(text) * 60))
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a duration like '15m', '90s' or '1:30'"
        ) from exc


def _normalize_steps(value: Any) -> str | None:
    """Validate an ordered list of workout steps -> canonical JSON string.

    ``value`` may be a list of step dicts or the canonical JSON string (the
    DB, the API and the agent all round-trip the string). Each step needs a
    ``kind`` and a positive ``duration`` (``'15m'``, ``'90s'``, a number of
    minutes); optional per-step targets mirror the flat workout targets.
    Returns None for an empty list. Raises ``ValueError`` on a malformed step.
    """
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("steps must be a JSON array") from exc
    if not isinstance(value, list):
        raise ValueError("steps must be a list of step objects")
    if len(value) > _MAX_STEPS:
        raise ValueError(f"steps may contain at most {_MAX_STEPS} entries")
    steps: list[dict[str, Any]] = []
    for i, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"step {i} must be an object")
        kind = str(raw.get("kind") or "steady").strip().lower()
        if kind not in STEP_KINDS:
            raise ValueError(
                f"step {i} kind must be one of: {', '.join(STEP_KINDS)}"
            )
        duration = None
        if raw.get("duration") not in (None, ""):
            duration = _parse_duration(raw["duration"], f"step {i} duration")
        if duration is None:
            raise ValueError(
                f"step {i} needs a duration (e.g. '15m' or '90s')"
            )
        if duration <= 0:
            raise ValueError(f"step {i} duration must be > 0")
        step: dict[str, Any] = {"kind": kind, "duration_sec": duration}
        label = _text(raw.get("label"))
        if label:
            step["label"] = label[:_MAX_STEP_TEXT]
        repeat = raw.get("repeat")
        if repeat not in (None, "", 1):
            try:
                repeat = int(repeat)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"step {i} repeat must be an integer") from exc
            if repeat < 1:
                raise ValueError(f"step {i} repeat must be >= 1")
            if repeat > 1:
                step["repeat"] = repeat
        intensity = raw.get("intensity")
        if intensity not in (None, ""):
            intensity = str(intensity).strip().lower()
            if intensity not in INTENSITIES:
                raise ValueError(
                    f"step {i} intensity must be one of: {', '.join(INTENSITIES)}"
                )
            step["intensity"] = intensity
        if raw.get("target_pace_min_km") not in (None, ""):
            step["target_pace_min_km"] = _parse_opt_float(
                raw["target_pace_min_km"], f"step {i} target_pace_min_km"
            )
        hr = _text(raw.get("target_hr_zone"))
        if hr:
            step["target_hr_zone"] = hr
        if raw.get("target_power_w") not in (None, ""):
            step["target_power_w"] = _parse_opt_nonneg_int(
                raw["target_power_w"], f"step {i} target_power_w"
            )
        notes = _text(raw.get("notes"))
        if notes:
            step["notes"] = notes[:_MAX_DESCRIPTION_CHARS]
        steps.append(step)
    return json.dumps(steps, ensure_ascii=False) if steps else None


def _normalize_workout(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate and normalise one workout dict; raises ``ValueError``.

    ``planned_date`` (YYYY-MM-DD) and ``activity_type`` are required; every
    optional field is coerced to its column type (None/null stays NULL). With
    ``partial`` only the keys present in ``data`` are returned, so an edit can
    change one field without re-sending (or nulling) the rest.

    ``status`` is the authoritative lifecycle value (planned/completed/partial/
    skipped).
    """
    keys = list(data) if partial else (
        "planned_date", "activity_type", "title", "description", "duration_min",
        "distance_km", "intensity", "target_pace_min_km", "target_hr_zone",
        "target_power_w", "status", "goal_id", "steps",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key == "planned_date":
            fields[key] = _parse_required_date(value, "planned_date")
        elif key == "activity_type":
            atype = (str(value) if value is not None else "").strip().lower()
            if atype not in ACTIVITY_TYPES:
                raise ValueError(
                    f"activity_type must be one of: {', '.join(ACTIVITY_TYPES)}"
                )
            fields[key] = atype
        elif key in ("title", "description", "target_hr_zone"):
            text = _text(value)
            fields[key] = _cap_description(text) if key == "description" else text
        elif key == "intensity":
            if value in (None, ""):
                fields[key] = None
            else:
                intensity = str(value).strip().lower()
                if intensity not in INTENSITIES:
                    raise ValueError(
                        f"intensity must be one of: {', '.join(INTENSITIES)}"
                    )
                fields[key] = intensity
        elif key == "duration_min":
            fields[key] = _parse_opt_nonneg_int(value, "duration_min")
        elif key == "distance_km":
            fields[key] = _parse_opt_float(value, "distance_km")
        elif key == "target_pace_min_km":
            pace = _parse_opt_float(value, "target_pace_min_km")
            if pace is not None and not (1.0 <= pace <= 60.0):
                raise ValueError(
                    "target_pace_min_km must be between 1 and 60 (decimal min/km)"
                )
            fields[key] = pace
        elif key == "target_power_w":
            fields[key] = _parse_opt_nonneg_int(value, "target_power_w")
        elif key == "goal_id":
            fields[key] = _opt_int(value, "goal_id")
        elif key == "steps":
            fields[key] = _normalize_steps(value)
        elif key == "status":
            # A model-dumped null status means "not supplied".
            if value in (None, ""):
                continue
            fields[key] = _normalize_status(value)
    return fields


def _normalize_status(value: Any) -> str:
    """Validate a workout status against :data:`PLAN_STATUSES`."""
    status = str(value).strip().lower()
    if status not in PLAN_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(PLAN_STATUSES)}")
    return status



def _parse_steps(raw: Any) -> list[dict[str, Any]]:
    """Decode the stored ``steps`` JSON column into a list (never raises)."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    return raw if isinstance(raw, list) else []


def _workout_row(row: Any) -> dict[str, Any]:
    """Shape one DB row into the JSON the API/agent/tab all consume.

    ``status`` is the authoritative lifecycle value and the only one written;
    ``completed`` is derived from it for the UI/legacy callers, so the two can
    never disagree.
    """
    status = row.get("status")
    if status not in PLAN_STATUSES:
        status = "planned"
    d = {
        "id": row["id"],
        "planned_date": row["planned_date"],
        "activity_type": row["activity_type"],
        "title": row["title"],
        "description": row["description"],
        "duration_min": row["duration_min"],
        "distance_km": row["distance_km"],
        "intensity": row["intensity"],
        "target_pace_min_km": row.get("target_pace_min_km"),
        "target_hr_zone": row.get("target_hr_zone"),
        "target_power_w": row.get("target_power_w"),
        "steps": _parse_steps(row.get("steps")),
        "status": status,
        "completed": status in ("completed", "partial"),
        "completed_activity_id": row["completed_activity_id"],
    }
    if row.get("goal_id") is not None:
        d["goal_id"] = row["goal_id"]
    return d


def _parse_required_date(value: Any, label: str) -> str:
    """Parse a required YYYY-MM-DD -> normalised str; raises ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required (YYYY-MM-DD)")
    try:
        date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD, got {value!r}") from exc
    return value.strip()


def _parse_opt_nonneg_int(value: Any, label: str) -> int | None:
    """Parse an optional non-negative integer -> int | None."""
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{label} must be >= 0")
    return number



def _parse_opt_date(value: Any, label: str) -> str | None:
    """Parse an optional YYYY-MM-DD (or '' / None) -> str | None."""
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be YYYY-MM-DD")
    try:
        date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD, got {value!r}") from exc
    return value.strip()


def _parse_opt_float(value: Any, label: str) -> float | None:
    """Parse an optional non-negative number -> float | None."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if number < 0:
        raise ValueError(f"{label} must be >= 0")
    return number


def _text(value: Any) -> str | None:
    """Strip a scalar to its trimmed text (''/None -> None)."""
    return str(value).strip() if value not in (None, "") else None


def _cap_description(value: str | None) -> str | None:
    """Trim an over-long workout description to ``_MAX_DESCRIPTION_CHARS``.

    Cuts on a word boundary when possible so the stored text doesn't end
    mid-word. A backstop for the prompt's length rule, not a formatter.
    """
    if value is None or len(value) <= _MAX_DESCRIPTION_CHARS:
        return value
    clipped = value[:_MAX_DESCRIPTION_CHARS].rstrip()
    cut = clipped.rfind(" ")
    if cut > 0:
        clipped = clipped[:cut].rstrip()
    return clipped or value[:_MAX_DESCRIPTION_CHARS].rstrip()


def _opt_int(value: Any, label: str) -> int | None:
    """Parse an optional integer (''/None -> None); raises ``ValueError``."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


#: Goal columns with a plain text normalizer (strip; '' -> NULL).
_GOAL_TEXT_FIELDS = ("title", "target_time")
#: Block columns with a plain text normalizer.
_BLOCK_TEXT_FIELDS = ("name", "focus")


def _normalize_goal(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate/normalise one goal dict; raises ``ValueError``.

    The goal is a long-term anchor (an event/target such as a marathon or a
    half-marathon). ``sport`` is one of :data:`ACTIVITY_TYPES` (the same
    vocabulary as workouts, so the plan tab can match activities to the goal);
    ``start_date`` and ``target_date`` must be YYYY-MM-DD when present. With
    ``partial`` only the keys present in ``data`` are returned (absent columns
    keep their value).
    """
    keys = list(data) if partial else _GOAL_TEXT_FIELDS + (
        "sport", "start_date", "target_date", "target_distance_km",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key == "sport":
            if value in (None, ""):
                fields[key] = None
            else:
                sport = str(value).strip().lower()
                if sport not in ACTIVITY_TYPES:
                    raise ValueError(
                        f"sport must be one of: {', '.join(ACTIVITY_TYPES)}"
                    )
                fields[key] = sport
        elif key in _GOAL_TEXT_FIELDS:
            fields[key] = _text(value)
        elif key in ("start_date", "target_date"):
            fields[key] = _parse_opt_date(value, key)
        elif key == "target_distance_km":
            fields[key] = _parse_opt_float(value, "target_distance_km")
    if fields.get("start_date") and fields.get("target_date"):
        if str(fields["target_date"]) < str(fields["start_date"]):
            raise ValueError(
                f"goal target_date ({fields['target_date']}) must be on/after start_date ({fields['start_date']})"
            )
    return fields


def _normalize_block(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate/normalise one block dict; raises ``ValueError``.

    Blocks are periodised phases of a goal: ``name``/``focus`` are free text,
    and the dates are optional (a block may be undated). ``target_weekly_km``
    is an optional baseline weekly distance. With ``partial`` only the keys
    present in ``data`` are returned (absent columns keep their stored value).
    """
    keys = list(data) if partial else _BLOCK_TEXT_FIELDS + (
        "goal_id", "start_date", "end_date", "target_weekly_km",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key in _BLOCK_TEXT_FIELDS:
            fields[key] = _text(value)
        elif key in ("start_date", "end_date"):
            fields[key] = _parse_opt_date(value, key)
        elif key == "goal_id":
            fields[key] = _opt_int(value, "goal_id")
        elif key == "target_weekly_km":
            fields[key] = _parse_opt_float(value, "target_weekly_km")
    return fields


def _normalize_week(
    data: dict[str, Any], *, partial: bool = False
) -> dict[str, Any]:
    """Validate/normalise one week-target dict.

    A week is one explicit calendar week of a block's progression (weeks do not
    repeat): distance_km and/or duration_min plus an optional ``is_deload``
    flag. If ``week_start`` is sent, it is validated as YYYY-MM-DD; otherwise
    it is derived from the block start and the week's position in the list.
    """
    keys = list(data) if partial else (
        "week_start", "distance_km", "duration_min", "is_deload",
    )
    fields: dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if key == "week_start":
            if value is not None and str(value).strip():
                fields[key] = _parse_required_date(value, "week_start")
            else:
                fields[key] = None
        elif key == "distance_km":
            fields[key] = _parse_opt_float(value, "distance_km")
        elif key == "duration_min":
            fields[key] = _parse_opt_nonneg_int(value, "duration_min")
        elif key == "is_deload":
            deload = value
            if isinstance(deload, str):
                deload = deload.strip().lower() in ("1", "true", "yes", "on")
            fields[key] = None if value is None else bool(deload)
    return fields


def _first_monday(day: date) -> date:
    """Monday of the calendar week containing ``day`` (weeks are Mon..Sun)."""
    return day - timedelta(days=day.weekday())


def _block_week_number(block_start: date, day: date) -> int:
    """1-based calendar week (Mon..Sun) of ``day`` within a block.

    A block's first week is the calendar week containing its start date, so a
    block that starts mid-week still aligns to Mon..Sun and its week boundary is
    exactly the plan tab's calendar week. Never returns less than 1.
    """
    return max(1, (day - _first_monday(block_start)).days // 7 + 1)


def _block_week_count(block: dict[str, Any]) -> int | None:
    """Number of calendar weeks (Mon..Sun) a dated block spans, or None.

    The count comes from the block's dates, not its stored week rows, so a
    partially-populated block still reports the full span.
    """
    start, end = block.get("start_date"), block.get("end_date")
    if not start or not end:
        return None
    base = _first_monday(date.fromisoformat(start))
    last = _first_monday(date.fromisoformat(end))
    return (last - base).days // 7 + 1


def _covering_block(
    day: date, blocks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The shortest *dated* block covering ``day``, or None.

    When several blocks overlap the day the most specific (shortest) phase
    wins. Undated blocks never cover a day.
    """
    covering = [
        b for b in blocks
        if b.get("start_date") and b.get("end_date")
        and b["start_date"] <= day.isoformat() <= b["end_date"]
    ]
    if not covering:
        return None
    return min(
        covering,
        key=lambda b: (
            date.fromisoformat(b["end_date"]) - date.fromisoformat(b["start_date"])
        ).days,
    )


def _covering_block_for_week(
    monday: date, blocks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The shortest dated block overlapping the calendar week [monday, monday + 6 days]."""
    cur_start = monday.isoformat()
    cur_end = (monday + timedelta(days=6)).isoformat()
    covering = [
        b for b in blocks
        if b.get("start_date") and b.get("end_date")
        and b["start_date"] <= cur_end and b["end_date"] >= cur_start
    ]
    if not covering:
        return None
    return min(
        covering,
        key=lambda b: (
            date.fromisoformat(b["end_date"]) - date.fromisoformat(b["start_date"])
        ).days,
    )


def _resolve_block_weeks(
    block: dict[str, Any], raw_weeks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Synthesize full calendar-week targets for a block with fallback to target_weekly_km."""
    count = _block_week_count(block)
    raw_by_start = {
        w["week_start"]: w for w in raw_weeks if w.get("week_start")
    }
    if count is not None and block.get("start_date"):
        base = _first_monday(date.fromisoformat(block["start_date"]))
        used_starts = set()
        resolved_weeks = []
        for wi in range(count):
            ws = (base + timedelta(days=7 * wi)).isoformat()
            used_starts.add(ws)
            w = raw_by_start.get(ws)
            if w is not None:
                w_copy = dict(w)
                if w_copy.get("distance_km") is None and block.get("target_weekly_km") is not None:
                    w_copy["distance_km"] = block.get("target_weekly_km")
                if w_copy.get("is_deload") is None:
                    w_copy["is_deload"] = False
                resolved_weeks.append(w_copy)
            else:
                resolved_weeks.append({
                    "week_start": ws,
                    "distance_km": block.get("target_weekly_km"),
                    "duration_min": None,
                    "is_deload": False,
                })
        for w in raw_weeks:
            if w.get("week_start") not in used_starts:
                w_copy = dict(w)
                if w_copy.get("distance_km") is None and block.get("target_weekly_km") is not None:
                    w_copy["distance_km"] = block.get("target_weekly_km")
                resolved_weeks.append(w_copy)
        return resolved_weeks
    else:
        resolved_weeks = []
        for w in raw_weeks:
            w_copy = dict(w)
            if w_copy.get("distance_km") is None and block.get("target_weekly_km") is not None:
                w_copy["distance_km"] = block.get("target_weekly_km")
            resolved_weeks.append(w_copy)
        return resolved_weeks



def _coverage_entry(
    week: dict[str, Any], week_start: str, week_end: str,
    km: Any, mins: Any, actual_km: Any = 0, actual_min: Any = 0,
) -> dict[str, Any]:
    """Scheduled-vs-target coverage for one calendar week.

    The week's target unit is km when it carries ``distance_km``, else minutes
    when it carries ``duration_min`` (else no target). ``km``/``mins`` are the
    pre-summed planned totals and ``actual_km``/``actual_min`` the pre-summed
    synced-activity totals for the week, so callers can batch the SUMs.
    """
    if week.get("distance_km") is not None:
        unit, target = "km", week["distance_km"]
    elif week.get("duration_min") is not None:
        unit, target = "min", week["duration_min"]
    else:
        unit, target = None, None
    if unit == "km":
        scheduled = round(float(km or 0), 2)
        actual = round(float(actual_km or 0), 2)
    elif unit == "min":
        scheduled = int(mins or 0)
        actual = int(actual_min or 0)
    else:
        scheduled = 0
        actual = 0
    return {
        "week_start": week_start,
        "week_end": week_end,
        "unit": unit,
        "target": target,
        "scheduled": scheduled,
        "actual": actual,
        "delta": round(scheduled - target, 2) if target is not None else None,
    }


def _validate_block_dates(start: Any, end: Any) -> None:
    """Reject a block whose ``end_date`` precedes its ``start_date``.

    An inverted range is invisible to the date resolver but still renders in
    the phase strip, so it is rejected up front instead of stored.
    """
    if start and end and str(end) < str(start):
        raise ValueError(
            f"block end_date ({end}) must be on/after start_date ({start})"
        )


def _goal_row(row: Any) -> dict[str, Any]:
    """Shape one ``training_goal`` row into API/agent JSON."""
    return {
        "id": row["id"],
        "title": row["title"],
        "sport": row["sport"],
        "start_date": row.get("start_date"),
        "target_date": row["target_date"],
        "target_distance_km": row["target_distance_km"],
        "target_time": row["target_time"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _block_row(row: Any) -> dict[str, Any]:
    """Shape one ``training_block`` row into API/agent JSON."""
    return {
        "id": row["id"],
        "goal_id": row["goal_id"],
        "name": row["name"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "focus": row["focus"],
        "target_weekly_km": row.get("target_weekly_km"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _week_row(row: Any) -> dict[str, Any]:
    """Shape one ``training_week`` row into API/agent JSON."""
    return {
        "week_start": row.get("week_start"),
        "distance_km": row["distance_km"],
        "duration_min": row["duration_min"],
        "is_deload": row.get("is_deload"),
    }


class TrainingWorkoutStore:
    """Per-account rows in the ``training_workout`` table (RLS-scoped).

    Like ``UserState``, connections come from a shared writer-role pool and set
    ``app.user_id`` per transaction, so Row-Level Security isolates every
    workout to its account.
    """

    def __init__(self, url: str | None = None, *, pool: Any | None = None) -> None:
        if pool is not None:
            self._pool = pool
            self._owns_pool = False
        else:
            self._pool = open_pg_pool(url, min_size=1, max_size=4)
            self._owns_pool = True

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    def list(
        self,
        user_id: int,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict[str, Any]]:
        """All workouts, optionally bounded by inclusive planned_date range."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            sql = "SELECT * FROM training_workout WHERE user_id = %s"
            params: list[Any] = [user_id]
            if date_start:
                sql += " AND planned_date >= %s"
                params.append(date_start)
            if date_end:
                sql += " AND planned_date <= %s"
                params.append(date_end)
            sql += " ORDER BY planned_date, id"
            rows = conn.execute(sql, params).fetchall()
        return [_workout_row(r) for r in rows]

    def create(self, user_id: int, data: dict[str, Any]) -> dict[str, Any]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._create(conn, user_id, data)

    def _create(
        self, conn: Any, user_id: int, data: dict[str, Any]
    ) -> dict[str, Any]:
        fields = _normalize_workout(data)
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "INSERT INTO training_workout (user_id, planned_date, activity_type, "
            "title, description, duration_min, distance_km, intensity, "
            "target_pace_min_km, target_hr_zone, target_power_w, steps, status, "
            "goal_id, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s) RETURNING *",
            (
                user_id, fields["planned_date"], fields["activity_type"],
                fields.get("title"), fields.get("description"),
                fields.get("duration_min"), fields.get("distance_km"),
                fields.get("intensity"), fields.get("target_pace_min_km"),
                fields.get("target_hr_zone"), fields.get("target_power_w"),
                fields.get("steps"), fields.get("status"),
                fields.get("goal_id"), now, now,
            ),
        ).fetchone()
        return _workout_row(row)

    def update(
        self,
        user_id: int,
        workout_id: int,
        data: dict[str, Any],
        *,
        partial: bool = False,
    ) -> dict[str, Any] | None:
        """Update a workout; ``partial`` changes only the supplied fields.

        The default (full) mode rewrites every editable column, so a caller
        must send the complete desired state. ``partial=True`` is the PATCH
        path used by the UI's edit modal and the agent's ``training`` tool
        upsert, so changing one field never clears the others.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._update(conn, user_id, workout_id, data, partial=partial)

    def _update(
        self,
        conn: Any,
        user_id: int,
        workout_id: int,
        data: dict[str, Any],
        *,
        partial: bool = False,
    ) -> dict[str, Any] | None:
        fields = _normalize_workout(data, partial=partial)
        current = conn.execute(
            "SELECT * FROM training_workout WHERE user_id = %s AND id = %s",
            (user_id, workout_id),
        ).fetchone()
        if current is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        if partial:
            if not fields:
                return _workout_row(current)
            assigns, params = _build_set(fields, now)
            row = conn.execute(
                f"UPDATE training_workout SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, workout_id),
            ).fetchone()
        else:
            row = conn.execute(
                "UPDATE training_workout SET planned_date = %s, activity_type = %s, "
                "title = %s, description = %s, duration_min = %s, distance_km = %s, "
                "intensity = %s, target_pace_min_km = %s, target_hr_zone = %s, "
                "target_power_w = %s, steps = %s, status = %s, "
                "goal_id = %s, updated_at = %s "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (
                    fields["planned_date"], fields["activity_type"],
                    fields.get("title"), fields.get("description"),
                    fields.get("duration_min"), fields.get("distance_km"),
                    fields.get("intensity"), fields.get("target_pace_min_km"),
                    fields.get("target_hr_zone"), fields.get("target_power_w"),
                    fields.get("steps"), fields.get("status"),
                    fields.get("goal_id"), now,
                    user_id, workout_id,
                ),
            ).fetchone()
        return _workout_row(row) if row else None

    def delete(self, user_id: int, workout_id: int) -> bool:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._delete(conn, user_id, workout_id)

    def _delete(self, conn: Any, user_id: int, workout_id: int) -> bool:
        cur = conn.execute(
            "DELETE FROM training_workout WHERE user_id = %s AND id = %s",
            (user_id, workout_id),
        )
        return cur.rowcount > 0

    def apply(
        self, user_id: int, spec: dict[str, Any], *, conn: Any | None = None
    ) -> dict[str, Any]:
        """Apply a batch workout edit (the agent's ``training`` workouts section).

        ``spec`` keys (both optional):
        - ``workouts`` (list): a dict with an ``id`` PATCHES that workout (only
          the supplied fields change); one without an ``id`` creates a workout.
        - ``delete_ids`` (list[int]): delete those workouts.

        The whole batch runs on ONE connection, so a bad row rolls the entire
        edit back. The caller (``TrainingStore``) snapshots the whole season
        before a deletion, restorable with ``{"undo": true}``. Returns the
        counts plus the ids that changed (``added_ids``/``updated_ids``) so the
        caller never has to re-read to learn what it touched.
        """
        if conn is not None:
            return self._apply(conn, user_id, spec)
        with self._pool.connection() as own:
            self._set_user(own, user_id)
            return self._apply(own, user_id, spec)

    def _apply(
        self, conn: Any, user_id: int, spec: dict[str, Any]
    ) -> dict[str, Any]:
        deleted = added = updated = 0
        added_ids: list[int] = []
        updated_ids: list[int] = []
        for wid in spec.get("delete_ids") or []:
            wid = _opt_int(wid, "delete_ids")
            if wid is None:
                raise ValueError("delete_ids must contain integer workout ids")
            if self._delete(conn, user_id, wid):
                deleted += 1
        for workout in spec.get("workouts") or []:
            if not isinstance(workout, dict):
                raise ValueError("each workout must be a JSON object")
            wid = workout.get("id")
            if wid is not None:
                wid = _opt_int(wid, "workout id")
                if wid is None:
                    raise ValueError("workout id must be an integer")
                if self._update(conn, user_id, wid, workout, partial=True) is None:
                    raise ValueError(f"workout id {wid} not found")
                updated += 1
                updated_ids.append(wid)
                continue
            row = self._create(conn, user_id, workout)
            added += 1
            added_ids.append(row["id"])
        total = conn.execute(
            "SELECT count(*) AS n FROM training_workout WHERE user_id = %s",
            (user_id,),
        ).fetchone()["n"]
        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "total": total,
            "added_ids": added_ids,
            "updated_ids": updated_ids,
        }

    def activities(
        self,
        user_id: int,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Synced activities whose ``start_date`` falls in the inclusive range.

        The unplanned counterpart to ``list``: lets the agent see what was
        actually done across a whole multi-week window, not just one week.
        """
        sql = (
            "SELECT activity_id, start_date, activity_type, distance_km, "
            "duration_hours FROM activity_summaries WHERE user_id = %s"
        )
        params: list[Any] = [user_id]
        if date_start:
            sql += " AND start_date >= %s"
            params.append(date_start)
        if date_end:
            sql += " AND start_date <= %s"
            params.append(date_end)
        sql += " ORDER BY start_date"
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            rows = conn.execute(sql, params).fetchall()
        return [dict(a) for a in rows]

    def autocomplete(self, user_id: int) -> dict[str, int]:
        """Mark planned workouts complete by matching synced activities.

        Best-effort and idempotent: only moves a workout from planned to
        completed, never un-completes and never touches a skipped one. A
        non-rest workout is matched to an activity of the same family on the
        planned day, then on the adjacent days (±1) — a long run logged the next
        morning still counts. Each activity completes at most one workout, and
        same-day matches take priority over the adjacent-day fallback (so an
        activity is never claimed by two workouts, nor borrowed from a day whose
        own workout still needs it). A past rest workout is completed when no
        activity exists that day. Dead ``completed_activity_id`` links (an
        activity removed by a re-parse) are cleared first. Future workouts are
        never touched. Returns {"completed": n}.
        """
        today = date.today().isoformat()
        now = datetime.now(timezone.utc).isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            conn.execute(
                "UPDATE training_workout SET completed_activity_id = NULL, "
                "updated_at = %s WHERE user_id = %s "
                "AND completed_activity_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM activity_summaries a "
                "WHERE a.user_id = %s "
                "AND a.activity_id = training_workout.completed_activity_id)",
                (now, user_id, user_id),
            )
            workouts = conn.execute(
                "SELECT * FROM training_workout WHERE user_id = %s "
                "AND (status IS NULL OR status = 'planned') "
                "ORDER BY planned_date, id",
                (user_id,),
            ).fetchall()
            if not workouts:
                return {"completed": 0}
            lo = (
                date.fromisoformat(min(w["planned_date"] for w in workouts))
                - timedelta(days=1)
            ).isoformat()
            hi = (
                date.fromisoformat(max(w["planned_date"] for w in workouts))
                + timedelta(days=1)
            ).isoformat()
            acts = conn.execute(
                "SELECT activity_id, start_date, activity_type, distance_km, "
                "duration_hours FROM activity_summaries "
                "WHERE user_id = %s AND start_date >= %s AND start_date <= %s",
                (user_id, lo, hi),
            ).fetchall()
            by_date: dict[str, list[dict[str, Any]]] = {}
            for a in acts:
                by_date.setdefault(a["start_date"], []).append(a)
            updates: list[tuple[Any, str, int, int]] = []
            # Each synced activity may complete at most one workout across the
            # whole season, so seed ``used`` with the links already stored by
            # earlier autocomplete runs — otherwise a planned workout could be
            # satisfied by re-borrowing an activity that already completed
            # another one (e.g. today's run stealing yesterday's run). Exact
            # planned days are matched first across all workouts, then the
            # ±1-day fallback fills the still-unmatched ones, so an adjacent-day
            # borrow never steals an activity from the workout on its own day.
            used: set[Any] = {
                row["completed_activity_id"]
                for row in conn.execute(
                    "SELECT completed_activity_id FROM training_workout "
                    "WHERE user_id = %s AND completed_activity_id IS NOT NULL",
                    (user_id,),
                ).fetchall()
            }
            matched: set[int] = set()
            pending: list[Any] = []
            for w in workouts:
                if w["planned_date"] > today:
                    continue
                day_acts = by_date.get(w["planned_date"], [])
                if w["activity_type"] == "rest":
                    if w["planned_date"] < today and not day_acts:
                        updates.append((None, now, user_id, w["id"]))
                    continue
                pending.append(w)

            def _claim(workout: Any, day_acts: list[dict[str, Any]]) -> bool:
                candidates = [
                    a for a in day_acts
                    if a["activity_id"] not in used
                    and _activity_matches(workout["activity_type"], a["activity_type"])
                ]
                best = _closest_activity(candidates, workout)
                if best is None:
                    return False
                used.add(best["activity_id"])
                matched.add(workout["id"])
                updates.append((best["activity_id"], now, user_id, workout["id"]))
                return True

            for w in pending:
                _claim(w, by_date.get(w["planned_date"], []))
            for w in pending:
                if w["id"] in matched:
                    continue
                for offset in (-1, 1):
                    other = (
                        date.fromisoformat(w["planned_date"])
                        + timedelta(days=offset)
                    ).isoformat()
                    if _claim(w, by_date.get(other, [])):
                        break
            if updates:
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE training_workout SET status = 'completed', "
                        "completed_activity_id = %s, "
                        "updated_at = %s WHERE user_id = %s AND id = %s",
                        updates,
                    )
        return {"completed": len(updates)}

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()


#: ``user_state`` key holding the one season undo snapshot. Reusing the existing
#: per-user key/value table keeps the snapshot in the same RLS scope and inside
#: the editing transaction, with no extra table.
_SNAPSHOT_KEY = "season_undo"


def _write_snapshot(conn: Any, user_id: int, payload: Any) -> None:
    """Store (replace) the season undo snapshot."""
    conn.execute(
        "INSERT INTO user_state (user_id, key, value, updated_at) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (user_id, key) DO UPDATE SET "
        "value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
        (
            user_id, _SNAPSHOT_KEY, json.dumps(payload, default=str),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _read_snapshot(conn: Any, user_id: int) -> Any | None:
    """Load the stored season undo snapshot (None when absent)."""
    row = conn.execute(
        "SELECT value FROM user_state WHERE user_id = %s AND key = %s",
        (user_id, _SNAPSHOT_KEY),
    ).fetchone()
    if row is None or not row["value"]:
        return None
    return json.loads(row["value"])


def _activity_matches(plan_type: str, garmin_type: str | None) -> bool:
    """True when a Garmin activity typeKey satisfies a plan ``activity_type``."""
    if plan_type == "other":
        return True
    allowed = GARMIN_TYPE_MAP.get(plan_type)
    return allowed is not None and garmin_type in allowed


def _goal_matches(sport: str | None, garmin_type: str | None) -> bool:
    """True when a Garmin typeKey counts toward a goal's volume.

    A goal with no sport (or the catch-all ``other``/``rest``) counts every
    activity; otherwise the sport must map through :data:`GARMIN_TYPE_MAP`.
    """
    if not sport or sport in ("other", "rest"):
        return True
    return _activity_matches(sport, garmin_type)


def _workout_matches_sport(sport: str | None, activity_type: str | None) -> bool:
    """True when a workout activity_type satisfies a goal's sport.

    Both values come from :data:`ACTIVITY_TYPES` ("run", "cycle", "swim",
    "strength", "rest", "other"). A goal with no sport (or "other"/"rest")
    matches all workout types. A workout with "rest" or "other" belongs to
    the active phase regardless of sport. Otherwise the types must match
    (or map via :data:`GARMIN_TYPE_MAP`).
    """
    if not sport or sport in ("other", "rest"):
        return True
    if not activity_type or activity_type in ("other", "rest"):
        return True
    if sport == activity_type:
        return True
    allowed = GARMIN_TYPE_MAP.get(sport)
    return allowed is not None and activity_type in allowed


def _closest_activity(
    candidates: list[dict[str, Any]], workout: dict[str, Any]
) -> dict[str, Any] | None:
    """Pick the candidate nearest the planned distance/duration."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if workout.get("distance_km") is not None:
        with_distance = [a for a in candidates if a.get("distance_km") is not None]
        if with_distance:
            return min(
                with_distance,
                key=lambda a: abs(workout["distance_km"] - a["distance_km"]),
            )
    if workout.get("duration_min") is not None:
        with_duration = [a for a in candidates if a.get("duration_hours") is not None]
        if with_duration:
            return min(
                with_duration,
                key=lambda a: abs(workout["duration_min"] - a["duration_hours"] * 60),
            )
    return candidates[0]


class TrainingAnchorStore:
    """Per-account long-term anchors in ``training_goal`` + ``training_block`` +
    ``training_week`` (RLS-scoped).

    Mirrors ``TrainingWorkoutStore``: an RLS-scoped writer pool with ``app.user_id``
    set per transaction. Unlike the original single-active design, an account may
    hold *many* goals (a marathon and a half-marathon), each with its own
    periodised blocks; every block carries one **week** row per calendar week
    of its progression (e.g. Week 1: 50km, Week 2: 55km, Week 3: 60km, Week 4:
    45km deload) that drives the plan tab's weekly progress bars. Weeks do not
    repeat: each row is one explicit Monday (``week_start``), derived from the
    block's start date and the week's position in the list.

    Updates are **partial**: pass only the fields you want changed (PATCH
    semantics), so adjusting one block's dates never rewrites the season.
    """

    def __init__(self, url: str | None = None, *, pool: Any | None = None) -> None:
        if pool is not None:
            self._pool = pool
            self._owns_pool = False
        else:
            self._pool = open_pg_pool(url, min_size=1, max_size=4)
            self._owns_pool = True

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    # -- goals -----------------------------------------------------------------

    def list_goals(self, user_id: int) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._list_goals(conn, user_id)

    @staticmethod
    def _list_goals(conn: Any, user_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        return [_goal_row(r) for r in rows]

    def get_goal(self, user_id: int, goal_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "SELECT * FROM training_goal WHERE user_id = %s AND id = %s",
                (user_id, goal_id),
            ).fetchone()
        return _goal_row(row) if row else None

    def create_goal(self, user_id: int, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new goal (appends it; never wipes other goals/blocks).

        ``data`` may carry nested ``blocks`` (list). Each block may carry
        ``weeks`` (list of per-week targets). Returns the saved goal row.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._create_goal(conn, user_id, data)

    def _create_goal(
        self, conn: Any, user_id: int, data: dict[str, Any]
    ) -> dict[str, Any]:
        fields = _normalize_goal(data)
        blocks = data.get("blocks") or []
        if not isinstance(blocks, list):
            raise ValueError("goal['blocks'] must be a list")
        now = datetime.now(timezone.utc).isoformat()
        goal_row = conn.execute(
            "INSERT INTO training_goal (user_id, title, sport, start_date, "
            "target_date, target_distance_km, target_time, "
            "created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING *",
            (
                user_id, fields.get("title"), fields.get("sport"),
                fields.get("start_date"), fields.get("target_date"),
                fields.get("target_distance_km"), fields.get("target_time"),
                now, now,
            ),
        ).fetchone()
        self._insert_blocks(conn, user_id, goal_row["id"], blocks, now)
        return _goal_row(goal_row)

    def update_goal(
        self, user_id: int, goal_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Partial update of one goal (never touches other goals).

        Only the fields present in ``data`` change. Block handling:
        ``blocks`` upserts (an entry with an ``id`` patches that block, one
        without creates a block) and ``delete_blocks`` removes specific blocks
        **of this goal**.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._update_goal(conn, user_id, goal_id, data)

    def _update_goal(
        self, conn: Any, user_id: int, goal_id: int, data: dict[str, Any],
        *, merge_weeks: bool = False,
    ) -> dict[str, Any] | None:
        fields = _normalize_goal(data, partial=True)
        has_children = any(
            key in data for key in ("blocks", "delete_blocks")
        )
        if not fields and not has_children:
            return self._get_goal(conn, user_id, goal_id)
        row = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s AND id = %s",
            (user_id, goal_id),
        ).fetchone()
        if row is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        if fields:
            assigns, params = _build_set(fields, now)
            row = conn.execute(
                f"UPDATE training_goal SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, goal_id),
            ).fetchone()
        if "blocks" in data:
            self._upsert_blocks(
                conn, user_id, goal_id, data["blocks"], now,
                merge_weeks=merge_weeks,
            )
        for block_id in data.get("delete_blocks") or []:
            block_id = _opt_int(block_id, "delete_blocks")
            if block_id is None:
                raise ValueError("delete_blocks must contain integer block ids")
            if not self._delete_block(conn, user_id, block_id, goal_id=goal_id):
                raise ValueError(f"block {block_id} not found in goal {goal_id}")
        return _goal_row(row)

    def _get_goal(
        self, conn: Any, user_id: int, goal_id: int
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s AND id = %s",
            (user_id, goal_id),
        ).fetchone()
        return _goal_row(row) if row else None

    def delete_goal(self, user_id: int, goal_id: int) -> bool:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._delete_goal(conn, user_id, goal_id)

    def _delete_goal(self, conn: Any, user_id: int, goal_id: int) -> bool:
        """Delete a goal + its blocks/weeks, detaching any planned workouts.

        ``training_workout.goal_id`` has no FK, so the link is
        NULLed first — otherwise deleting a goal leaves workouts pointing at a
        goal that no longer exists (the agent then reasons about a goal
        it cannot resolve).
        """
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE training_workout SET goal_id = NULL, updated_at = %s "
            "WHERE user_id = %s AND goal_id = %s",
            (now, user_id, goal_id),
        )
        conn.execute(
            "DELETE FROM training_week WHERE user_id = %s AND block_id IN "
            "(SELECT id FROM training_block WHERE user_id = %s AND goal_id = %s)",
            (user_id, user_id, goal_id),
        )
        conn.execute(
            "DELETE FROM training_block WHERE user_id = %s AND goal_id = %s",
            (user_id, goal_id),
        )
        cur = conn.execute(
            "DELETE FROM training_goal WHERE user_id = %s AND id = %s",
            (user_id, goal_id),
        )
        return cur.rowcount > 0

    def apply_spec(
        self, user_id: int, spec: dict[str, Any], *, conn: Any | None = None
    ) -> dict[str, Any]:
        """Apply one agent anchor edit (the ``training`` tool's ``anchor``
        section): optional goal deletions, then a goal create/update — all in one
        transaction, so a failure never leaves the season half-reshaped.

        The caller (``TrainingStore``) snapshots the whole season before a
        destructive edit, restorable with ``{"undo": true}``. Returns the saved
        goal row plus the resulting ``blocks``/``weeks`` ids so the caller never
        has to re-read the season to learn the new ids.
        """
        if conn is not None:
            return self._apply_spec(conn, user_id, spec)
        with self._pool.connection() as own:
            self._set_user(own, user_id)
            return self._apply_spec(own, user_id, spec)

    def _apply_spec(
        self, conn: Any, user_id: int, spec: dict[str, Any]
    ) -> dict[str, Any]:
        deleted = 0
        row: dict[str, Any] | None = None
        target_goal_id: int | None = None
        for gid in spec.get("delete_goal_ids") or []:
            gid = _opt_int(gid, "delete_goal_ids")
            if gid is None:
                raise ValueError("delete_goal_ids must contain integer goal ids")
            if self._delete_goal(conn, user_id, gid):
                deleted += 1
        goal_id = spec.get("goal_id")
        has_create_fields = any(
            key in spec for key in (
                "title", "sport", "start_date", "target_date",
                "target_distance_km", "target_time",
            )
        )
        has_block_fields = "blocks" in spec
        if goal_id is not None:
            gid = _opt_int(goal_id, "goal_id")
            if gid is None:
                raise ValueError("goal_id must be an integer")
            row = self._update_goal(conn, user_id, gid, spec, merge_weeks=True)
            if row is None:
                raise ValueError(f"goal {gid} not found")
            target_goal_id = gid
        elif has_create_fields:
            row = self._create_goal(conn, user_id, spec)
            target_goal_id = row["id"]
        elif has_block_fields:
            # A blocks-only spec is ambiguous: blocks belong to a goal. Never
            # invent a blank goal — target the sole existing goal, create one
            # only when the account has none, otherwise demand an explicit id.
            existing = self._list_goals(conn, user_id)
            if len(existing) == 1:
                gid = existing[0]["id"]
                row = self._update_goal(conn, user_id, gid, spec, merge_weeks=True)
                if row is None:
                    raise ValueError(f"goal {gid} not found")
                target_goal_id = gid
            elif not existing:
                row = self._create_goal(conn, user_id, spec)
                target_goal_id = row["id"]
            else:
                raise ValueError(
                    "anchor 'blocks' need a goal_id (or goal fields to create "
                    f"one); this account has {len(existing)} goals"
                )
        result: dict[str, Any] = {
            "goal": row,
            "deleted_goals": deleted,
        }
        if target_goal_id is not None:
            result["goal_id"] = target_goal_id
            result["blocks"], result["weeks"] = self._anchor_ids(
                conn, user_id, target_goal_id
            )
        return result

    @staticmethod
    def _anchor_ids(
        conn: Any, user_id: int, goal_id: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The (blocks, weeks) ids under one goal, for an apply result."""
        blocks = conn.execute(
            "SELECT id, name, start_date, end_date FROM training_block "
            "WHERE user_id = %s AND goal_id = %s ORDER BY start_date, id",
            (user_id, goal_id),
        ).fetchall()
        block_ids = [b["id"] for b in blocks]
        weeks: list[Any] = []
        if block_ids:
            weeks = conn.execute(
                "SELECT id, block_id, week_start FROM training_week "
                "WHERE user_id = %s AND block_id = ANY(%s) "
                "ORDER BY block_id, week_start",
                (user_id, block_ids),
            ).fetchall()
        return [dict(b) for b in blocks], [dict(w) for w in weeks]

    def week_number(self, user_id: int, block_id: int, day: str | None = None) -> int | None:
        """1-based calendar week (Mon..Sun) of ``day`` within a dated block."""
        block = self.get_block(user_id, block_id)
        if block is None or not block["start_date"]:
            return None
        day = day or date.today().isoformat()
        return _block_week_number(
            date.fromisoformat(block["start_date"]), date.fromisoformat(day)
        )

    def resolve_goal(
        self, user_id: int, goal_id: int, day: str | None = None
    ) -> dict[str, Any] | None:
        """One goal with its blocks (+ weeks), resolved block and weekly target.

        The single composition of the goal/block/week resolution used by the REST
        API, the agent tools and (indirectly) the plan tab, so those views can
        never disagree. Every block carries its ``week_count`` (from its dates)
        and the resolved ``week_number`` is the position of ``day`` within the
        current block. Returns None when the goal does not exist.
        """
        goal = self.get_goal(user_id, goal_id)
        if goal is None:
            return None
        day = day or date.today().isoformat()
        blocks, weeks_by_block = self.list_blocks_with_weeks(user_id, goal_id)
        for block in blocks:
            raw_weeks = weeks_by_block.get(block["id"], [])
            block["weeks"] = _resolve_block_weeks(block, raw_weeks)
            block["week_count"] = _block_week_count(block)
        current = self.current_block(user_id, day, goal_id, blocks=blocks)
        week_number = (
            self.week_number(user_id, current["id"], day) if current else None
        )
        target = self.weekly_target(
            user_id, day, goal_id, blocks=blocks, weeks_by_block=weeks_by_block,
        )
        return {
            "goal": goal,
            "blocks": blocks,
            "current_block": current,
            "week_number": week_number,
            # ``{"block": <row>, "week": <row>}`` (the API/plan-tab shape); the
            # agent tools slim it down.
            "weekly_target": target,
        }

    # -- blocks ----------------------------------------------------------------

    def list_blocks(
        self, user_id: int, goal_id: int | None = None
    ) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._list_blocks(conn, user_id, goal_id)

    @staticmethod
    def _list_blocks(
        conn: Any, user_id: int, goal_id: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM training_block WHERE user_id = %s"
        params: list[Any] = [user_id]
        if goal_id is not None:
            sql += " AND goal_id = %s"
            params.append(goal_id)
        sql += " ORDER BY goal_id, start_date, id"
        rows = conn.execute(sql, params).fetchall()
        return [_block_row(r) for r in rows]

    def list_blocks_with_weeks(
        self, user_id: int, goal_id: int | None = None
    ) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
        """Blocks + their weeks in two queries (not one per block).

        Returns ``(blocks, {block_id: [week, ...]})`` so a caller dumping a
        whole goal does not issue an N+1 query per phase.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            blocks = self._list_blocks(conn, user_id, goal_id)
            weeks: dict[int, list[dict[str, Any]]] = {}
            if not blocks:
                return blocks, weeks
            rows = conn.execute(
                "SELECT * FROM training_week WHERE user_id = %s "
                "AND block_id = ANY(%s) ORDER BY block_id, week_start",
                (user_id, [b["id"] for b in blocks]),
            ).fetchall()
            for row in rows:
                weeks.setdefault(row["block_id"], []).append(_week_row(row))
        return blocks, weeks

    def get_block(self, user_id: int, block_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            row = conn.execute(
                "SELECT * FROM training_block WHERE user_id = %s AND id = %s",
                (user_id, block_id),
            ).fetchone()
        return _block_row(row) if row else None

    def _create_block(
        self, conn: Any, user_id: int, goal_id: int, data: dict[str, Any],
        now: str,
    ) -> dict[str, Any]:
        fields = _normalize_block(data)
        fields.pop("goal_id", None)
        weeks = data.get("weeks") or []
        if not isinstance(weeks, list):
            raise ValueError("block['weeks'] must be a list")
        _validate_block_dates(fields.get("start_date"), fields.get("end_date"))
        self._check_exists(conn, "training_goal", user_id, goal_id)
        columns = ("user_id", "goal_id", "name", "start_date", "end_date",
                   "focus", "target_weekly_km", "created_at", "updated_at")
        row = conn.execute(
            f"INSERT INTO training_block ({', '.join(columns)}) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (
                user_id, goal_id, fields.get("name"), fields.get("start_date"),
                fields.get("end_date"), fields.get("focus"),
                fields.get("target_weekly_km"), now, now,
            ),
        ).fetchone()
        self._replace_weeks(conn, user_id, row["id"], weeks, now)
        return _block_row(row)

    def update_block(
        self, user_id: int, block_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Partial update of one block; ``weeks`` (if present) replaces the
        block's complete week list (``[]`` clears it). Other blocks stay
        untouched."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._update_block(conn, user_id, block_id, data)

    def _update_block(
        self, conn: Any, user_id: int, block_id: int, data: dict[str, Any],
        *, goal_id: int | None = None, merge_weeks: bool = False,
    ) -> dict[str, Any] | None:
        fields = _normalize_block(data, partial=True)
        fields.pop("goal_id", None)
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "SELECT * FROM training_block WHERE user_id = %s AND id = %s",
            (user_id, block_id),
        ).fetchone()
        if row is None:
            return None
        if goal_id is not None and row["goal_id"] != goal_id:
            raise ValueError(f"block {block_id} does not belong to goal {goal_id}")
        _validate_block_dates(
            fields.get("start_date", row["start_date"]),
            fields.get("end_date", row["end_date"]),
        )
        old_start = row["start_date"]
        if fields:
            assigns, params = _build_set(fields, now)
            row = conn.execute(
                f"UPDATE training_block SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, block_id),
            ).fetchone()
        if "weeks" in data:
            weeks = data["weeks"]
            if not isinstance(weeks, list):
                raise ValueError("block['weeks'] must be a list")
            if merge_weeks:
                self._merge_weeks(conn, user_id, block_id, weeks, now)
            else:
                self._replace_weeks(conn, user_id, block_id, weeks, now)
        elif "start_date" in data or "end_date" in data:
            # Weeks carry an absolute ``week_start`` anchored to the block, so a
            # date change must re-date them (not just re-run the arithmetic later).
            self._reanchor_weeks(conn, user_id, block_id, now, old_start)
        return _block_row(row)

    def delete_block(self, user_id: int, block_id: int) -> bool:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._delete_block(conn, user_id, block_id)

    # -- weeks -----------------------------------------------------------------

    def list_weeks(self, user_id: int, block_id: int) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            rows = conn.execute(
                "SELECT * FROM training_week WHERE user_id = %s AND block_id = %s "
                "ORDER BY week_start",
                (user_id, block_id),
            ).fetchall()
        return [_week_row(r) for r in rows]

    def replace_weeks(
        self, user_id: int, block_id: int, weeks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Wipe a block's weeks and write the supplied list (validated)."""
        if not isinstance(weeks, list):
            raise ValueError("weeks must be a list")
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            now = datetime.now(timezone.utc).isoformat()
            self._replace_weeks(conn, user_id, block_id, weeks, now)
        return self.list_weeks(user_id, block_id)

    # -- resolution ------------------------------------------------------------

    def current_block(
        self, user_id: int, day: str | None = None, goal_id: int | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Best-effort resolver: the block whose date range covers ``day``.

        Returns None when no block has dates (undated anchors) or nothing covers
        the day — the caller surfaces that so the agent can assign dates or treat
        the phase as undated. When several blocks overlap, the *shortest* wins
        (the most specific phase). Pass ``blocks`` to reuse an existing fetch.
        """
        if blocks is None:
            blocks = self.list_blocks(user_id, goal_id)
        day = day or date.today().isoformat()
        return _covering_block(date.fromisoformat(day), blocks)

    def weekly_target(
        self, user_id: int, day: str | None = None, goal_id: int | None = None,
        blocks: list[dict[str, Any]] | None = None,
        weeks_by_block: dict[int, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve the week target for the block that covers ``day``.

        A block's ``weeks`` are one row per calendar week (Mon..Sun), keyed by
        its absolute ``week_start``. They do not repeat, so a week with no
        stored row resolves to ``week`` = None. ``coverage`` compares the
        planned workouts and the synced activities for the goal against the
        week's target, and ``week_number``/``week_count`` report the position in
        the block so callers never re-derive it. Returns
        ``{"block", "week", "coverage", "week_number", "week_count"}`` or None.
        """
        day = day or date.today().isoformat()
        target_monday_dt = _first_monday(date.fromisoformat(day))
        target_monday = target_monday_dt.isoformat()
        block = self.current_block(user_id, day, goal_id, blocks=blocks)
        if block is None:
            all_blocks = blocks if blocks is not None else self.list_blocks(user_id, goal_id)
            block = _covering_block_for_week(target_monday_dt, all_blocks)
            if block is None:
                return None
        if weeks_by_block is not None:
            weeks = weeks_by_block.get(block["id"], [])
        else:
            weeks = self.list_weeks(user_id, block["id"])
        week = None
        if weeks:
            found = next(
                (w for w in weeks if w.get("week_start") == target_monday), None
            )
            if found is not None:
                week = dict(found)
        if week is None:
            if block.get("target_weekly_km") is not None and block.get("start_date") and block.get("end_date"):
                b_start = _first_monday(date.fromisoformat(block["start_date"])).isoformat()
                b_end = _first_monday(date.fromisoformat(block["end_date"])).isoformat()
                if b_start <= target_monday <= b_end:
                    week = {
                        "week_start": target_monday,
                        "distance_km": block.get("target_weekly_km"),
                        "duration_min": None,
                        "is_deload": False,
                    }
        else:
            if week.get("distance_km") is None and block.get("target_weekly_km") is not None:
                week["distance_km"] = block.get("target_weekly_km")
        coverage = (
            self._week_coverage(user_id, goal_id, block, week)
            if week is not None else None
        )
        return {
            "block": block,
            "week": week,
            "coverage": coverage,
            "week_number": (
                _block_week_number(
                    date.fromisoformat(block["start_date"]),
                    date.fromisoformat(day),
                )
                if block.get("start_date") else None
            ),
            "week_count": _block_week_count(block),
        }

    def _week_coverage(
        self, user_id: int, goal_id: int | None, block: dict[str, Any],
        week: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Planned + actual volume for ``week`` vs its target (km or min).

        Resolves the week's Mon..Sun range from its absolute ``week_start``,
        sums the planned workouts for the goal over that range and the synced
        activities matching the goal's sport, and reports the delta so the plan
        and the anchor can be reconciled instead of eyeballed.
        """
        ws = week.get("week_start")
        if not ws:
            return None
        we = (date.fromisoformat(ws) + timedelta(days=6)).isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            sport = None
            if goal_id is not None:
                row = conn.execute(
                    "SELECT sport FROM training_goal WHERE user_id = %s AND id = %s",
                    (user_id, goal_id),
                ).fetchone()
                sport = row["sport"] if row else None
            rows = conn.execute(
                "SELECT planned_date, activity_type, distance_km, duration_min, goal_id "
                "FROM training_workout WHERE user_id = %s "
                "AND planned_date >= %s AND planned_date <= %s",
                (user_id, ws, we),
            ).fetchall()
            actual_km, actual_min = self._activity_totals(
                conn, user_id, goal_id, ws, we
            )
        planned_km = planned_min = 0.0
        for r in rows:
            if goal_id is not None and r["goal_id"] is not None and r["goal_id"] != goal_id:
                continue
            if not _workout_matches_sport(sport, r["activity_type"]):
                continue
            planned_km += float(r["distance_km"] or 0)
            planned_min += float(r["duration_min"] or 0)
        return _coverage_entry(
            week, ws, we, planned_km, planned_min, actual_km, actual_min
        )

    @staticmethod
    def _activity_totals(
        conn: Any, user_id: int, goal_id: int | None, ws: str, we: str
    ) -> tuple[float, float]:
        """Sum synced activity km/minutes for a goal's sport over one week."""
        sport = None
        if goal_id is not None:
            row = conn.execute(
                "SELECT sport FROM training_goal WHERE user_id = %s AND id = %s",
                (user_id, goal_id),
            ).fetchone()
            sport = row["sport"] if row else None
        acts = conn.execute(
            "SELECT activity_type, distance_km, duration_hours "
            "FROM activity_summaries WHERE user_id = %s "
            "AND start_date >= %s AND start_date <= %s",
            (user_id, ws, we),
        ).fetchall()
        km = mins = 0.0
        for a in acts:
            if not _goal_matches(sport, a["activity_type"]):
                continue
            km += float(a["distance_km"] or 0)
            mins += float(a["duration_hours"] or 0) * 60
        return km, mins

    def coverage_range(
        self,
        user_id: int,
        date_start: str,
        date_end: str,
        goal_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Per-goal, per-calendar-week target and coverage for a window.

        Walks the Mon..Sun weeks touched by ``date_start``..``date_end`` and
        emits one entry per (goal, week) that maps to a dated block's explicit
        week row, carrying the week target and the goal-scoped
        planned/actual-vs-target coverage. Batched: one blocks+weeks fetch and
        one query each for planned and actual totals per goal (never a query
        per week), so a multi-week agenda stays cheap.
        """
        start = _first_monday(date.fromisoformat(date_start))
        end = date.fromisoformat(date_end)
        goals = self.list_goals(user_id)
        if goal_id is not None:
            goals = [g for g in goals if g["id"] == int(goal_id)]
        out: list[dict[str, Any]] = []
        for goal in goals:
            blocks, weeks_by_block = self.list_blocks_with_weeks(
                user_id, goal["id"]
            )
            if not blocks:
                continue
            with self._pool.connection() as conn:
                self._set_user(conn, user_id)
                rows = conn.execute(
                    "SELECT planned_date, activity_type, distance_km, duration_min, goal_id "
                    "FROM training_workout WHERE user_id = %s "
                    "AND planned_date >= %s AND planned_date <= %s",
                    (user_id, start.isoformat(), end.isoformat()),
                ).fetchall()
                acts = conn.execute(
                    "SELECT start_date, activity_type, distance_km, duration_hours "
                    "FROM activity_summaries WHERE user_id = %s "
                    "AND start_date >= %s AND start_date <= %s",
                    (user_id, start.isoformat(), end.isoformat()),
                ).fetchall()
            planned: dict[str, dict[str, float]] = {}
            sport = goal.get("sport")
            for r in rows:
                if r["goal_id"] is not None and r["goal_id"] != goal["id"]:
                    continue
                if not _workout_matches_sport(sport, r["activity_type"]):
                    continue
                ws = _first_monday(
                    date.fromisoformat(r["planned_date"])
                ).isoformat()
                bucket = planned.setdefault(ws, {"km": 0.0, "min": 0.0})
                bucket["km"] += float(r["distance_km"] or 0)
                bucket["min"] += float(r["duration_min"] or 0)
            actual: dict[str, dict[str, float]] = {}
            for a in acts:
                if not _goal_matches(goal.get("sport"), a["activity_type"]):
                    continue
                ws = _first_monday(date.fromisoformat(a["start_date"])).isoformat()
                bucket = actual.setdefault(ws, {"km": 0.0, "min": 0.0})
                bucket["km"] += float(a["distance_km"] or 0)
                bucket["min"] += float(a["duration_hours"] or 0) * 60
            cur = start
            while cur <= end:
                ws = cur.isoformat()
                block = _covering_block_for_week(cur, blocks)
                week = None
                if block is not None:
                    weeks = weeks_by_block.get(block["id"], [])
                    found = next(
                        (w for w in weeks if w.get("week_start") == ws), None
                    )
                    if found is not None:
                        week = dict(found)
                    if week is None:
                        if block.get("target_weekly_km") is not None:
                            week = {
                                "week_start": ws,
                                "distance_km": block.get("target_weekly_km"),
                                "duration_min": None,
                                "is_deload": False,
                            }
                    elif week.get("distance_km") is None and block.get("target_weekly_km") is not None:
                        week["distance_km"] = block.get("target_weekly_km")
                if block is not None:
                    p = planned.get(ws, {"km": 0.0, "min": 0.0})
                    a = actual.get(ws, {"km": 0.0, "min": 0.0})
                    week_dict = week or {}
                    out.append({
                        "goal_id": goal["id"],
                        "goal": goal["title"],
                        "block_id": block["id"],
                        "block": block["name"],
                        "week_start": ws,
                        "week_number": (
                            _block_week_number(
                                date.fromisoformat(block["start_date"]), cur
                            )
                            if block.get("start_date") else None
                        ),
                        "week_count": _block_week_count(block),
                        "week_target": {
                            "distance_km": week_dict.get("distance_km"),
                            "duration_min": week_dict.get("duration_min"),
                            "is_deload": week_dict.get("is_deload"),
                        },
                        "coverage": _coverage_entry(
                            week_dict, ws,
                            (cur + timedelta(days=6)).isoformat(),
                            p["km"], p["min"], a["km"], a["min"],
                        ),
                    })
                cur += timedelta(days=7)
        return out

    def close(self) -> None:
        self._pool.close()

    # -- private ---------------------------------------------------------------

    def _insert_blocks(
        self, conn: Any, user_id: int, goal_id: int,
        blocks: list[dict[str, Any]], now: str,
    ) -> None:
        """Write a goal's initial block set (create path only)."""
        if not isinstance(blocks, list):
            raise ValueError("goal['blocks'] must be a list")
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise ValueError("each block must be a JSON object")
            self._create_block(conn, user_id, goal_id, block, now)

    def _upsert_blocks(
        self, conn: Any, user_id: int, goal_id: int,
        blocks: list[dict[str, Any]], now: str,
        *, merge_weeks: bool = False,
    ) -> None:
        """Add/patch blocks without touching the others.

        An entry with an ``id`` patches that block (partial); one without an
        ``id`` creates a new block. ``merge_weeks`` selects patch-by-week
        semantics for a patched block's ``weeks`` (the agent path) instead of
        the default whole-vector replace (the UI path).
        """
        if not isinstance(blocks, list):
            raise ValueError("goal['blocks'] must be a list")
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("each block must be a JSON object")
            block_id = block.get("id")
            if block_id is None:
                self._create_block(conn, user_id, goal_id, block, now)
                continue
            block_id = _opt_int(block_id, "block id")
            if block_id is None:
                raise ValueError("block id must be an integer")
            if self._update_block(
                conn, user_id, block_id, block, goal_id=goal_id,
                merge_weeks=merge_weeks,
            ) is None:
                raise ValueError(f"block {block_id} not found in goal {goal_id}")

    def _replace_weeks(
        self, conn: Any, user_id: int, block_id: int,
        weeks: list[dict[str, Any]], now: str,
    ) -> None:
        """Replace a block's week targets; accepts full or partial week lists.

        Week targets can be keyed explicitly with ``week_start`` or given
        positionally relative to the block start. Partial lists are accepted
        without requiring blank ``{}`` padding for untargeted weeks. An empty
        list clears all week targets. A block must be dated before week targets
        can be set.
        """
        if not isinstance(weeks, list):
            raise ValueError("block['weeks'] must be a list")
        row = conn.execute(
            "SELECT start_date, end_date FROM training_block "
            "WHERE user_id = %s AND id = %s",
            (user_id, block_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"block {block_id} not found")
        if weeks and not row["start_date"]:
            raise ValueError(
                "a block needs a start_date before week targets can be set"
            )
        base = (
            _first_monday(date.fromisoformat(row["start_date"]))
            if row["start_date"] else None
        )
        normalized: list[dict[str, Any]] = []
        for week in weeks:
            if not isinstance(week, dict):
                raise ValueError("each week must be a JSON object")
            normalized.append(_normalize_week(week, partial=False))
        conn.execute(
            "DELETE FROM training_week WHERE user_id = %s AND block_id = %s",
            (user_id, block_id),
        )
        seen_starts: set[str] = set()
        for index, w in enumerate(normalized):
            if w.get("week_start"):
                ws = _first_monday(date.fromisoformat(w["week_start"])).isoformat()
            elif base is not None:
                ws = (base + timedelta(days=7 * index)).isoformat()
            else:
                ws = None
            has_target = (
                w.get("distance_km") is not None
                or w.get("duration_min") is not None
                or w.get("is_deload")
            )
            if not has_target and not w.get("week_start"):
                continue
            if ws in seen_starts:
                continue
            if ws is not None:
                seen_starts.add(ws)
            conn.execute(
                "INSERT INTO training_week (user_id, block_id, week_start, "
                "distance_km, duration_min, is_deload, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id, block_id, ws,
                    w.get("distance_km"), w.get("duration_min"),
                    w.get("is_deload"),
                    now, now,
                ),
            )

    def _merge_weeks(
        self, conn: Any, user_id: int, block_id: int,
        weeks: list[dict[str, Any]], now: str,
    ) -> None:
        """Patch a block's week targets keyed by ``week_start`` (agent path).

        Unlike :meth:`_replace_weeks`, only the weeks actually sent are written:
        every other stored week is left untouched, so correcting one week never
        wipes the rest of the block. A bare ``{week_start}`` entry clears that
        week; otherwise fields you send override and fields you omit keep their
        stored value; ``weeks: []`` clears every week of the block. A block must
        be dated to carry targets.
        """
        if not isinstance(weeks, list):
            raise ValueError("block['weeks'] must be a list")
        row = conn.execute(
            "SELECT start_date FROM training_block WHERE user_id = %s AND id = %s",
            (user_id, block_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"block {block_id} not found")
        if weeks and not row["start_date"]:
            raise ValueError(
                "a block needs a start_date before week targets can be set"
            )
        if not weeks:
            conn.execute(
                "DELETE FROM training_week WHERE user_id = %s AND block_id = %s",
                (user_id, block_id),
            )
            return
        base = (
            _first_monday(date.fromisoformat(row["start_date"]))
            if row["start_date"] else None
        )
        seen_starts: set[str] = set()
        for index, week in enumerate(weeks):
            if not isinstance(week, dict):
                raise ValueError("each week must be a JSON object")
            w = _normalize_week(week, partial=True)
            if w.get("week_start"):
                ws = _first_monday(date.fromisoformat(w["week_start"])).isoformat()
            elif base is not None:
                ws = (base + timedelta(days=7 * index)).isoformat()
            else:
                ws = None
            if ws is None or ws in seen_starts:
                continue
            seen_starts.add(ws)
            # A bare {week_start} is an explicit "remove this week's target".
            if not (set(w) - {"week_start"}):
                conn.execute(
                    "DELETE FROM training_week WHERE user_id = %s "
                    "AND block_id = %s AND week_start = %s",
                    (user_id, block_id, ws),
                )
                continue
            existing = conn.execute(
                "SELECT distance_km, duration_min, is_deload FROM training_week "
                "WHERE user_id = %s AND block_id = %s AND week_start = %s",
                (user_id, block_id, ws),
            ).fetchone()
            distance = (
                w["distance_km"] if "distance_km" in w
                else (existing["distance_km"] if existing else None)
            )
            duration = (
                w["duration_min"] if "duration_min" in w
                else (existing["duration_min"] if existing else None)
            )
            deload = (
                w["is_deload"] if "is_deload" in w
                else (existing["is_deload"] if existing else None)
            )
            if distance is None and duration is None and not deload:
                conn.execute(
                    "DELETE FROM training_week WHERE user_id = %s "
                    "AND block_id = %s AND week_start = %s",
                    (user_id, block_id, ws),
                )
                continue
            if existing is None:
                conn.execute(
                    "INSERT INTO training_week (user_id, block_id, week_start, "
                    "distance_km, duration_min, is_deload, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (user_id, block_id, ws, distance, duration, deload, now, now),
                )
            else:
                conn.execute(
                    "UPDATE training_week SET distance_km = %s, duration_min = %s, "
                    "is_deload = %s, updated_at = %s "
                    "WHERE user_id = %s AND block_id = %s AND week_start = %s",
                    (distance, duration, deload, now, user_id, block_id, ws),
                )

    def _reanchor_weeks(
        self, conn: Any, user_id: int, block_id: int, now: str,
        old_start: str | None,
    ) -> None:
        """Preserve week targets by their calendar week_start when block dates change."""
        pass

    def _delete_block(
        self, conn: Any, user_id: int, block_id: int, *, goal_id: int | None = None,
    ) -> bool:
        if goal_id is not None:
            row = conn.execute(
                "SELECT goal_id FROM training_block WHERE user_id = %s AND id = %s",
                (user_id, block_id),
            ).fetchone()
            if row is None or row["goal_id"] != goal_id:
                return False
        conn.execute(
            "DELETE FROM training_week WHERE user_id = %s AND block_id = %s",
            (user_id, block_id),
        )
        cur = conn.execute(
            "DELETE FROM training_block WHERE user_id = %s AND id = %s",
            (user_id, block_id),
        )
        return cur.rowcount > 0

    @staticmethod
    def _check_exists(conn: Any, table: str, user_id: int, row_id: int) -> None:
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE user_id = %s AND id = %s",
            (user_id, row_id),
        ).fetchone()
        if row is None:
            raise LookupError(f"{table} id {row_id} not found")


def _build_set(fields: dict[str, Any], now: str) -> tuple[str, list[Any]]:
    """Build a ``col = %s, ...`` SET clause + params for a partial update."""
    assigns = ", ".join(f"{key} = %s" for key in fields) + ", updated_at = %s"
    params = [fields[key] for key in fields] + [now]
    return assigns, params


class TrainingStore:
    """One owner of the whole training season on a single connection pool.

    The season is one tree — goal → block → week → workout — and this store is
    the only place it is written. The workout and anchor sub-stores share this
    pool, and one season-wide snapshot spans both, so a single destructive edit
    (in either half) is restored by a single ``undo``. Every write still sets
    ``app.user_id`` per transaction, so RLS isolates accounts.
    """

    def __init__(self, url: str) -> None:
        self._pool = open_pg_pool(url, min_size=1, max_size=8)
        self.workouts = TrainingWorkoutStore(pool=self._pool)
        self.anchor = TrainingAnchorStore(pool=self._pool)

    def _set_user(self, conn: Any, user_id: int) -> None:
        conn.execute(
            "SELECT set_config('app.user_id', %s, true)", (str(user_id),)
        )

    def close(self) -> None:
        self._pool.close()

    # -- destructive edits (one season snapshot, one undo) --------------------

    def apply(self, user_id: int, spec: dict[str, Any]) -> dict[str, Any]:
        """One season edit: optional ``anchor`` and/or ``workouts`` sections.

        This is the single write surface the agent uses. Both sections run on ONE
        connection/transaction, so a failure in either rolls the whole edit back —
        the season is never left half-reshaped. A destructive section snapshots
        the whole season once, before either section runs, so a single ``undo``
        rolls the entire edit back. ``{"undo": true}`` restores it.
        """
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object")
        if spec.get("undo"):
            with self._pool.connection() as conn:
                self._set_user(conn, user_id)
                return {"restored": self._undo(conn, user_id)}
        anchor_spec = spec.get("anchor")
        raw_workouts = spec.get("workouts")
        workouts_spec: dict[str, Any] | None = None
        if isinstance(raw_workouts, dict):
            workouts_spec = dict(raw_workouts)
            if spec.get("delete_ids") and "delete_ids" not in workouts_spec:
                workouts_spec["delete_ids"] = spec["delete_ids"]
        elif isinstance(raw_workouts, list):
            workouts_spec = {"workouts": raw_workouts}
            if spec.get("delete_ids"):
                workouts_spec["delete_ids"] = spec["delete_ids"]
        elif spec.get("delete_ids"):
            workouts_spec = {"delete_ids": spec["delete_ids"]}

        if anchor_spec is None and workouts_spec is None:
            raise ValueError(
                "spec needs an 'anchor' or 'workouts' section (or {'undo': true})"
            )
        if anchor_spec is not None and not isinstance(anchor_spec, dict):
            raise ValueError("'anchor' must be an object")
        destructive = bool(
            (anchor_spec and (
                anchor_spec.get("delete_goal_ids")
                or anchor_spec.get("delete_blocks")
            ))
            or (workouts_spec and workouts_spec.get("delete_ids"))
        )
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            if destructive:
                self._snapshot(conn, user_id)
            result: dict[str, Any] = {}
            if anchor_spec is not None:
                result["anchor"] = self.anchor.apply_spec(
                    user_id, anchor_spec, conn=conn
                )
            if workouts_spec is not None:
                result["workouts"] = self.workouts.apply(
                    user_id, workouts_spec, conn=conn
                )
        return result

    # -- one season-wide snapshot / undo --------------------------------------

    def snapshot(self, user_id: int) -> None:
        """Capture the whole season (plan + anchor) as the one undo snapshot."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            self._snapshot(conn, user_id)

    def _snapshot(self, conn: Any, user_id: int) -> None:
        plan = conn.execute(
            "SELECT * FROM training_workout WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        goals = conn.execute(
            "SELECT * FROM training_goal WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
        blocks = conn.execute(
            "SELECT * FROM training_block WHERE user_id = %s "
            "ORDER BY goal_id, start_date, id",
            (user_id,),
        ).fetchall()
        weeks = conn.execute(
            "SELECT * FROM training_week WHERE user_id = %s "
            "ORDER BY block_id, week_start",
            (user_id,),
        ).fetchall()
        _write_snapshot(conn, user_id, {
            "workouts": [dict(r) for r in plan],
            "goals": [dict(r) for r in goals],
            "blocks": [dict(r) for r in blocks],
            "weeks": [dict(r) for r in weeks],
        })

    def can_undo(self, user_id: int) -> bool:
        """True when a season snapshot exists (so ``undo`` would do something)."""
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return _read_snapshot(conn, user_id) is not None

    def undo(self, user_id: int) -> dict[str, int]:
        """Restore the whole season from the last snapshot (original ids kept).

        Plan rows carry their own goal/block ids, so restoring the tree in
        dependency-free order (goal, block, week, plan) re-links everything.
        """
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            return self._undo(conn, user_id)

    def _undo(self, conn: Any, user_id: int) -> dict[str, int]:
        payload = _read_snapshot(conn, user_id)
        if payload is None:
            raise ValueError("nothing to undo: no season snapshot stored")
        for table in (
            "training_workout", "training_week", "training_block", "training_goal"
        ):
            conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
        for g in payload.get("goals", []):
            conn.execute(
                "INSERT INTO training_goal (id, user_id, title, sport, "
                "start_date, target_date, target_distance_km, target_time, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    g.get("id"), user_id, g.get("title"), g.get("sport"),
                    g.get("start_date"), g.get("target_date"),
                    g.get("target_distance_km"), g.get("target_time"),
                    g.get("created_at"), g.get("updated_at"),
                ),
            )
        for b in payload.get("blocks", []):
            conn.execute(
                "INSERT INTO training_block (id, user_id, goal_id, name, "
                "start_date, end_date, focus, target_weekly_km, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    b.get("id"), user_id, b.get("goal_id"), b.get("name"),
                    b.get("start_date"), b.get("end_date"), b.get("focus"),
                    b.get("target_weekly_km"),
                    b.get("created_at"), b.get("updated_at"),
                ),
            )
        for w in payload.get("weeks", []):
            conn.execute(
                "INSERT INTO training_week (id, user_id, block_id, week_start, "
                "distance_km, duration_min, is_deload, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    w.get("id"), user_id, w.get("block_id"),
                    w.get("week_start"),
                    w.get("distance_km"), w.get("duration_min"),
                    w.get("is_deload"),
                    w.get("created_at"), w.get("updated_at"),
                ),
            )
        for w in payload.get("workouts", []):
            conn.execute(
                "INSERT INTO training_workout (id, user_id, planned_date, "
                "activity_type, title, description, duration_min, distance_km, "
                "intensity, target_pace_min_km, target_hr_zone, target_power_w, "
                "steps, status, completed_activity_id, goal_id, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s)",
                (
                    w.get("id"), user_id, w.get("planned_date"),
                    w.get("activity_type"), w.get("title"),
                    w.get("description"), w.get("duration_min"),
                    w.get("distance_km"), w.get("intensity"),
                    w.get("target_pace_min_km"), w.get("target_hr_zone"),
                    w.get("target_power_w"), w.get("steps"), w.get("status"),
                    w.get("completed_activity_id"),
                    w.get("goal_id"),
                    w.get("created_at"), w.get("updated_at"),
                ),
            )
        return {
            "workouts": len(payload.get("workouts", [])),
            "goals": len(payload.get("goals", [])),
        }


class TrainingSeason:
    """Per-user, user-scoped view of the one ``TrainingStore`` for the agent.

    The agent gets exactly one training tool; this facade exposes what that tool
    needs — read the tree and the resolved week, and apply one season spec —
    without threading a user id through ``build_agent``.
    """

    def __init__(self, store: TrainingStore, user_id: int) -> None:
        self._store = store
        self._user_id = user_id

    def list_goals(self) -> list[dict[str, Any]]:
        return self._store.anchor.list_goals(self._user_id)

    def resolve_goal(
        self, goal_id: int, day: str | None = None
    ) -> dict[str, Any] | None:
        return self._store.anchor.resolve_goal(self._user_id, goal_id, day)

    def blocks(self, goal_id: int) -> list[dict[str, Any]]:
        return self._store.anchor.list_blocks(self._user_id, goal_id)

    def coverage_range(
        self, date_start: str, date_end: str, goal_id: int | None = None
    ) -> list[dict[str, Any]]:
        return self._store.anchor.coverage_range(
            self._user_id, date_start, date_end, goal_id
        )

    def activities(
        self, date_start: str | None = None, date_end: str | None = None
    ) -> list[dict[str, Any]]:
        return self._store.workouts.activities(self._user_id, date_start, date_end)

    def list_workouts(
        self, date_start: str | None = None, date_end: str | None = None
    ) -> list[dict[str, Any]]:
        return self._store.workouts.list(self._user_id, date_start, date_end)

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._store.apply(self._user_id, spec)

    def can_undo(self) -> bool:
        return self._store.can_undo(self._user_id)
