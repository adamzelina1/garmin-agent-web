"""Training-plan vocabulary, input normalisers and DB-row shapers.

Everything that turns untrusted dicts (REST bodies, agent specs) into clean
column values, and DB rows into the JSON the API/agent/tab consume. No
database access happens here.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from ...zone_pace import is_hr_target


#: Workout descriptions are meant to be a short one-line summary, but the agent
#: tends to write paragraphs. The prompt asks it to stay brief; this is the hard
#: backstop that holds regardless of model behaviour.
_MAX_DESCRIPTION_CHARS = 200

#: Allowed training-plan values (kept here so the store, the API and the agent
#: tools share one vocabulary).
#: There is no rest workout: a day without a workout is a rest day.
ACTIVITY_TYPES = ("run", "cycle", "swim", "strength", "other")
INTENSITIES = ("easy", "moderate", "hard", "race_pace")

#: Workout lifecycle. ``completed`` and ``partial`` both count as done; the
#: API's derived ``completed`` boolean is computed from this, never stored.
PLAN_STATUSES = ("planned", "completed", "partial", "skipped")

#: Structured-workout step roles. A workout's ``steps`` is an ordered JSON list
#: of ``{kind, duration_sec, ...}`` intervals (e.g. warm-up, threshold,
#: recovery); the targets (HR zone, pace) live on the steps, and the workout's
#: own ``target_hr_zone`` is the headline zone the plan shows on its card.
STEP_KINDS = ("warmup", "steady", "work", "recovery", "cooldown", "rest")

#: Hard backstops so a runaway edit can't bloat a single workout row.
_MAX_STEPS = 50
_MAX_STEP_TEXT = 80


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
    minutes); its optional targets are an HR zone and a pace. Returns None
    for an empty list. Raises ``ValueError`` on a malformed step.
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
        elif raw.get("duration_sec") not in (None, ""):
            # The stored/read-back form (whole seconds), so a step round-trips
            # through the API unchanged.
            try:
                duration = int(round(float(raw["duration_sec"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"step {i} duration_sec must be a number") from exc
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
        intensity = raw.get("intensity")
        if intensity not in (None, ""):
            intensity = str(intensity).strip().lower()
            if intensity not in INTENSITIES:
                raise ValueError(
                    f"step {i} intensity must be one of: {', '.join(INTENSITIES)}"
                )
            step["intensity"] = intensity
        if raw.get("target_pace_min_km") not in (None, ""):
            pace = _parse_opt_float(raw["target_pace_min_km"], f"step {i} target_pace_min_km")
            if not 1.0 <= pace <= 60.0:
                raise ValueError(
                    f"step {i} target_pace_min_km must be between 1 and 60 (decimal min/km)"
                )
            step["target_pace_min_km"] = pace
        hr = _text(raw.get("target_hr_zone"))
        if hr:
            step["target_hr_zone"] = hr
        notes = _text(raw.get("notes"))
        if notes:
            step["notes"] = notes[:_MAX_DESCRIPTION_CHARS]
        steps.append(step)
    return json.dumps(steps, ensure_ascii=False) if steps else None


def _require_steps(activity_type: Any, steps: Any) -> None:
    """Raise ``ValueError`` unless the workout has at least one step and
    every non-rest step of a run carries a pace or an HR zone (so the run's
    distance can be computed from its steps)."""
    parsed = _parse_steps(steps)
    if not parsed:
        raise ValueError(f"a {activity_type} workout needs at least one step")
    if activity_type != "run":
        return
    missing = [
        i for i, step in enumerate(parsed, start=1)
        if step.get("kind") != "rest"
        and not step.get("target_pace_min_km")
        and not is_hr_target(step.get("target_hr_zone"))
    ]
    if missing:
        raise ValueError(
            f"run step(s) {missing} need a target_pace_min_km or a "
            "target_hr_zone (e.g. 'Z2'); only 'rest' steps may have neither"
        )


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
        "distance_km", "intensity", "target_hr_zone", "status", "steps",
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
        elif key == "steps":
            fields[key] = _normalize_steps(value)
        elif key == "status":
            # A model-dumped null status means "not supplied".
            if value in (None, ""):
                continue
            fields[key] = _normalize_status(value)
    if not partial:
        _require_steps(fields["activity_type"], fields.get("steps"))
    return fields


def _normalize_status(value: Any) -> str:
    """Validate a workout status against :data:`PLAN_STATUSES`."""
    status = str(value).strip().lower()
    if status not in PLAN_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(PLAN_STATUSES)}")
    return status



def _parse_steps(raw: str | None) -> list[dict[str, Any]]:
    """Decode the stored ``steps`` JSON column into a list (one entry per
    rep)."""
    return json.loads(raw) if raw else []


def _workout_row(row: Any) -> dict[str, Any]:
    """Shape one DB row into the JSON the API/agent/tab all consume.
    ``status`` is the lifecycle value ('completed' and 'partial' both count
    as done)."""
    return {
        "id": row["id"],
        "planned_date": row["planned_date"],
        "activity_type": row["activity_type"],
        "title": row["title"],
        "description": row["description"],
        "duration_min": row["duration_min"],
        "distance_km": row["distance_km"],
        "intensity": row["intensity"],
        "target_hr_zone": row["target_hr_zone"],
        "steps": _parse_steps(row["steps"]),
        "status": row["status"],
        "completed_activity_id": row["completed_activity_id"],
    }


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


def _normalize_week(data: dict[str, Any]) -> dict[str, Any]:
    """Validate/normalise one week-target dict; raises ``ValueError``.

    A week is one calendar week (Mon..Sun) addressed by ``week_start``
    (required; any date in the week): a ``duration_min`` target (null
    removes the week) plus an ``is_deload`` flag (targets are minutes only;
    km is just an estimate). Only the keys present are returned, so a week
    can be patched.
    """
    fields: dict[str, Any] = {
        "week_start": _parse_required_date(data.get("week_start"), "week_start"),
    }
    if "duration_min" in data:
        fields["duration_min"] = _parse_opt_nonneg_int(data["duration_min"], "duration_min")
    if "is_deload" in data:
        value = data["is_deload"]
        if isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        fields["is_deload"] = bool(value)
    return fields


def _goal_row(row: Any) -> dict[str, Any]:
    """Shape the ``training_goal`` row into API/agent JSON."""
    return {
        "title": row["title"],
        "sport": row["sport"],
        "start_date": row["start_date"],
        "target_date": row["target_date"],
        "target_distance_km": row["target_distance_km"],
        "target_time": row["target_time"],
    }


def _week_row(row: Any) -> dict[str, Any]:
    """Shape one ``training_week`` row into API/agent JSON."""
    return {
        "week_start": row["week_start"],
        "duration_min": row["duration_min"],
        "is_deload": row["is_deload"],
    }


def _build_set(fields: dict[str, Any], now: str) -> tuple[str, list[Any]]:
    """Build a ``col = %s, ...`` SET clause + params for a partial update."""
    assigns = ", ".join(f"{key} = %s" for key in fields) + ", updated_at = %s"
    params = [fields[key] for key in fields] + [now]
    return assigns, params
