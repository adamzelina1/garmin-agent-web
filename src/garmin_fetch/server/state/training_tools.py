"""The agent's two training tools: ``training`` (read) and ``training_apply``
(one atomic edit), with the typed models of the edit.

The edit used to be a JSON *string* the model had to escape inside its JSON
tool call. As real objects the field rules live in the JSON schema the model
receives, and pydantic-ai rejects a malformed edit (with field-level errors
the model can fix) before the store sees it. The models mirror what
``TrainingStore.apply`` accepts and stay deliberately loose — the store's
normalisers in :mod:`.training_validation` remain the authority on values.
Unknown keys are dropped, as the store would ignore them. Dumps use
``exclude_unset=True``: an omitted field must stay absent (a PATCH leaves it
alone) while an explicit null clears it.

The tool functions live here, not in ``ask.py``, because pydantic-ai resolves
a tool's annotations in its defining module, and ``ask.py`` cannot import this
package at load time (``server`` imports ``ask``).
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...tool_args import Loose
from .training_validation import ACTIVITY_TYPES, INTENSITIES, PLAN_STATUSES, STEP_KINDS


ActivityType = Literal[ACTIVITY_TYPES]
Intensity = Literal[INTENSITIES]

_PACE = "decimal min/km (5.5 = 5:30)"
_HR_ZONE = "HR zone such as 'Z2' or 'Z1-Z2' (or a bpm range like '140-150')"


class _Spec(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Step(_Spec):
    """One segment of a workout."""

    kind: Literal[STEP_KINDS] = "steady"
    duration: str | float | None = Field(
        None, description="'15m', '90s', '1:30' or '1h30m'; a bare number is minutes"
    )
    duration_sec: int | None = Field(
        None, description="whole seconds — the form a read returns; send `duration` instead"
    )
    repeat: int | None = Field(None, ge=1, description="run this step N times")
    label: str | None = Field(None, description="1-2 words naming the segment; never digits or times")
    intensity: Intensity | None = None
    target_pace_min_km: float | None = Field(None, description=_PACE)
    target_hr_zone: str | None = Field(None, description=_HR_ZONE)
    target_power_w: int | None = None
    notes: str | None = None


class WorkoutEdit(_Spec):
    """A workout to create (no ``id``) or patch (with ``id``: send only what
    changes; an explicit null clears a field)."""

    id: int | None = Field(None, description="existing workout id to PATCH; omit to create")
    planned_date: str | None = Field(None, description="YYYY-MM-DD; required when creating")
    activity_type: ActivityType | None = Field(None, description="required when creating")
    title: str | None = None
    description: str | None = Field(None, description="ONE short line (<= ~200 chars) about THIS session")
    distance_km: float | None = Field(
        None,
        description="for a run, checked against its steps (time / pace); omit to have the steps' estimate filled in",
    )
    intensity: Intensity | None = None
    target_pace_min_km: float | None = Field(None, description=_PACE)
    target_hr_zone: str | None = Field(None, description=_HR_ZONE)
    target_power_w: int | None = None
    status: Literal[PLAN_STATUSES] | None = None
    goal_id: int | None = Field(None, description="optional override; workouts attach to the active block/goal by date")
    steps: Annotated[list[Step] | None, Loose] = Field(
        None,
        description=(
            "ordered segments; REQUIRED (at least one) when creating anything but a rest "
            "day — a single steady step for a plain run. Every non-rest step of a RUN "
            "needs target_pace_min_km (reps / race pace) or target_hr_zone (easy / steady)"
        ),
    )


class WeekEdit(_Spec):
    """One week's target. A bare ``{week_start}`` removes that week's target."""

    week_start: str = Field(description="any date in the week (YYYY-MM-DD; snapped to its Monday)")
    distance_km: float | None = None
    duration_min: int | None = None
    is_deload: bool | None = Field(None, description="true for a planned recovery week")


class BlockEdit(_Spec):
    """A periodized phase: with ``id`` it is patched, without it is created."""

    id: int | None = None
    name: str | None = Field(None, description="e.g. Base / Build / Peak / Taper")
    focus: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    target_weekly_km: float | None = Field(
        None, description="baseline for weeks without their own target — set it when weeks vary"
    )
    weeks: Annotated[list[WeekEdit] | None, Loose] = Field(
        None,
        description="per-week patch keyed by week_start: only the weeks sent are written; [] clears them all",
    )


class AnchorEdit(_Spec):
    """The goal and its blocks. With ``goal_id`` that goal is patched; with
    goal fields and no id one is created; blocks alone target the sole goal."""

    goal_id: int | None = None
    title: str | None = None
    sport: ActivityType | None = None
    start_date: str | None = None
    target_date: str | None = Field(None, description="the event / race day")
    target_time: str | None = Field(None, description="e.g. '3:45:00'")
    target_distance_km: float | None = None
    blocks: Annotated[list[BlockEdit] | None, Loose] = None
    delete_blocks: list[int] | None = Field(None, description="block ids to delete (undo restores them)")
    delete_goal_ids: list[int] | None = Field(None, description="goal ids to delete (undo restores them)")


def training_tools(season: Any) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """The read and edit tools bound to one user's season (a
    ``TrainingSeason``). Registration (status text, retries, JSON replies) is
    up to the caller."""

    def training(
        day: str | None = None,
        weeks: int = 1,
        date_start: str | None = None,
        date_end: str | None = None,
        goal_id: int | None = None,
        full: bool = False,
        detail: bool = False,
    ) -> Any:
        """Read the training season: goals -> periodized blocks -> weekly
        targets -> dated workouts.

        Returns each goal with the blocks overlapping the window, plus
        ``weeks``: one entry per calendar week (Mon..Sun) with that week's
        per-goal ``targets`` (target and planned/actual ``coverage``),
        ``planned`` workouts and ``actual`` activities. ``can_undo`` says
        whether the last destructive edit can be rolled back. ``warnings``
        lists runs this week or next whose distance disagrees with their
        steps: when present, tell the user and offer to fix them — don't edit
        the plan unasked. Read before building or reshaping a week, and to
        answer "how is my week going?".

        Args:
            day: YYYY-MM-DD the window starts from (default today).
            weeks: calendar weeks from Monday of ``day``'s week.
            date_start: explicit inclusive window start (with date_end); overrides weeks.
            date_end: explicit inclusive window end.
            goal_id: focus one goal.
            full: add every block/week of each goal plus the resolved current
                block — use it before a season-wide reshape.
            detail: keep full workout descriptions (otherwise previews).
        """
        return season.agenda(
            day=day, weeks=weeks, date_start=date_start, date_end=date_end,
            goal_id=goal_id, full=full, detail=detail,
        )

    def training_apply(
        workouts: Annotated[list[WorkoutEdit] | None, Loose] = None,
        delete_ids: list[int] | None = None,
        anchor: Annotated[AnchorEdit | None, Loose] = None,
        undo: bool = False,
    ) -> Any:
        """Make ONE atomic edit to the training season (all or nothing).

        The result lists the changed ids plus ``weeks``: each touched week's
        target, ``scheduled`` volume and ``delta``, so no follow-up read is
        needed. Before answering, fix every ``warnings`` entry (a run this
        week or next whose distance disagrees with its steps) and bring every
        ``delta`` near 0 by adjusting workouts — never do the week sums
        yourself. To move or copy workouts, read them first; never guess ids.
        A workout's ``duration_min`` is computed from its steps, so it is not
        a field here.

        Titles, descriptions, step labels and block name/focus describe that
        row only — never rules, preferences or reminders (durable guidance
        belongs in ``memory``). Derive target paces/zones from recent data
        (volume, HR zones, race predictions, VO2max). When the user sets a new
        target, reshape the anchor AND re-derive the near-term workouts in the
        same call.

        Args:
            workouts: workouts to create (no id) or patch (with id; send only what changes).
            delete_ids: workout ids to delete (undo restores them).
            anchor: the goal and its blocks / weekly targets.
            undo: roll back the last destructive edit instead (alone, nothing else).
        """
        if undo:
            return season.apply({"undo": True})
        spec: dict[str, Any] = {}
        if workouts:
            spec["workouts"] = [w.model_dump(exclude_unset=True) for w in workouts]
        if delete_ids:
            spec["delete_ids"] = list(delete_ids)
        if anchor is not None:
            spec["anchor"] = anchor.model_dump(exclude_unset=True)
        return season.apply(spec)

    return training, training_apply
