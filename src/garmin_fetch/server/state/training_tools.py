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


class _StrictSpec(_Spec):
    """Rejects unknown keys. For steps and weeks a dropped key is not
    harmless: a ``repeat`` would quietly turn 4 reps into 1, and a week left
    with only ``week_start`` deletes that week's target."""

    model_config = ConfigDict(extra="forbid")


class Step(_StrictSpec):
    """One segment of a workout. Intervals are written out step by step
    (4x4 = work, recovery, work, recovery, work, recovery, work)."""

    kind: Literal[STEP_KINDS] = "steady"
    duration: str | float | None = Field(
        None, description="'15m', '90s', '1:30' or '1h30m'; a bare number is minutes"
    )
    duration_sec: int | None = Field(
        None, description="whole seconds — the form a read returns; send `duration` instead"
    )
    label: str | None = Field(None, description="1-2 words naming the segment; never digits or times")
    intensity: Intensity | None = None
    target_hr_zone: str | None = Field(None, description=_HR_ZONE)
    target_pace_min_km: float | None = Field(None, description=_PACE)
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
        description="only for non-run workouts; a run's distance is always estimated from its steps",
    )
    intensity: Intensity | None = None
    target_hr_zone: str | None = Field(
        None, description="the session's headline HR zone shown on its plan card (e.g. 'Z2'); per-segment zones go on the steps",
    )
    status: Literal[PLAN_STATUSES] | None = None
    steps: Annotated[list[Step] | None, Loose] = Field(
        None,
        description=(
            "ordered segments; REQUIRED (at least one) when creating — a single steady "
            "step for a plain run. Every non-rest step of a RUN needs target_hr_zone "
            "(e.g. 'Z2') or target_pace_min_km"
        ),
    )


class WeekEdit(_StrictSpec):
    """One week's target, in minutes. A bare ``{week_start}`` removes it."""

    week_start: str = Field(description="any date in the week (YYYY-MM-DD; snapped to its Monday)")
    duration_min: int | None = Field(
        None, description="the week's total planned minutes; required for a new week, null removes the week",
    )
    is_deload: bool | None = Field(None, description="true for a planned recovery week")


class AnchorEdit(_Spec):
    """The account's ONE goal and the week targets: goal fields sent patch
    the goal (or create it when there is none)."""

    title: str | None = None
    sport: ActivityType | None = Field(None, description="only this sport's minutes count toward the week targets")
    start_date: str | None = Field(None, description="the season's first day; weeks are numbered from it")
    target_date: str | None = Field(None, description="the event / race day")
    target_time: str | None = Field(None, description="e.g. '3:45:00'")
    target_distance_km: float | None = None
    weeks: Annotated[list[WeekEdit] | None, Loose] = Field(
        None, description="week targets by date; only the weeks sent change",
    )
    delete_goal: bool | None = Field(
        None, description="delete the goal first; week targets and workouts stay (undo restores it)",
    )


def training_tools(season: Any) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """The read and edit tools bound to one user's season (a
    ``TrainingSeason``). Registration (status text, retries, JSON replies) is
    up to the caller."""

    def training(
        date_start: str | None = None,
        date_end: str | None = None,
        full: bool = False,
    ) -> Any:
        """Read the training season: the goal, the weekly targets and the
        dated workouts.

        Returns the ``goal`` plus ``weeks``: one entry per calendar week
        (Mon..Sun) of the window with its ``target`` (``week`` — its place
        in the season, "12 of 31" —, ``target_min``, ``scheduled_min``,
        ``delta_min``, ``change_pct`` — the target vs the previous week's —,
        ``est_km``, ``actual_min`` / ``actual_km`` and ``deload``; absent
        when the week has no target), ``planned`` workouts (a workout with
        steps carries its zone/pace targets on the steps) and ``actual``
        synced activities. Empty fields are left out. ``can_undo`` says
        whether the last destructive edit can be rolled back. Read before
        building or reshaping a week, and to answer "how is my week going?".

        Args:
            date_start: inclusive window start, YYYY-MM-DD (with date_end;
                default: this week).
            date_end: inclusive window end, YYYY-MM-DD.
            full: add ``season``: every stored week target, in the same
                shape as ``weeks[].target`` plus its ``week_start`` (no
                workouts) — use it before a season-wide reshape.
        """
        return season.agenda(date_start=date_start, date_end=date_end, full=full)

    def training_apply(
        workouts: Annotated[list[WorkoutEdit] | None, Loose] = None,
        delete_ids: list[int] | None = None,
        anchor: Annotated[AnchorEdit | None, Loose] = None,
        undo: bool = False,
    ) -> Any:
        """Make ONE atomic edit to the training season (all or nothing).

        Volume is planned in MINUTES: a workout's ``duration_min`` is computed
        from its steps and a week's target is ``duration_min``. km is only an
        estimate (a run's distance is priced from its steps) — never a target.
        The result lists the changed ids plus ``weeks``: each touched week's
        ``target_min``, ``scheduled_min``, ``delta_min``, ``change_pct`` (the
        target vs the previous week's, in %) and ``est_km``, so no follow-up
        read is needed. Bring every ``delta_min`` to 0 by adjusting
        workouts; change a week's target only when the user asks. To move or
        copy workouts, read them first; never guess ids.

        Titles, descriptions, step labels and the goal's title describe that
        row only — never rules, preferences or reminders (durable guidance
        belongs in ``memory``). Derive target paces/zones from recent data
        (volume, HR zones, race predictions, VO2max). When the user sets a new
        target, reshape the anchor AND re-derive the near-term workouts in the
        same call.

        Args:
            workouts: workouts to create (no id) or patch (with id; send only what changes).
            delete_ids: workout ids to delete (undo restores them).
            anchor: the goal and the weekly targets.
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
