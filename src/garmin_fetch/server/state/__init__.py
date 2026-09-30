"""Per-account state: agent memory/session/trace and the training season.

* :mod:`.user_state` — ``UserState`` key/value rows, agent memory and principles.
* :mod:`.training_validation` — plan vocabulary, normalisers, row shapers.
* :mod:`.training_rules` — pure week math and goal/activity matching.
* :mod:`.training_workouts` — planned workouts store (+ autocomplete).
* :mod:`.training_anchor` — goals, blocks and week targets store.
* :mod:`.training_agenda` — the agent's windowed season agenda (read-only).
* :mod:`.training_store` — the transactional season writer, undo, agent facade.

Everything the rest of the app imports is re-exported here, so callers keep
using ``from .state import ...``.
"""

from .training_anchor import TrainingAnchorStore
from .training_rules import (
    GARMIN_TYPE_MAP, goal_activities, goal_workouts, workout_in_goal,
)
from .training_store import TrainingSeason, TrainingStore
from .training_validation import (
    ACTIVITY_TYPES, INTENSITIES, PLAN_STATUSES, STEP_KINDS,
)
from .training_workouts import TrainingWorkoutStore
from .user_state import PgMemory, PgPrinciples, UserState

__all__ = [
    "ACTIVITY_TYPES", "GARMIN_TYPE_MAP", "INTENSITIES", "PLAN_STATUSES",
    "PgMemory", "PgPrinciples", "STEP_KINDS", "TrainingAnchorStore",
    "TrainingSeason", "TrainingStore", "TrainingWorkoutStore", "UserState",
    "goal_activities", "goal_workouts", "workout_in_goal",
]
