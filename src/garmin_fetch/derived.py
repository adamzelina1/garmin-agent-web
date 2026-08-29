"""Rebuild all derived daily metrics into the ``derived_metrics`` table.

Derived metrics (training readiness, ACWR, ...) are computed from the parsed
source tables (``daily_metrics``, ``activity_summaries``) and stored once per
sync. The whole series is recomputed and replaced each time, so the stored
table always reflects the current algorithms — and the AI agent and the UI
read exactly the same values.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .readiness import (
    compute_readiness,
    fetch_daily_source,
    to_metric_rows as readiness_rows,
)
from .run_workload import (
    compute_gait,
    compute_run_acwr,
    fetch_running_runs,
    to_gait_rows,
    to_metric_rows as run_rows,
)
from .workload import (
    compute_acwr,
    fetch_activity_loads,
    to_metric_rows as acwr_rows,
)


def build_derived(db: Any) -> dict[str, int]:
    """Recompute readiness + ACWR + running ACWR + gait drift from the parsed
    tables.

    ``db`` is a user-bound ``Database``. Only readiness days that received a
    score are stored; the cardio ACWR and running ACWR series always run from
    the first training/run day to today (rest days are stored with a null ratio
    once chronic load decays to zero); gait drift is stored per run day.
    Returns {"readiness": scored_days, "acwr": days, "running_acwr": days,
    "running_gait": days, "derived": rows_written}.
    """
    built_at = datetime.now(UTC).isoformat()

    rows: list[dict[str, Any]] = []
    readiness_count = 0
    for day in compute_readiness(fetch_daily_source(db.conn)):
        if day["score"] is None:
            continue
        rows.extend(readiness_rows(day, built_at))
        readiness_count += 1

    acwr_days = compute_acwr(fetch_activity_loads(db.conn))
    for day in acwr_days:
        rows.extend(acwr_rows(day, built_at))

    run_days = compute_run_acwr(fetch_running_runs(db.conn))
    for day in run_days:
        rows.extend(run_rows(day, built_at))

    gait_days = compute_gait(db.conn)
    for day in gait_days:
        rows.extend(to_gait_rows(day, built_at))

    written = db.replace_derived_metrics(rows)
    return {
        "readiness": readiness_count,
        "acwr": len(acwr_days),
        "running_acwr": len(run_days),
        "running_gait": len(gait_days),
        "derived": written,
    }

