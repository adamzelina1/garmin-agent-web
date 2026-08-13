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
from .workload import (
    compute_acwr,
    fetch_activity_loads,
    to_metric_rows as acwr_rows,
)


def build_derived(db: Any) -> dict[str, int]:
    """Recompute readiness + ACWR from the parsed tables and store them.

    ``db`` is a user-bound ``Database``. Only readiness days that received a
    score are stored; the ACWR series always runs from the first training day
    to today (rest days are stored with a null ratio once chronic load decays
    to zero). Returns {"readiness": scored_days, "acwr": days,
    "derived": rows_written}.
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

    written = db.replace_derived_metrics(rows)
    return {
        "readiness": readiness_count,
        "acwr": len(acwr_days),
        "derived": written,
    }
