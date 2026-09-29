"""``TrainingAnchorStore``: goals, blocks and week targets (RLS-scoped)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from ...db import open_pg_pool
from .training_validation import (
    _block_row,
    _build_set,
    _goal_row,
    _normalize_block,
    _normalize_goal,
    _normalize_week,
    _opt_int,
    _validate_block_dates,
    _week_row,
)
from .training_rules import (
    _block_week_count,
    _block_week_number,
    _coverage_entry,
    _covering_block,
    _effective_week,
    _first_monday,
    _resolve_block_weeks,
    goal_activities,
    goal_workouts,
)


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

    def _update_goal(
        self, conn: Any, user_id: int, goal_id: int, data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Partial update of one goal (never touches other goals).

        Only the fields present in ``data`` change. Block handling:
        ``blocks`` upserts (an entry with an ``id`` patches that block, one
        without creates a block) and ``delete_blocks`` removes specific blocks
        **of this goal**. Reached only through :meth:`TrainingStore.apply`.
        """
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
            self._upsert_blocks(conn, user_id, goal_id, data["blocks"], now)
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
            row = self._update_goal(conn, user_id, gid, spec)
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
                row = self._update_goal(conn, user_id, gid, spec)
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

    def resolve_goal(
        self, user_id: int, goal_id: int, day: str | None = None
    ) -> dict[str, Any] | None:
        """One goal with its blocks (+ weeks), resolved block and weekly target.

        The single composition of the goal/block/week resolution used by the REST
        API, the agent tools and (indirectly) the plan tab, so those views can
        never disagree. Every block carries its ``week_count`` (from its dates)
        and ``week_number`` is the position of ``day``'s week within its block.
        Returns None when the goal does not exist.
        """
        goal = self.get_goal(user_id, goal_id)
        if goal is None:
            return None
        day = day or date.today().isoformat()
        blocks, weeks_by_block = self.list_blocks_with_weeks(user_id, goal_id)
        for block in blocks:
            block["weeks"] = _resolve_block_weeks(
                block, weeks_by_block.get(block["id"], [])
            )
            block["week_count"] = _block_week_count(block)
        target = self._weekly_target(user_id, day, goal, blocks, weeks_by_block)
        return {
            "goal": goal,
            "blocks": blocks,
            "current_block": _covering_block(blocks, date.fromisoformat(day)),
            "week_number": target["week_number"] if target else None,
            # ``{"block", "week", "coverage", "week_number", "week_count"}``
            # (the API/plan-tab shape); the agent tools slim it down.
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
        if weeks:
            self._merge_weeks(conn, user_id, row["id"], weeks, now)
        return _block_row(row)

    def _update_block(
        self, conn: Any, user_id: int, block_id: int, data: dict[str, Any],
        *, goal_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Partial block update; ``weeks`` (if present) patches by ``week_start``
        (see :meth:`_merge_weeks`). Weeks are keyed by absolute Monday, so a
        date change never needs to re-date them."""
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
        if fields:
            assigns, params = _build_set(fields, now)
            row = conn.execute(
                f"UPDATE training_block SET {assigns} "
                "WHERE user_id = %s AND id = %s RETURNING *",
                (*params, user_id, block_id),
            ).fetchone()
        if "weeks" in data:
            self._merge_weeks(conn, user_id, block_id, data["weeks"], now)
        return _block_row(row)

    # -- resolution ------------------------------------------------------------

    def _weekly_volumes(
        self, user_id: int, goal: dict[str, Any], blocks: list[dict[str, Any]],
        start: date, end: date,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Planned and actual km/min per Mon..Sun week (keyed by Monday) for one goal.

        Two queries for the whole window, never one per week. Planned counts the
        workouts attributed to the goal (:func:`workout_in_goal`); actual counts
        the synced activities of the goal's sport (:func:`goal_activities`).
        """
        lo, hi = start.isoformat(), end.isoformat()
        with self._pool.connection() as conn:
            self._set_user(conn, user_id)
            workouts = conn.execute(
                "SELECT planned_date, activity_type, distance_km, duration_min, "
                "goal_id FROM training_workout WHERE user_id = %s "
                "AND planned_date >= %s AND planned_date <= %s",
                (user_id, lo, hi),
            ).fetchall()
            acts = conn.execute(
                "SELECT start_date, activity_type, distance_km, duration_hours "
                "FROM activity_summaries WHERE user_id = %s "
                "AND start_date >= %s AND start_date <= %s",
                (user_id, lo, hi),
            ).fetchall()
        planned: dict[str, dict[str, float]] = {}
        for w in goal_workouts(goal, blocks, workouts):
            ws = _first_monday(date.fromisoformat(w["planned_date"])).isoformat()
            bucket = planned.setdefault(ws, {"km": 0.0, "min": 0.0})
            bucket["km"] += float(w["distance_km"] or 0)
            bucket["min"] += float(w["duration_min"] or 0)
        actual: dict[str, dict[str, float]] = {}
        for a in goal_activities(goal, acts):
            ws = _first_monday(date.fromisoformat(a["start_date"])).isoformat()
            bucket = actual.setdefault(ws, {"km": 0.0, "min": 0.0})
            bucket["km"] += float(a["distance_km"] or 0)
            bucket["min"] += float(a["duration_hours"] or 0) * 60
        return planned, actual

    def _coverage_rows(
        self, user_id: int, goal: dict[str, Any],
        blocks: list[dict[str, Any]],
        weeks_by_block: dict[int, list[dict[str, Any]]],
        start: date, end: date,
    ) -> list[dict[str, Any]]:
        """One row per Mon..Sun week touched by ``start``..``end`` that a dated
        block covers: ``{block, week, week_number, week_count, coverage}``.

        The single week-resolution path: ``week`` is the block's effective
        target (:func:`_effective_week`) and ``coverage`` compares the planned and
        actual volume of that whole week against it. Both the plan tab's current
        week and the agent's multi-week agenda are built from these rows.
        """
        if not blocks:
            return []
        cur = _first_monday(start)
        planned, actual = self._weekly_volumes(
            user_id, goal, blocks, cur, _first_monday(end) + timedelta(days=6)
        )
        zero = {"km": 0.0, "min": 0.0}
        rows: list[dict[str, Any]] = []
        while cur <= end:
            ws = cur.isoformat()
            we = (cur + timedelta(days=6)).isoformat()
            block = _covering_block(blocks, cur, cur + timedelta(days=6))
            if block is not None:
                stored = next(
                    (w for w in weeks_by_block.get(block["id"], [])
                     if w.get("week_start") == ws),
                    None,
                )
                week = _effective_week(block, ws, stored)
                p, a = planned.get(ws, zero), actual.get(ws, zero)
                rows.append({
                    "block": block,
                    "week": week,
                    "week_number": _block_week_number(
                        date.fromisoformat(block["start_date"]), cur
                    ),
                    "week_count": _block_week_count(block),
                    "coverage": _coverage_entry(
                        week or {}, ws, we, p["km"], p["min"], a["km"], a["min"]
                    ),
                })
            cur += timedelta(days=7)
        return rows

    def _weekly_target(
        self, user_id: int, day: str, goal: dict[str, Any],
        blocks: list[dict[str, Any]],
        weeks_by_block: dict[int, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        """The target row for the calendar week containing ``day`` (or None)."""
        monday = _first_monday(date.fromisoformat(day))
        rows = self._coverage_rows(
            user_id, goal, blocks, weeks_by_block, monday, monday
        )
        if not rows:
            return None
        row = rows[0]
        if row["week"] is None:
            row = {**row, "coverage": None}
        return row

    def coverage_range(
        self,
        user_id: int,
        date_start: str,
        date_end: str,
        goal_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Per-goal, per-calendar-week target and coverage for a window.

        One entry per (goal, week) that maps to a dated block, carrying the week
        target and the goal-scoped planned/actual-vs-target coverage. Batched:
        one blocks+weeks fetch and two volume queries per goal (never a query
        per week), so a multi-week agenda stays cheap.
        """
        start = date.fromisoformat(date_start)
        end = date.fromisoformat(date_end)
        goals = self.list_goals(user_id)
        if goal_id is not None:
            goals = [g for g in goals if g["id"] == int(goal_id)]
        out: list[dict[str, Any]] = []
        for goal in goals:
            blocks, weeks_by_block = self.list_blocks_with_weeks(user_id, goal["id"])
            for row in self._coverage_rows(
                user_id, goal, blocks, weeks_by_block, start, end
            ):
                week = row["week"] or {}
                block = row["block"]
                out.append({
                    "goal_id": goal["id"],
                    "goal": goal["title"],
                    "block_id": block["id"],
                    "block": block["name"],
                    "week_start": row["coverage"]["week_start"],
                    "week_number": row["week_number"],
                    "week_count": row["week_count"],
                    "week_target": {
                        "distance_km": week.get("distance_km"),
                        "duration_min": week.get("duration_min"),
                        "is_deload": week.get("is_deload"),
                    },
                    "coverage": row["coverage"],
                })
        return out

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()

    # -- private ---------------------------------------------------------------

    def _insert_blocks(
        self, conn: Any, user_id: int, goal_id: int,
        blocks: list[dict[str, Any]], now: str,
    ) -> None:
        """Write a goal's initial block set (create path only)."""
        if not isinstance(blocks, list):
            raise ValueError("goal['blocks'] must be a list")
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("each block must be a JSON object")
            self._create_block(conn, user_id, goal_id, block, now)

    def _upsert_blocks(
        self, conn: Any, user_id: int, goal_id: int,
        blocks: list[dict[str, Any]], now: str,
    ) -> None:
        """Add/patch blocks without touching the others.

        An entry with an ``id`` patches that block (partial); one without an
        ``id`` creates a new block.
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
                conn, user_id, block_id, block, goal_id=goal_id
            ) is None:
                raise ValueError(f"block {block_id} not found in goal {goal_id}")

    def _merge_weeks(
        self, conn: Any, user_id: int, block_id: int,
        weeks: list[dict[str, Any]], now: str,
    ) -> None:
        """Patch a block's week targets keyed by ``week_start`` (agent path).

        The only week writer (UI, REST and agent all use it). Only the weeks
        actually sent are written:
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
