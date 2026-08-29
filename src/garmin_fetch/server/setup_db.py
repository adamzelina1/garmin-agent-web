"""Database role bootstrap for the multi-user server (Phase 3).

Row-Level Security only means something if the runtime connections are NOT a
superuser (superusers bypass RLS). The server therefore runs as two dedicated
roles, created once by this module (or the shipped ``db_setup.sql``) using a
superuser DSN:

- ``garmin_app``      — owns the tables, performs syncs + auth writes. Subject
                        to RLS (enforced via ``FORCE ROW LEVEL SECURITY``).
- ``garmin_readonly`` — SELECT-only on the five agent-facing tables, used by
                        the LLM agent. RLS scopes it per ``app.user_id``.

``ensure_roles()`` is idempotent and safe to run at every server start.
"""

from __future__ import annotations

import re

APP_ROLE = "garmin_app"
READONLY_ROLE = "garmin_readonly"


def _dsn_credentials(dsn: str) -> tuple[str | None, str | None]:
    """Extract (user, password) from a ``postgresql://user:pass@...`` DSN."""
    m = re.match(r"postgres(?:ql)?://([^:/@]+):([^@]*)@", dsn)
    if not m:
        return None, None
    user = m.group(1)
    password = m.group(2)
    return user, password or None


def ensure_roles(admin_url: str, app_url: str, readonly_url: str) -> None:
    """Create the ``garmin_app`` / ``garmin_readonly`` roles if missing.

    ``admin_url`` must connect as a superuser. Passwords are taken from the
    runtime DSNs, so the roles stay in sync with the configured connection
    strings. Existing roles are left untouched.
    """
    if not admin_url:
        return
    import psycopg
    import psycopg.sql as sq

    roles: list[tuple[str, str, str | None]] = [
        (APP_ROLE, "app", app_url),
        (READONLY_ROLE, "readonly", readonly_url),
    ]
    with psycopg.connect(admin_url, autocommit=True) as conn:
        db_name = conn.execute("SELECT current_database()").fetchone()[0]
        for role, label, url in roles:
            role_id = sq.Identifier(role)
            exists = conn.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)
            ).fetchone()
            if not exists:
                _user, password = _dsn_credentials(url)
                if not password:
                    raise RuntimeError(
                        f"{role} role is missing and no password can be derived "
                        f"from the {label} DSN"
                    )
                conn.execute(
                    sq.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                        role_id, sq.Literal(password)
                    )
                )
        for role in (APP_ROLE, READONLY_ROLE):
            conn.execute(
                sq.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sq.Identifier(db_name), sq.Identifier(role)
                )
            )
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}, {READONLY_ROLE}")
        conn.execute(f"GRANT CREATE ON SCHEMA public TO {APP_ROLE}")
        _ensure_user_config_columns(conn)


def _ensure_user_config_columns(conn: Any) -> None:
    """Idempotently add per-user config columns to the ``users`` table.

    New installs get them from ``db.py`` DDL; this keeps existing databases
    in sync without a migration framework. Each column is an optional override
    (NULL = fall back to the server-level .env value).
    """
    for column, column_type in (
        ("home_lat", "TEXT"),
        ("home_lon", "TEXT"),
        ("home_city", "TEXT"),
        ("home_country", "TEXT"),
        ("excluded_data_types", "TEXT"),
        ("sync_start_date", "TEXT"),
        ("auto_sync", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ):
        # Skip silently until the table exists (fresh volumes create it later).
        exists = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'users'"
        ).fetchone()
        if not exists:
            continue
        conn.execute(
            f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {column} {column_type}"
        )
