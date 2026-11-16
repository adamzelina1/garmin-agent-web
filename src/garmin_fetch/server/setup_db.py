"""Database role bootstrap for the multi-user server (Phase 3).

Row-Level Security only means something if the runtime connections are NOT a
superuser (superusers bypass RLS). The server therefore runs as two dedicated
roles, created once by this module (or the shipped ``db_setup.sql``) using a
superuser DSN:

- ``garmin_app``      — owns the tables, performs syncs + auth writes. Subject
                        to RLS (enforced via ``FORCE ROW LEVEL SECURITY``).
- ``garmin_readonly`` — SELECT-only on the agent-facing tables, used by the
                        LLM agent. RLS scopes it per ``app.user_id``, and it
                        has no EXECUTE on ``set_config`` so a query cannot
                        change that setting.

``ensure_roles()`` is idempotent and safe to run at every server start.
"""

from __future__ import annotations

import re

APP_ROLE = "garmin_app"
READONLY_ROLE = "garmin_readonly"

_SET_CONFIG = "pg_catalog.set_config(text, text, boolean)"


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
        # RLS keys off the ``app.user_id`` setting, so whoever can call
        # set_config() can switch accounts — even from inside a SELECT. Only
        # the app role keeps that right; the agent role binds its user with
        # ``SET LOCAL`` (a utility command no query can issue).
        conn.execute(f"REVOKE EXECUTE ON FUNCTION {_SET_CONFIG} FROM PUBLIC")
        conn.execute(f"GRANT EXECUTE ON FUNCTION {_SET_CONFIG} TO {APP_ROLE}")


def agent_role_can_switch_user(readonly_url: str) -> bool:
    """Whether the role behind ``readonly_url`` may still call ``set_config``.

    True means a query run as the agent could rewrite ``app.user_id`` and read
    another account's rows, i.e. ``ensure_roles`` has not been applied to this
    database yet.
    """
    import psycopg

    with psycopg.connect(readonly_url) as conn:
        return bool(
            conn.execute(
                "SELECT has_function_privilege(current_user, %s, 'EXECUTE')",
                (_SET_CONFIG,),
            ).fetchone()[0]
        )
