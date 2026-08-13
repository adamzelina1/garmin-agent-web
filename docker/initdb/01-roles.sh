#!/bin/bash
# Create the non-superuser roles the Garmin server runs as (runs once on a
# fresh volume via docker-entrypoint-initdb.d). RLS only means something when
# the runtime connections are not superusers.
#
# Passwords come from POSTGRES_APP_PASSWORD / POSTGRES_READONLY_PASSWORD env
# (see docker-compose.yml). The app role gets CREATE on public so the server
# can build its own tables; the read-only role gets only USAGE.
set -euo pipefail

APP_USER="${POSTGRES_APP_USER:-garmin_app}"
APP_PASS="${POSTGRES_APP_PASSWORD:-garmin_app}"
RO_USER="${POSTGRES_READONLY_USER:-garmin_readonly}"
RO_PASS="${POSTGRES_READONLY_PASSWORD:-garmin_readonly}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${APP_USER}') THEN
    CREATE ROLE ${APP_USER} LOGIN PASSWORD '${APP_PASS}';
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${RO_USER}') THEN
    CREATE ROLE ${RO_USER} LOGIN PASSWORD '${RO_PASS}';
  END IF;
END
\$\$;
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO ${APP_USER}, ${RO_USER};
GRANT USAGE ON SCHEMA public TO ${APP_USER}, ${RO_USER};
GRANT CREATE ON SCHEMA public TO ${APP_USER};
EOSQL
