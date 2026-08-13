# The multi-user FastAPI server (auth + sync worker + read-only ask agent).
# All state lives in Postgres, so the image is stateless: no volumes needed.
#
#   docker compose build server
#   docker compose up -d server
FROM python:3.14-slim

# uv handles the dependency install and the project build (pyproject.toml +
# uv.lock); no system python, matching local dev.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Layer 1: resolve + install dependencies from the lockfile, cached unless the
# manifest changes. --no-install-project skips building our own package here.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: build and install the project itself.
COPY src ./src
RUN uv sync --frozen --no-dev

# Run as a non-root user; expose the venv entry points directly so the runtime
# never needs the uv cache or network.
ENV PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["garmin-server", "--host", "0.0.0.0", "--port", "8000"]
