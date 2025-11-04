# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:debian AS base
ENV VENV=/opt/venv
RUN uv python install 3.12 && uv venv "$VENV"
ENV PATH="$VENV/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e .

ENTRYPOINT ["kgpipe"]
