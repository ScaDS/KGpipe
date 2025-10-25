# syntax=docker/dockerfile:1.7

# ---------- Base image ----------
FROM ghcr.io/astral-sh/uv:debian AS base
ENV VENV=/opt/venv
RUN uv python install 3.12 && uv venv "$VENV"
ENV PATH="$VENV/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app

# ---------- Build wheel for kgcore (from external context) ----------
FROM base AS build-kgcore
WORKDIR /src_kgcore
COPY --from=kgcore / /src_kgcore
RUN uv build --wheel

FROM base AS build-pyodibel
WORKDIR /src_pyodibel
COPY --from=pyodibel / /src_pyodibel
RUN uv build --wheel

# ---------- Final app stage ----------
FROM base AS app
WORKDIR /app

# Copy dependency metadata first
COPY pyproject.toml ./

# Bring in kgcore wheel
RUN mkdir -p /wheelhouse
COPY --from=build-kgcore /src_kgcore/dist/*.whl /wheelhouse/

COPY --from=build-pyodibel /src_pyodibel/dist/*.whl /wheelhouse/

# Copy app source
COPY . .

# Install everything into the venv, preferring the local kgcore wheel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --find-links /wheelhouse -e .

RUN /opt/venv/bin/python -c "import kgcore" \
    && /opt/venv/bin/python -c "import importlib.metadata as m; print('OK:', any(d.metadata.get('Name')=='kgcore' for d in m.distributions()))"


# Run the installed console entry point directly (no uv auto-sync)
CMD ["kgpipe"]
