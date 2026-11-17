# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.10.2@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 AS uv
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

FROM base AS builder
COPY --from=uv /uv /usr/local/bin/uv
# Build tools stay in this stage (some locked packages ship source archives).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

FROM base AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home app \
    && mkdir -p /cache/huggingface /app/config \
    && chown -R app:app /cache /home/app
COPY --from=builder /app/.venv /app/.venv
# Explicit allowlist: no host environment, credentials, data or model weights.
COPY src/api/ ./src/api/
COPY src/predict.py src/train.py src/experiment.py src/tracking.py ./src/
ENV HF_HOME=/cache/huggingface \
    CUDA_VISIBLE_DEVICES="" \
    SUPPLY_CHAIN_SELECTION_PATH=/app/config/selection.json \
    SUPPLY_CHAIN_ENV_FILE=/app/config/no-local-env \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import json, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); assert r.status == 200 and json.load(r)['model_ready'] is True"]
CMD ["uvicorn", "api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
