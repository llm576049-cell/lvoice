# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Pinned CosyVoice2 checkpoint. Bump this (and re-build) to switch model variants.
ARG MODEL_REPO=FunAudioLLM/CosyVoice2-0.5B
ARG MODEL_DIR=/models/CosyVoice2-0.5B

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    LVOICE_MODEL_DIR=${MODEL_DIR} \
    PATH="/app/.venv/bin:$PATH"

# build-essential: compiles pyworld's C extension during `uv sync`.
# libsndfile1: runtime dependency of the `soundfile` package (audio I/O).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# uv manages its own Python toolchain, so the base image doesn't need python preinstalled.
RUN uv python install 3.10

COPY pyproject.toml uv.lock ./
COPY third_party ./third_party
RUN uv sync --frozen --no-dev --extra cosyvoice

# Bakes the model weights into the image (see PLAN.md: chosen over a runtime
# download so the container is self-contained). --no-dev keeps `uv run` from
# re-syncing the dev dependency group (ruff/pytest/httpx) into this layer.
# Deliberately placed before `COPY app` so app code changes don't bust this
# layer's cache and force a ~4.6GB re-download.
RUN uv run --no-dev python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', local_dir='${MODEL_DIR}')"

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
