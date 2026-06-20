# CosyVoice HTTP API Wrapper

## Context

The repo (`lvoice`) is currently empty (just a `LICENSE`). The goal is to build a Dockerized HTTP API that wraps [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (Alibaba's TTS model family) so other services can call TTS over HTTP instead of dealing with the Python/PyTorch model directly.

Decisions already made with the user:
- **Stack**: Python + FastAPI, importing CosyVoice in-process (no IPC hop).
- **Hardware**: support both GPU (CUDA) and CPU, auto-detected at runtime.
- **Feature scope**: full CosyVoice surface — basic TTS, zero-shot voice cloning, cross-lingual synthesis, instruct/emotion control.
- **Model weights**: baked into the Docker image at build time (self-contained image, no first-run download).
- **Chinese support**: confirmed — CosyVoice natively supports Mandarin/Cantonese (plus English/Japanese/Korean and cross-lingual); pin a checkpoint with Chinese capability (e.g. `CosyVoice2-0.5B` or `CosyVoice-300M`).
- **Tooling**: package management via `uv` (`pyproject.toml` + `uv.lock`, no `requirements.txt`), code formatting/linting via `ruff`.
- **Manual test page**: a simple static webpage to type text, hit the API, and play back the resulting audio in-browser, for quick manual sanity checks.
- **Engine extensibility**: add a thin `BaseTTSEngine` abstraction now so other TTS backends (Coqui, Piper, Edge-TTS, etc.) can be plugged in later without touching the API/router layer; CosyVoice is the only concrete implementation for now.

## Project Layout

```
lvoice/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml         # deps + ruff config, managed with uv
├── uv.lock
├── .dockerignore
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, startup/shutdown, router mounting
│   ├── config.py            # env-driven settings (model dir, device, ports, limits)
│   ├── deps.py               # singleton engine provider (lifespan-managed), picks impl via config.ENGINE
│   ├── engines/
│   │   ├── base.py            # BaseTTSEngine ABC: synthesize/clone/cross_lingual/instruct/list_voices
│   │   └── cosyvoice.py        # CosyVoiceEngine(BaseTTSEngine): load, synth, clone, cross-lingual, instruct
│   ├── schemas.py            # pydantic request/response models
│   ├── audio.py              # tensor -> wav/mp3 encoding helpers
│   ├── static/
│   │   └── index.html        # manual test page: textarea + speaker picker + <audio> playback
│   └── routers/
│       ├── tts.py             # POST /v1/tts, /v1/tts/clone, /v1/tts/cross-lingual, /v1/tts/instruct
│       ├── voices.py          # GET /v1/voices (built-in speakers), reference-voice upload/list
│       └── health.py          # GET /healthz, /readyz
└── tests/
    ├── test_health.py
    └── test_tts.py
```

## Core Design

### 1. Engine abstraction (`app/engines/base.py`, `app/engines/cosyvoice.py`)
- `BaseTTSEngine` ABC defines the contract every backend must implement:
  - `synthesize(text, speaker_id, speed) -> np.ndarray` (preset/sft voices)
  - `clone(text, prompt_wav_bytes, prompt_text) -> np.ndarray` (zero-shot cloning)
  - `cross_lingual(text, prompt_wav_bytes) -> np.ndarray`
  - `instruct(text, speaker_id, instruct_text) -> np.ndarray` (emotion/style control)
  - `list_voices() -> list[str]`
- `CosyVoiceEngine(BaseTTSEngine)` is the only concrete implementation for now, loaded once at process startup (FastAPI lifespan) holding the loaded CosyVoice model.
- `app/deps.py` selects which engine class to instantiate based on `config.ENGINE` (default/only value: `cosyvoice`) — routers depend only on `BaseTTSEngine`, never import `CosyVoiceEngine` directly. This is what makes adding a second backend later a matter of writing `engines/<name>.py` + registering it in the factory, with no router/API changes.
- Device selection: `torch.cuda.is_available()` → `cuda`, else `cpu`, overridable via `DEVICE` env var.
- All inference calls run via `run_in_threadpool` (or a small worker queue) since CosyVoice inference is CPU/GPU-bound and synchronous — keeps the event loop responsive and lets us cap concurrency to avoid OOM on shared GPU.

### 2. API surface (`app/routers/tts.py`)
- `POST /v1/tts` — `{text, speaker_id, speed?, format?}` → audio bytes (wav/mp3 via `audio.py`)
- `POST /v1/tts/clone` — multipart: `text`, `prompt_text`, `prompt_audio` (file) → audio bytes
- `POST /v1/tts/cross-lingual` — multipart: `text`, `prompt_audio` (file) → audio bytes
- `POST /v1/tts/instruct` — `{text, speaker_id, instruct_text}` → audio bytes
- Response: streamed audio (`StreamingResponse`, `audio/wav` or `audio/mpeg`) plus an optional `?return=base64` mode for JSON-only clients.
- `GET /v1/voices` — lists built-in CosyVoice speaker IDs available in the loaded model.
- `GET /healthz` — liveness (process up). `GET /readyz` — readiness (model loaded).

### 2b. Manual test page (`app/static/index.html`)
- FastAPI mounts `app/static` via `StaticFiles` at `/` (or `/test`), serving a single self-contained HTML file (vanilla JS, no build step).
- UI: a textarea for input text, a `<select>` populated from `GET /v1/voices` for speaker/mode choice, a "Speak" button, and an `<audio controls>` element.
- On submit: `fetch('/v1/tts', {method:'POST', body: JSON.stringify({text, speaker_id})})` → read response as `blob()` → `URL.createObjectURL(blob)` → set as the `<audio>` `src` and autoplay.
- Purely for manual sanity-checking during development/demos — not a production UI, no auth, no styling beyond basics.

### 3. Config (`app/config.py`)
Env vars: `MODEL_DIR` (baked-in path, e.g. `/models/CosyVoice2-0.5B`), `MODEL_NAME`, `DEVICE` (`auto|cuda|cpu`), `MAX_TEXT_LENGTH`, `MAX_CONCURRENT_REQUESTS`, `LOG_LEVEL`.

### 4. Concurrency / resource control
- A bounded `asyncio.Semaphore` (or a small thread pool with fixed size) around engine calls to prevent unbounded concurrent GPU/CPU inference from OOMing the container. Size configurable via `MAX_CONCURRENT_REQUESTS` (default 1 for GPU, a few for CPU).

### 5. Error handling
- Validate text length, supported audio formats for uploaded prompt audio, and speaker_id existence → return 400 with a clear message.
- Catch model inference exceptions → 500 with a generic message (don't leak internals), log full stack trace.

## Docker

### Dockerfile (multi-stage)
- Base: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (works for both GPU and CPU-only hosts — CUDA runtime libs are present but unused if no GPU is exposed) or a slimmer CPU-targeted alternate stage if image size matters later. Given "support both" was chosen, standardize on the CUDA runtime base so the same image runs on either.
- Install Python 3.10+ and `uv` (`COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/` is the standard pattern), then `uv sync --frozen` against the committed `pyproject.toml`/`uv.lock` (torch, torchaudio matching CUDA version, CosyVoice + its deps — `WeTextProcessing`, `onnxruntime`, etc.). `ruff` is a dev-only dependency (not needed at runtime, but `uv sync` keeps it out of the final layer via `--no-dev` if we split groups).
- Clone/vendor CosyVoice source (as a git submodule or pinned pip-installed wheel if available) into the image.
- `RUN` step downloads/bakes the chosen CosyVoice checkpoint (e.g. via `modelscope` or `huggingface_hub` download) into `MODEL_DIR` at build time — this is the step that makes the image self-contained per the user's choice. Document the model variant pinned (e.g. `CosyVoice2-0.5B`) as a build arg so it's easy to bump.
- `EXPOSE 8000`, `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`

### docker-compose.yml
- Single service `lvoice`, port `8000:8000`.
- GPU profile via `deploy.resources.reservations.devices` (Compose v2 GPU syntax) so `docker compose up` picks up GPU if `nvidia-container-toolkit` is installed and host has a GPU; otherwise falls back to CPU automatically inside the app via `DEVICE=auto`.
- Healthcheck hitting `/healthz`.

## Verification Plan
1. `docker build -t lvoice .` succeeds and produces a runnable image with model baked in.
2. `docker compose up` (no GPU) → container starts, `/healthz` and `/readyz` return 200, `DEVICE` resolves to `cpu` in logs.
3. `curl -X POST localhost:8000/v1/tts -d '{"text":"hello world","speaker_id":"<default>"}' -H 'Content-Type: application/json' --output out.wav` → produces playable audio.
4. Repeat with `/v1/tts/clone` using a short reference wav to confirm voice cloning path works end-to-end.
5. `GET /v1/voices` returns the expected list of built-in speakers.
6. Open `http://localhost:8000/` in a browser, type Chinese and English text, pick a voice, click Speak, confirm audio plays back.
7. If a GPU host is available, re-run with `docker compose --profile gpu up` (or `docker run --gpus all`) and confirm `DEVICE=cuda` in logs and faster inference.
8. `pytest tests/` for unit-level schema/validation tests (mocking the engine) so CI doesn't need a GPU or the real model to run basic checks.

## Milestones

**M1 — Project scaffold**
- `uv init`, `pyproject.toml` (deps + `[tool.ruff]`), `config.py`, repo skeleton in place.
- Deliverable: `uv sync` runs cleanly, empty FastAPI app boots locally.

**M2 — Engine layer (CosyVoice, no Docker yet)**
- `BaseTTSEngine` ABC + `CosyVoiceEngine` implementation, built against CosyVoice's real inference API (pin version/commit first).
- Runs locally on CPU against a manually-downloaded checkpoint (not yet baked into an image).
- Deliverable: a local script/test that calls `CosyVoiceEngine.synthesize(...)` and produces a `.wav` file.

**M3 — HTTP API**
- Schemas, routers (`/v1/tts`, `/v1/tts/clone`, `/v1/tts/cross-lingual`, `/v1/tts/instruct`, `/v1/voices`), health routes, concurrency guard, error handling.
- Deliverable: `uvicorn app.main:app` locally + `curl` against all endpoints produces playable audio.

**M4 — Manual test page**
- Static `index.html` served by FastAPI for typing text and playing back audio in-browser.
- Deliverable: open the page in a browser, synthesize Chinese + English text, hear it play.

**M5 — Dockerization**
- Dockerfile (`uv sync --frozen`, CosyVoice + checkpoint baked in), docker-compose with GPU profile.
- Deliverable: `docker build` + `docker compose up` works end-to-end on CPU; GPU path verified if a GPU host is available.

**M6 — Polish**
- `ruff format`/`check` clean, `pytest tests/` (mocked-engine unit tests), README with usage examples.
- Deliverable: repo is in a shareable, documented state.
