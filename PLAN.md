# lvoice — Design

## Context

`lvoice` is a Dockerized HTTP API wrapping [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)
(Alibaba's zero-shot TTS model), built so other services can call TTS over HTTP instead of dealing
with the Python/PyTorch model directly. This document was originally written as a forward-looking
implementation plan before any code existed; it now describes the architecture as actually shipped.
For user-facing usage (endpoints, env vars, local dev), see [README.md](README.md) — this file is
the "why it's built this way" companion.

## Decisions

- **Stack**: Python + FastAPI, importing CosyVoice in-process (no IPC hop).
- **Hardware**: supports both GPU (CUDA) and CPU, auto-detected at runtime via `LVOICE_DEVICE=auto`.
- **Feature scope**: full CosyVoice2 surface — zero-shot cloning, registered/reusable voices,
  cross-lingual synthesis, instruct/emotion control.
- **Model weights**: baked into the Docker image at build time (self-contained image, no
  first-run download).
- **Tooling**: `uv` for package management (`pyproject.toml` + `uv.lock`, no `requirements.txt`),
  `ruff` for formatting/linting.
- **Engine extensibility**: a thin `BaseTTSEngine` abstraction so other backends could be plugged
  in later without touching the router/API layer. `CosyVoiceEngine` is the only implementation.
- **Network independence**: the built image makes no network calls at runtime — verified with
  `docker run --network none`. See "Offline resources" below.
- **Speaker persistence**: registered voices survive a container restart via a mounted volume,
  not just in-process memory.

## Why CosyVoice2 specifically

CosyVoice2 is zero-shot-only — unlike the older CosyVoice v1, it has no baked-in preset/SFT
speakers. Every voice comes from a short reference clip. This shaped `BaseTTSEngine`: there's no
`list of built-in speakers` concept, only `register_speaker` (clone + remember) and one-shot
`clone`/`cross_lingual`/`instruct` calls.

## Project Layout

```
lvoice/
├── Dockerfile
├── docker-compose.yml       # CPU-only by default
├── docker-compose.gpu.yml   # GPU overlay: -f docker-compose.yml -f docker-compose.gpu.yml
├── pyproject.toml           # deps (base + "cosyvoice" extra) + ruff/pytest config
├── uv.lock
├── .dockerignore
├── README.md                # user-facing docs
├── PLAN.md                  # this file
├── app/
│   ├── main.py               # FastAPI app, lifespan engine loading, router/static mounting
│   ├── config.py             # env-driven Settings (LVOICE_* prefix)
│   ├── deps.py                # engine singleton + require_engine (503 if not loaded) + inference semaphore
│   ├── engines/
│   │   ├── base.py             # BaseTTSEngine ABC
│   │   └── cosyvoice.py         # CosyVoiceEngine: load, register/synthesize/clone/cross_lingual/instruct
│   ├── schemas.py             # pydantic request/response models (JSON endpoints only)
│   ├── audio.py               # stdlib-only float32 -> 16-bit PCM wav encoder
│   ├── static/
│   │   └── index.html          # manual test page (incl. in-browser mic recording)
│   └── routers/
│       ├── tts.py               # POST /v1/tts, /register, /clone, /cross-lingual, /instruct
│       ├── voices.py            # GET /v1/voices
│       └── health.py            # GET /healthz, /readyz
├── tests/                    # mocked-engine tests; run without the ML stack
└── third_party/CosyVoice/    # git submodule, pinned commit, vendors CosyVoice + its own Matcha-TTS submodule
```

## Core Design

### 1. Engine abstraction (`app/engines/base.py`, `app/engines/cosyvoice.py`)
`BaseTTSEngine` defines: `register_speaker`, `synthesize`, `clone`, `cross_lingual`, `instruct`,
`list_voices`, `sample_rate`. Routers depend only on this interface via `app.deps.require_engine`,
never on `CosyVoiceEngine` directly — adding a second backend means writing
`engines/<name>.py` and extending `build_engine()` (currently in `cosyvoice.py`), no router changes.

`CosyVoiceEngine` is loaded once at process startup (FastAPI lifespan in `main.py`) and held as a
module-level singleton in `deps.py`. Device selection (`cuda`/`cpu`/`auto`) is handled by setting
`CUDA_VISIBLE_DEVICES=""` before CosyVoice2's constructor runs, since CosyVoice2 itself has no
device parameter and always checks `torch.cuda.is_available()` internally.

Prompt audio is written to a temp file rather than kept in memory as `BytesIO`: CosyVoice's
frontend re-opens the `prompt_wav` path multiple times internally (once per target sample rate), so
a stream gets exhausted after the first read — a path is required.

### 2. API surface (`app/routers/tts.py`, `app/routers/voices.py`)
- `POST /v1/tts` (JSON) — synthesize with a previously registered `speaker_id`.
- `POST /v1/tts/register` (multipart) — clone + remember a voice under a `speaker_id`.
- `POST /v1/tts/clone` (multipart) — one-shot zero-shot cloning, no registration.
- `POST /v1/tts/cross-lingual` (multipart) — speak text in another language using the cloned voice.
- `POST /v1/tts/instruct` (multipart) — speak text following a style/emotion/accent instruction.
- `GET /v1/voices` — list registered speaker ids.
- `GET /healthz` / `GET /readyz` — liveness / readiness (`require_engine` dependency reports 503
  instead of crashing if the engine hasn't finished loading).

All synthesis endpoints return raw `audio/wav` bytes (see `app/audio.py` — a stdlib-only
16-bit-PCM encoder, deliberately not using `soundfile`, so the API/schema layer stays importable
without the "cosyvoice" extra's heavy ML dependencies).

### 3. Manual test page (`app/static/index.html`)
FastAPI mounts `app/static` via `StaticFiles` at `/`. Single self-contained HTML file, vanilla JS,
no build step. Covers all four inference modes via tabs, plus an in-browser microphone recorder for
the registration flow: browsers record to webm/opus, which the API doesn't accept, so the page
decodes the recording via the Web Audio API and re-encodes it client-side as 16-bit mono PCM wav
before upload (no extra libraries).

### 4. Config (`app/config.py`)
Pydantic `Settings`, `LVOICE_` env prefix. See README's configuration table for the full list —
notably `speaker_store_path` (persistence, below) and `offline_resources` (below).

### 5. Speaker persistence
CosyVoice2 keeps registered voices in `model.frontend.spk2info`, an in-memory dict — gone on
restart by default. `CosyVoiceEngine` loads this dict from `LVOICE_SPEAKER_STORE_PATH` on startup
and saves it (via `torch.save`/`torch.load`) after every `register_speaker` call.
`docker-compose.yml` mounts a `lvoice_data` named volume at `/data` and points the env var there,
so registered voices survive `docker compose down`/`up`. Verified end-to-end against a real
container restart and a full `down`/`up` cycle.

### 6. Offline resources
CosyVoice's text-normalization fallback (`wetext`) calls
`modelscope.snapshot_download("pengzhendong/wetext")` on every engine load with no local-cache
check — it always tries to resolve the "master" revision over the network first, and hard-fails if
unreachable, even when the files are already cached locally. `LVOICE_OFFLINE_RESOURCES=true` makes
`CosyVoiceEngine` monkeypatch `modelscope.snapshot_download` to force `local_files_only=True`
*before* `cosyvoice`/`wetext` get imported. The Dockerfile bakes the resource into the image at
build time (replicating `frontend.py`'s exact `Normalizer()` calls) and sets this env var, so the
built image needs zero network access at runtime — verified with `docker run --network none`.

### 7. Concurrency / resource control
A bounded `asyncio.Semaphore` (`LVOICE_MAX_CONCURRENT_REQUESTS`, default 1) around engine calls,
guarded inside `run_in_threadpool` since CosyVoice inference is synchronous and CPU/GPU-bound. Keeps
the event loop responsive and prevents unbounded concurrent inference from OOMing the container.

### 8. Error handling
- Text length and prompt-audio content-type validated → 400 with a clear message.
- Unknown `speaker_id` → 400 (raised as `ValueError` in the engine, mapped to 400 in the router).
- Other inference exceptions → 500 with a generic message (no internals leaked), full traceback logged.
- Engine not yet loaded → 503 via `require_engine`, not a crash.

## Docker

### Dockerfile
- Base: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` — same image runs on GPU or CPU hosts.
- `uv` manages its own Python toolchain (`uv python install 3.10`); the base image doesn't need
  system Python.
- Layer order is deliberate: `uv sync` → bake model weights → bake wetext resources → `COPY app`.
  The model/resource bakes are placed *before* `COPY app` so app-code-only changes don't bust their
  cache and force multi-GB re-downloads on every rebuild (this was a real bug, fixed after noticing
  rebuilds were re-downloading the model on unrelated code changes).
- `LVOICE_OFFLINE_RESOURCES=true` and `LVOICE_MODEL_DIR` are baked in via `ENV`.

### docker-compose.yml + docker-compose.gpu.yml
CPU-only by default; GPU support is a separate `docker-compose.gpu.yml` overlay
(`-f docker-compose.yml -f docker-compose.gpu.yml`) rather than baked into the base file. A hard
`deploy.resources.reservations.devices` block in the base compose file would make
`docker compose up` fail outright on hosts without `nvidia-container-toolkit` — that would defeat
the "support both" device handling this project chose. Base file also defines the `lvoice_data`
volume and a `/healthz`-based healthcheck.

## Verification (all done, not just planned)

1. `docker build` succeeds; image is self-contained (model + wetext resources baked in).
2. `docker compose up` (no GPU) → `/healthz`/`/readyz` 200, `DEVICE` resolves to `cpu`.
3. All five synthesis-adjacent endpoints exercised over real HTTP against the real
   CosyVoice2-0.5B checkpoint (`/v1/tts`, `/register`, `/clone`, `/cross-lingual`, `/instruct`,
   `/voices`) — confirmed correct, playable audio each time.
4. Registered a voice, ran `docker compose restart`, then a full `down`/`up` cycle — voice survived
   both, synthesis with it still worked.
5. `docker run --network none` — engine loads and `/v1/tts/clone` produces correct audio with zero
   network calls in the logs.
6. `pytest` (mocked engine, no ML stack) + `ruff format`/`check` clean, both run repeatedly across
   every change.
7. Test page served and its exact `fetch()` calls replicated via curl for every mode; could not
   click through it in an actual browser — this sandbox has no working headless browser (Playwright
   refuses to install on this OS, no chromium binary, no snapd) and no real microphone to test the
   in-browser recorder against.
