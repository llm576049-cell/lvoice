# lvoice

An HTTP API wrapper around [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) (Alibaba's
zero-shot TTS model), packaged to run in Docker. Supports Mandarin, Cantonese, English,
Japanese, Korean, and cross-lingual synthesis.

CosyVoice2 is zero-shot only — there are no built-in preset speakers. Every voice comes from a
short reference clip, either used once (`clone`) or registered for reuse (`register` + `tts`).

## Quick start

```sh
docker compose up --build
```

On a GPU host with `nvidia-container-toolkit` installed:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The first build downloads the CosyVoice2-0.5B checkpoint (~4.6 GB) and bakes it into the image, so
later runs/restarts don't re-download anything. Once it's up:

- `GET /healthz` / `GET /readyz` — liveness / readiness.
- `GET /` — a manual test page: type text, pick or register a voice (upload a clip or record one
  with your microphone), hear the result.

Voices registered via `/v1/tts/register` are persisted to the `lvoice_data` named volume
(`/data/spk2info.pt` inside the container), so they survive `docker compose down`/`up` —
no need to re-register after a restart.

## API

All synthesis endpoints return raw `audio/wav` bytes.

### `POST /v1/tts/clone` (multipart)

One-shot zero-shot cloning: speak `text` in the voice from a reference clip.

```sh
curl -X POST localhost:8000/v1/tts/clone \
  -F "text=收到好友从远方寄来的生日礼物，我感到非常开心。" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_audio=@reference.wav;type=audio/wav" \
  -o out.wav
```

### `POST /v1/tts/register` (multipart) + `POST /v1/tts` (JSON)

Register a voice once, then reuse it by `speaker_id` without re-uploading the reference clip.
Persisted to disk (see `LVOICE_SPEAKER_STORE_PATH` below), so it survives a restart.

```sh
curl -X POST localhost:8000/v1/tts/register \
  -F "speaker_id=alice" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_audio=@reference.wav;type=audio/wav"

curl -X POST localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "这是用已注册声音合成的句子。", "speaker_id": "alice", "speed": 1.0}' \
  -o out.wav
```

### `POST /v1/tts/cross-lingual` (multipart)

Speak `text` (typically a different language than the reference clip) in the cloned voice.
Wrap text in `<|en|>`, `<|zh|>`, `<|ja|>`, `<|yue|>`, or `<|ko|>` to mark its language.

```sh
curl -X POST localhost:8000/v1/tts/cross-lingual \
  -F "text=<|en|>This sentence is spoken in the cloned voice, cross-lingually." \
  -F "prompt_audio=@reference.wav;type=audio/wav" \
  -o out.wav
```

### `POST /v1/tts/instruct` (multipart)

Speak `text` in the cloned voice, following a style/emotion/accent instruction.

```sh
curl -X POST localhost:8000/v1/tts/instruct \
  -F "text=收到好友从远方寄来的生日礼物，我感到非常开心。" \
  -F "instruct_text=用四川话说这句话<|endofprompt|>" \
  -F "prompt_audio=@reference.wav;type=audio/wav" \
  -o out.wav
```

### `GET /v1/voices`

Lists currently registered speaker ids.

## Configuration

Environment variables (all optional, `LVOICE_` prefix):

| Variable | Default | Meaning |
|---|---|---|
| `LVOICE_ENGINE` | `cosyvoice` | TTS backend to load (see "Adding another engine" below). |
| `LVOICE_MODEL_DIR` | `/models/CosyVoice2-0.5B` | Path to the model checkpoint (baked in by the Dockerfile). |
| `LVOICE_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu`. |
| `LVOICE_MAX_TEXT_LENGTH` | `2000` | Max characters per request. |
| `LVOICE_MAX_CONCURRENT_REQUESTS` | `1` | Bounds concurrent inference calls (raise on a beefier GPU). |
| `LVOICE_SPEAKER_STORE_PATH` | `data/spk2info.pt` | Where registered speakers are persisted. Set to a mounted volume path in production (`docker-compose.yml` already does this). |
| `LVOICE_OFFLINE_RESOURCES` | `false` | If `true`, forces CosyVoice's text-normalization fallback (wetext) to use only its pre-baked local cache instead of checking modelscope.cn on startup. The Docker image bakes that cache in and sets this to `true`, so the container runs with zero network access (verified with `docker run --network none`). |

## Local development (without Docker)

Requires [uv](https://docs.astral.sh/uv/).

```sh
git submodule update --init --recursive
uv sync                       # API layer only, no ML stack
uv run pytest                 # fast, mocked-engine tests

uv sync --extra cosyvoice      # pulls in torch/CosyVoice's full dependency tree
uv run ruff format . && uv run ruff check . --fix
LVOICE_MODEL_DIR=pretrained_models/CosyVoice2-0.5B LVOICE_DEVICE=cpu \
  uv run uvicorn app.main:app --reload
```

## Adding another engine

`app/engines/base.py` defines `BaseTTSEngine`; routers depend only on that interface, never on
`CosyVoiceEngine` directly. To add a backend: implement the interface in `app/engines/<name>.py`,
extend `build_engine()` (currently in `app/engines/cosyvoice.py`) to dispatch to it, and set
`LVOICE_ENGINE=<name>`.

See [PLAN.md](PLAN.md) for the architecture/design rationale (why things are built this way).
