import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.deps import require_engine
from app.engines.base import BaseTTSEngine
from app.main import app


class FakeEngine(BaseTTSEngine):
    def __init__(self):
        self._voices: dict[str, str] = {}

    def register_speaker(self, speaker_id, prompt_text, prompt_wav_bytes):
        self._voices[speaker_id] = prompt_text

    def synthesize(self, text, speaker_id, speed=1.0):
        if speaker_id not in self._voices:
            raise ValueError(f"Unknown speaker_id: {speaker_id}")
        return np.zeros((1, 100), dtype=np.float32)

    def clone(self, text, prompt_wav_bytes, prompt_text, speed=1.0):
        return np.zeros((1, 100), dtype=np.float32)

    def cross_lingual(self, text, prompt_wav_bytes, speed=1.0):
        return np.zeros((1, 100), dtype=np.float32)

    def instruct(self, text, instruct_text, prompt_wav_bytes, speed=1.0):
        return np.zeros((1, 100), dtype=np.float32)

    def list_voices(self):
        return list(self._voices)

    @property
    def sample_rate(self):
        return 24000


@pytest.fixture
def client():
    fake = FakeEngine()
    app.dependency_overrides[require_engine] = lambda: fake
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _wav(content_type="audio/wav"):
    return ("prompt.wav", b"RIFF....WAVEfmt ", content_type)


def test_voices_initially_empty(client):
    assert client.get("/v1/voices").json() == {"voices": []}


def test_register_then_synthesize(client):
    r = client.post(
        "/v1/tts/register",
        data={"speaker_id": "alice", "prompt_text": "hello"},
        files={"prompt_audio": _wav()},
    )
    assert r.status_code == 200, r.text
    assert client.get("/v1/voices").json() == {"voices": ["alice"]}

    r = client.post("/v1/tts", json={"text": "hi there", "speaker_id": "alice"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_synthesize_unknown_speaker_is_400(client):
    r = client.post("/v1/tts", json={"text": "hi", "speaker_id": "nobody"})
    assert r.status_code == 400


def test_clone(client):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "hi there", "prompt_text": "reference"},
        files={"prompt_audio": _wav()},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_clone_rejects_bad_content_type(client):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "hi there", "prompt_text": "reference"},
        files={"prompt_audio": _wav(content_type="image/png")},
    )
    assert r.status_code == 400


def test_text_too_long_is_400(client):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "a" * 3000, "prompt_text": "reference"},
        files={"prompt_audio": _wav()},
    )
    assert r.status_code == 400


def test_cross_lingual(client):
    r = client.post(
        "/v1/tts/cross-lingual",
        data={"text": "hi there"},
        files={"prompt_audio": _wav()},
    )
    assert r.status_code == 200


def test_instruct(client):
    r = client.post(
        "/v1/tts/instruct",
        data={"text": "hi there", "instruct_text": "speak cheerfully"},
        files={"prompt_audio": _wav()},
    )
    assert r.status_code == 200
