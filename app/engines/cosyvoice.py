"""CosyVoice backend. Implemented in M2 against the real CosyVoice inference API."""

import numpy as np

from app.config import settings
from app.engines.base import BaseTTSEngine


class CosyVoiceEngine(BaseTTSEngine):
    def __init__(self) -> None:
        raise NotImplementedError("CosyVoiceEngine is implemented in milestone M2")

    def synthesize(self, text: str, speaker_id: str, speed: float = 1.0) -> np.ndarray:
        raise NotImplementedError

    def clone(self, text: str, prompt_wav_bytes: bytes, prompt_text: str) -> np.ndarray:
        raise NotImplementedError

    def cross_lingual(self, text: str, prompt_wav_bytes: bytes) -> np.ndarray:
        raise NotImplementedError

    def instruct(self, text: str, speaker_id: str, instruct_text: str) -> np.ndarray:
        raise NotImplementedError

    def list_voices(self) -> list[str]:
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError


def build_engine() -> BaseTTSEngine:
    if settings.engine == "cosyvoice":
        return CosyVoiceEngine()
    raise ValueError(f"Unknown engine: {settings.engine}")
