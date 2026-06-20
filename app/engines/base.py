from abc import ABC, abstractmethod

import numpy as np


class BaseTTSEngine(ABC):
    """Contract every TTS backend must implement.

    Shaped around CosyVoice2's zero-shot-first design: there is no fixed set of
    "built-in" speakers, so `synthesize`/`instruct` operate on speaker ids that were
    previously registered via `register_speaker` (which wraps zero-shot cloning).
    """

    @abstractmethod
    def register_speaker(self, speaker_id: str, prompt_text: str, prompt_wav_bytes: bytes) -> None:
        """Clone a voice from a reference clip and remember it under `speaker_id`."""
        ...

    @abstractmethod
    def synthesize(self, text: str, speaker_id: str, speed: float = 1.0) -> np.ndarray:
        """Speak `text` using a previously registered `speaker_id`."""
        ...

    @abstractmethod
    def clone(
        self, text: str, prompt_wav_bytes: bytes, prompt_text: str, speed: float = 1.0
    ) -> np.ndarray:
        """One-shot zero-shot cloning: speak `text` in the voice of the reference clip."""
        ...

    @abstractmethod
    def cross_lingual(self, text: str, prompt_wav_bytes: bytes, speed: float = 1.0) -> np.ndarray:
        """Speak `text` (typically a different language than the reference) in the
        voice of the reference clip."""
        ...

    @abstractmethod
    def instruct(
        self, text: str, instruct_text: str, prompt_wav_bytes: bytes, speed: float = 1.0
    ) -> np.ndarray:
        """Speak `text` in the voice of the reference clip, following a style/emotion
        instruction (e.g. "speak cheerfully", "use a Sichuan accent")."""
        ...

    @abstractmethod
    def list_voices(self) -> list[str]:
        """Speaker ids available via `synthesize` (registered through `register_speaker`)."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...
