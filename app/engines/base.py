from abc import ABC, abstractmethod

import numpy as np


class BaseTTSEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str, speaker_id: str, speed: float = 1.0) -> np.ndarray: ...

    @abstractmethod
    def clone(self, text: str, prompt_wav_bytes: bytes, prompt_text: str) -> np.ndarray: ...

    @abstractmethod
    def cross_lingual(self, text: str, prompt_wav_bytes: bytes) -> np.ndarray: ...

    @abstractmethod
    def instruct(self, text: str, speaker_id: str, instruct_text: str) -> np.ndarray: ...

    @abstractmethod
    def list_voices(self) -> list[str]: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...
