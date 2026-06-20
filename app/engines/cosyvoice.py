"""CosyVoice2 backend, implemented against the vendored source in
third_party/CosyVoice (pinned commit, see .gitmodules / PLAN.md).
"""

import contextlib
import os
import sys
import tempfile

import numpy as np

from app.config import settings
from app.engines.base import BaseTTSEngine

_VENDOR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "CosyVoice")
_MATCHA_ROOT = os.path.join(_VENDOR_ROOT, "third_party", "Matcha-TTS")
for _path in (_VENDOR_ROOT, _MATCHA_ROOT):
    _path = os.path.abspath(_path)
    if _path not in sys.path:
        sys.path.insert(0, _path)


class CosyVoiceEngine(BaseTTSEngine):
    def __init__(self) -> None:
        # CosyVoice2 has no device constructor flag; it picks cuda/cpu internally
        # via torch.cuda.is_available(). To honor an explicit "cpu" request even on
        # a GPU host, hide the GPUs before torch initializes a CUDA context.
        if settings.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        from cosyvoice.cli.cosyvoice import CosyVoice2

        self._model = CosyVoice2(settings.model_dir)

    def register_speaker(self, speaker_id: str, prompt_text: str, prompt_wav_bytes: bytes) -> None:
        with self._prompt_wav_path(prompt_wav_bytes) as path:
            ok = self._model.add_zero_shot_spk(prompt_text, path, speaker_id)
        if not ok:
            raise RuntimeError(f"Failed to register speaker '{speaker_id}'")

    def synthesize(self, text: str, speaker_id: str, speed: float = 1.0) -> np.ndarray:
        if speaker_id not in self.list_voices():
            raise ValueError(f"Unknown speaker_id: {speaker_id}")
        return self._concat_chunks(
            self._model.inference_zero_shot(
                text, "", "", zero_shot_spk_id=speaker_id, stream=False, speed=speed
            )
        )

    def clone(
        self, text: str, prompt_wav_bytes: bytes, prompt_text: str, speed: float = 1.0
    ) -> np.ndarray:
        with self._prompt_wav_path(prompt_wav_bytes) as path:
            return self._concat_chunks(
                self._model.inference_zero_shot(text, prompt_text, path, stream=False, speed=speed)
            )

    def cross_lingual(self, text: str, prompt_wav_bytes: bytes, speed: float = 1.0) -> np.ndarray:
        with self._prompt_wav_path(prompt_wav_bytes) as path:
            return self._concat_chunks(
                self._model.inference_cross_lingual(text, path, stream=False, speed=speed)
            )

    def instruct(
        self, text: str, instruct_text: str, prompt_wav_bytes: bytes, speed: float = 1.0
    ) -> np.ndarray:
        with self._prompt_wav_path(prompt_wav_bytes) as path:
            return self._concat_chunks(
                self._model.inference_instruct2(
                    text, instruct_text, path, stream=False, speed=speed
                )
            )

    @staticmethod
    @contextlib.contextmanager
    def _prompt_wav_path(prompt_wav_bytes: bytes):
        """CosyVoice's frontend re-opens `prompt_wav` multiple times internally (once
        per target sample rate), so a path is required — a BytesIO gets exhausted
        after the first read."""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(prompt_wav_bytes)
            f.flush()
            yield f.name

    def list_voices(self) -> list[str]:
        return self._model.list_available_spks()

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    @staticmethod
    def _concat_chunks(chunks) -> np.ndarray:
        pieces = [chunk["tts_speech"].numpy() for chunk in chunks]
        if not pieces:
            raise RuntimeError("CosyVoice produced no audio")
        return np.concatenate(pieces, axis=1)


def build_engine() -> BaseTTSEngine:
    if settings.engine == "cosyvoice":
        return CosyVoiceEngine()
    raise ValueError(f"Unknown engine: {settings.engine}")
