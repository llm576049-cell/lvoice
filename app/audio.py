import io
import wave

import numpy as np


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 [-1, 1] audio as 16-bit PCM WAV bytes.

    Implemented with the stdlib `wave` module (not `soundfile`) so this module
    stays importable without the heavy "cosyvoice" extra installed.
    """
    pcm16 = np.clip(audio.squeeze(0), -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16.tobytes())
    return buf.getvalue()
