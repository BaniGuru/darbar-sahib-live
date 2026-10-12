"""Conservative activity detection on PCM before ASR feature normalization."""
from __future__ import annotations

import numpy as np

FRAME_SAMPLES = 320  # 20 ms at 16 kHz
_WINDOW = np.hanning(FRAME_SAMPLES)


def last_audio_sample(audio: np.ndarray) -> int | None:
    """Return the last credible signal frame's end, or None for idle noise.

    Only suppress quiet broadband noise: louder audio and even very quiet
    tonal audio remain eligible for decoding. Never modify or trim the PCM.
    These thresholds are intentionally conservative, not a speech VAD.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size:
        return None
    count = (audio.size + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    frames = np.pad(audio, (0, count * FRAME_SAMPLES - audio.size), mode="edge").reshape(count, FRAME_SAMPLES)
    # A float32 mean can leave a rounding residual on constant PCM. A Hann
    # window makes that residual look tonal; remove DC in double precision.
    frames = frames - frames.mean(axis=1, keepdims=True, dtype=np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    loud = np.flatnonzero(rms > 1e-3)
    if loud.size:
        return min(audio.size, (int(loud[-1]) + 1) * FRAME_SAMPLES)
    nonzero = np.flatnonzero(rms > 1e-8)
    if not nonzero.size:
        return None
    # 100–4000 Hz excludes DC/rumble and the high-frequency noise floor.
    power = np.abs(np.fft.rfft(frames[nonzero] * _WINDOW, axis=1)[:, 2:81]) ** 2
    flatness = np.exp(np.mean(np.log(power + 1e-30), axis=1)) / (power.mean(axis=1) + 1e-30)
    tonal = nonzero[flatness < 0.15]
    return min(audio.size, (int(tonal[-1]) + 1) * FRAME_SAMPLES) if tonal.size else None
