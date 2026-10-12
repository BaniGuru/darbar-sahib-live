"""Bounded offline inference with future context around every ownership boundary.

Reference text/timestamps never enter this module. Both the original greedy
decoder and updated decoders use the same window schedule for comparisons.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16000
CORE_SECONDS = 20
CONTEXT_SECONDS = 5
MAX_WINDOW_SECONDS = CORE_SECONDS + 2 * CONTEXT_SECONDS


def value(word, key):
    return getattr(word, key) if hasattr(word, key) else word[key]


def merge_overlapping_words(previous, incoming, overlap_start, overlap_end):
    """Join on a shared word near the center of the acoustic overlap.

    Audio timestamps alone can move across a nominal cut between decodes.
    Keep every word on either side of the matched word, rather than applying
    two independent timestamp filters that can both discard a boundary word.
    """
    if not previous:
        return list(incoming), "initial"
    middle = (overlap_start + overlap_end) / 2
    anchors = []
    for left, old in enumerate(previous):
        old_start = float(value(old, "start"))
        if not overlap_start - 1 <= old_start <= overlap_end:
            continue
        for right, new in enumerate(incoming):
            new_start = float(value(new, "start"))
            if not overlap_start <= new_start <= overlap_end + 1:
                continue
            if value(old, "text") != value(new, "text") or abs(old_start - new_start) > 1.5:
                continue
            # Timestamp distance distinguishes true repeats of the same word.
            penalty = abs((old_start + new_start) / 2 - middle) + 2 * abs(old_start - new_start)
            anchors.append((penalty, left, right))
    if anchors:
        _, left, right = min(anchors)
        return list(previous[:left]) + list(incoming[right:]), "shared-word"
    # Without a shared lexical anchor, use a single actual word boundary in
    # the newer decode. Never cut a word's token sequence into two pieces.
    candidates = [(abs(float(value(word, "start")) - middle), index)
                  for index, word in enumerate(incoming)
                  if overlap_start <= float(value(word, "start")) <= overlap_end]
    if candidates:
        _, right = min(candidates)
        cut = float(value(incoming[right], "start"))
        kept = [word for word in previous if float(value(word, "start")) < cut]
        return kept + list(incoming[right:]), "word-boundary"
    return ([word for word in previous if float(value(word, "start")) < middle]
            + [word for word in incoming if float(value(word, "start")) >= middle]), "time-fallback"


def transcribe_windows(audio: np.ndarray, transcribe: Callable, audio_offset=0.0,
                       progress: Callable | None = None, *, stitch="anchors") -> dict:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    core_samples, context_samples = CORE_SECONDS * SAMPLE_RATE, CONTEXT_SECONDS * SAMPLE_RATE
    words, windows = [], []
    processed_frames = 0
    for core_start in range(0, audio.size, core_samples):
        core_end = min(audio.size, core_start + core_samples)
        start, end = max(0, core_start - context_samples), min(audio.size, core_end + context_samples)
        result = transcribe(audio[start:end], audio_offset + start / SAMPLE_RATE)
        processed_frames += int(result["encoded_frames"])
        if stitch == "anchors":
            incoming = [word for word in result["words"]
                        if float(value(word, "start")) < audio_offset + end / SAMPLE_RATE]
            words, merge_policy = merge_overlapping_words(words, incoming,
                audio_offset + start / SAMPLE_RATE,
                windows[-1]["window_end"] if windows else audio_offset + end / SAMPLE_RATE)
        else:
            merge_policy = "core-ownership"
        for word in result["words"] if stitch != "anchors" else []:
            # Accept whole words by their onset, never slice their token IDs
            # at the nominal 20-second boundary. Five seconds of right audio
            # lets a word beginning there finish inside its decoding window.
            onset = float(word.start if hasattr(word, "start") else word["start"])
            if audio_offset + core_start / SAMPLE_RATE <= onset < audio_offset + core_end / SAMPLE_RATE:
                words.append(word)
        windows.append({"core_start": audio_offset + core_start / SAMPLE_RATE,
                        "core_end": audio_offset + core_end / SAMPLE_RATE,
                        "window_start": audio_offset + start / SAMPLE_RATE,
                        "window_end": audio_offset + end / SAMPLE_RATE,
                        "merge_policy": merge_policy,
                        "encoded_frames": int(result["encoded_frames"])})
        if progress:
            progress(core_end / SAMPLE_RATE, words, windows)
    token_ids = [token for word in words for token in (word.token_ids if hasattr(word, "token_ids") else word["token_ids"])]
    text = " ".join(word.text if hasattr(word, "text") else word["text"] for word in words)
    return {"text": text, "tokens": token_ids, "words": words, "samples": int(audio.size),
            "seconds": audio.size / SAMPLE_RATE, "window_start": audio_offset,
            "window_end": audio_offset + audio.size / SAMPLE_RATE,
            "feature_frames": math.ceil(audio.size / 160),
            "encoded_frames": processed_frames,
            "timeline_encoded_frames": math.ceil(audio.size / 640),
            "processed_encoded_frames": processed_frames, "offline_windows": windows,
            "max_encoder_window_seconds": MAX_WINDOW_SECONDS,
            "inference_mode": "offline-overlapping-windows"}
