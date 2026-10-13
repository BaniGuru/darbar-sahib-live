"""Cache-aware Punjabi RNNT streaming ASR backed by the NeMo 202/100 model."""
from __future__ import annotations

import asyncio
import io
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from threadpoolctl import threadpool_info, threadpool_limits
from audio_activity import last_audio_sample
from asr_config import available_cpu_count, configured_decoder_mode, configured_recognition, default_encoder_threads, language_model_enabled
from incremental_language_scorer import IncrementalLanguageScorer
from cached_rnnt_decoder import CachedRNNTDecoder
from gurbani_language_prior import GurbaniLanguagePrior, token_prefix_score, words as prior_words
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "kirtan_stream_model"
with (MODEL_DIR / "runtime.json").open(encoding="utf-8") as f:
    RUNTIME = json.load(f)

SAMPLE_RATE = int(RUNTIME["sample_rate"])
HOP_LENGTH = int(RUNTIME["hop_length"])
N_FFT = int(RUNTIME["n_fft"])
WIN_LENGTH = int(RUNTIME["win_length"])
N_MELS = int(RUNTIME["n_mels"])
PREEMPH = np.float32(RUNTIME["preemph"])
LOG_ZERO_GUARD = np.float32(RUNTIME["log_zero_guard_value"])
LOCAL_VOCAB_SIZE = int(RUNTIME["local_vocab_size"])
JOINT_BLANK_ID = int(RUNTIME["joint_blank_id"])
PREDICTOR_START_ID = int(RUNTIME["predictor_start_id"])
MAX_SYMBOLS_PER_STEP = int(RUNTIME["max_symbols_per_step"])
ENCODER_FRAME_SECONDS = HOP_LENGTH * 4 / SAMPLE_RATE
FEATURE_CHUNK_FRAMES = 400  # 4 seconds at the 10 ms frontend hop.
FEATURE_CHUNK_SAMPLES = FEATURE_CHUNK_FRAMES * HOP_LENGTH
MAX_PENDING_SAMPLES = 2 * FEATURE_CHUNK_SAMPLES
# The exported graph slices off two subsampled frames on every invocation,
# even though runtime.json reports drop_extra_pre_encoded=0. Supply their
# eight frontend frames as left context, rather than losing 80 ms per block.
PREENCODE_CONTEXT_FRAMES = 8
NORMALIZATION_HISTORY_SAMPLES = 16 * SAMPLE_RATE
PARTIAL_INTERVAL_SECONDS = 0.8
PARTIAL_INTERVAL_SAMPLES = int(round(SAMPLE_RATE * PARTIAL_INTERVAL_SECONDS))
SILENCE_GUARD = os.environ.get("ASR_SILENCE_GUARD", "1") != "0"
SILENCE_HANGOVER_SAMPLES = FEATURE_CHUNK_SAMPLES
ENCODER_CACHE_CHANNELS = 17
ENCODER_CACHE_DIM = 512
ENCODER_CACHE_TIME = 30
CPU_BUDGET = available_cpu_count()
ENCODER_THREADS = max(1, int(os.environ.get("ASR_ENCODER_THREADS", default_encoder_threads(CPU_BUDGET))))
BLAS_THREADS = max(1, int(os.environ.get("ASR_BLAS_THREADS", "1")))
# Apply after NumPy loads, including when the full server imported it first.
# Retain the controller for the process lifetime, rather than a short scope.
_blas_limits = threadpool_limits(limits=BLAS_THREADS, user_api="blas")
enable_language_model = language_model_enabled()
DECODER_MODE = configured_decoder_mode()
RECOGNITION_MODE, BLOCK_SECONDS = configured_recognition()
BEAM_WIDTH = max(1, int(os.environ.get("ASR_BEAM_WIDTH", "24")))
BEAM_PRUNE_LOGP = 6.0
LANGUAGE_PRIOR_WEIGHT = max(0.0, float(os.environ.get("ASR_LANGUAGE_PRIOR_WEIGHT", "0.6")))
KENLM_WEIGHT = max(0.0, float(os.environ.get("ASR_KENLM_WEIGHT", "0.3")))
KENLM_MODE = os.environ.get("ASR_KENLM_MODE", "ngram")
KENLM_WORD_BONUS = float(os.environ.get("ASR_KENLM_WORD_BONUS", "0"))
WORD_LOOKAHEAD_FRAMES = int(round(1.0 / ENCODER_FRAME_SECONDS))
WORD_BOUNDARY_TOLERANCE_FRAMES = 3  # RNNT paths can differ by 120 ms in alignment.

mel_filterbank = np.load(MODEL_DIR / RUNTIME["mel_filterbank"]).astype(np.float32)
stft_window = np.load(MODEL_DIR / RUNTIME["stft_window"]).astype(np.float32).reshape(-1)
tokenizer = spm.SentencePieceProcessor(model_file=str(MODEL_DIR / RUNTIME["tokenizer"]))
if tokenizer.get_piece_size() != LOCAL_VOCAB_SIZE:
    raise RuntimeError(f"Unexpected streaming tokenizer size: {tokenizer.get_piece_size()}")
language_prior = (GurbaniLanguagePrior.from_database(BASE_DIR / "data" / "bani.db")
                  if LANGUAGE_PRIOR_WEIGHT and (BASE_DIR / "data" / "bani.db").exists() else None)
kenlm_prior = None
if enable_language_model and KENLM_WEIGHT:
    from kenlm_language_prior import KenLMWordPrior
    kenlm_prior = KenLMWordPrior(Path(os.environ.get("ASR_KENLM_PATH", MODEL_DIR / "gurbani_3gram.bin")),
                                mode=KENLM_MODE, word_bonus=KENLM_WORD_BONUS)

def session_options_for(threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


encoder_session = ort.InferenceSession(
    str(MODEL_DIR / RUNTIME["encoder"]), sess_options=session_options_for(ENCODER_THREADS),
    providers=["CPUExecutionProvider"],
)
decoder_session = ort.InferenceSession(
    str(MODEL_DIR / RUNTIME["decoder_joint"]), sess_options=session_options_for(1),
    providers=["CPUExecutionProvider"],
)
if os.environ.get("ASR_CACHE_PREDICTOR", "1") != "0":
    decoder_session = CachedRNNTDecoder(decoder_session, str(MODEL_DIR / RUNTIME["decoder_joint"]),
                                       session_options_for(1))
encoder_inputs = [x.name for x in encoder_session.get_inputs()]
decoder_inputs = [x.name for x in decoder_session.get_inputs()]
INFERENCE_LOCK = asyncio.Lock()

app = FastAPI(title="Gurbani Live Streaming ASR")
PUBLIC_DIR = BASE_DIR / "public"
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


def normalize_features(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(features, axis=1, keepdims=True, ddof=int(features.shape[1] > 1), dtype=np.float32)
    return ((features - mean) / (std + np.float32(1e-5))).astype(np.float32)


def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.empty((1, N_MELS, 0), dtype=np.float32)
    emphasized = np.empty_like(audio)
    emphasized[0] = audio[0]
    if audio.size > 1:
        emphasized[1:] = audio[1:] - PREEMPH * audio[:-1]
    pad = N_FFT // 2
    padded = np.pad(emphasized, (pad, pad), mode="reflect" if emphasized.size > pad else "constant")
    n_frames = (padded.size - N_FFT) // HOP_LENGTH + 1
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH][:n_frames]
    window = np.zeros(N_FFT, dtype=np.float32)
    offset = (N_FFT - WIN_LENGTH) // 2
    window[offset:offset + WIN_LENGTH] = stft_window
    spectrum = np.fft.rfft(np.asarray(frames, np.float32) * window[None, :], n=N_FFT, axis=-1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    mel = np.matmul(power, mel_filterbank.T).astype(np.float32)
    features = np.log(mel + LOG_ZERO_GUARD).T
    return normalize_features(features)[None].astype(np.float32, copy=False)


@dataclass
class TimedToken:
    token_id: int
    encoder_frame: int


def new_encoder_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((1, ENCODER_CACHE_CHANNELS, 202, ENCODER_CACHE_DIM), np.float32),
        np.zeros((1, ENCODER_CACHE_CHANNELS, ENCODER_CACHE_DIM, ENCODER_CACHE_TIME), np.float32),
        np.zeros((1,), np.int64),
    )


def run_encoder(features: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray]):
    if features.shape[2] == 0:
        raise ValueError("No frontend frames available for encoder input.")
    if features.shape[2] < 20:
        features = np.pad(features, ((0, 0), (0, 0), (0, 20 - features.shape[2])))
    cache_channel, cache_time, cache_length = cache
    result = encoder_session.run(None, {
        encoder_inputs[0]: features,
        encoder_inputs[1]: np.asarray([features.shape[2]], dtype=np.int64),
        encoder_inputs[2]: cache_channel,
        encoder_inputs[3]: cache_time,
        encoder_inputs[4]: cache_length,
    })
    return result[0], int(result[1].reshape(-1)[0]), tuple(result[2:])


def new_predictor_state():
    h, c = np.zeros((1, 1, 640), np.float32), np.zeros((1, 1, 640), np.float32)
    h.setflags(write=False)
    c.setflags(write=False)
    return h, c, PREDICTOR_START_ID


@dataclass
class DecoderHypothesis:
    # Only the unsettled suffix is retained here. Predictor state still
    # includes the entire utterance, even after a settled prefix is removed.
    tokens: tuple[TimedToken, ...]
    token_ids: tuple[int, ...]
    predictor: tuple[np.ndarray, np.ndarray, int]
    score: float = 0.0
    predictor_states: tuple[tuple[np.ndarray, np.ndarray, int], ...] = ()


def new_decoder_beams() -> list[DecoderHypothesis]:
    return [DecoderHypothesis((), (), new_predictor_state())]


def _merge_hypothesis(candidates: dict, hypothesis: DecoderHypothesis) -> None:
    previous = candidates.get(hypothesis.token_ids)
    if previous is None:
        candidates[hypothesis.token_ids] = hypothesis
        return
    # Different RNNT alignments can emit the same tokens. Sum their
    # probabilities, keeping the more likely alignment's timestamps.
    best = hypothesis if hypothesis.score > previous.score else previous
    candidates[hypothesis.token_ids] = DecoderHypothesis(
        best.tokens, best.token_ids, best.predictor,
        float(np.logaddexp(previous.score, hypothesis.score)),
        best.predictor_states,
    )


def decode_encoder_beams(
    encoded: np.ndarray, length: int, beams: list[DecoderHypothesis], frame_offset: int,
    *, require_word_start: bool = False, previous_word: str = "", previous_words: tuple[str, ...] = (),
) -> list[DecoderHypothesis]:
    """Expose the original decoder for controlled live accuracy comparisons."""
    decode = (_decode_encoder_beams_optimized if DECODER_MODE == "optimized"
              else _decode_encoder_beams_accuracy)
    return decode(encoded, length, beams, frame_offset,
                  require_word_start=require_word_start, previous_word=previous_word,
                  previous_words=previous_words)


def _decode_encoder_beams_accuracy(
    encoded: np.ndarray, length: int, beams: list[DecoderHypothesis], frame_offset: int,
    *, require_word_start: bool = False, previous_word: str = "", previous_words: tuple[str, ...] = (),
) -> list[DecoderHypothesis]:
    """Keep competing word interpretations alive across acoustic commits.

    Batch the predictor/joint calls across a small, probability-pruned beam.
    A blank advances acoustic time without consuming predictor state.
    """
    start_labels = np.asarray([label for label in range(LOCAL_VOCAB_SIZE)
                               if tokenizer.id_to_piece(label).startswith("▁")], dtype=np.int32) if require_word_start else None
    prior_scores = {}
    def prior_score(ids):
        if language_prior is None and kenlm_prior is None:
            return 0.0
        if ids not in prior_scores:
            value = (LANGUAGE_PRIOR_WEIGHT * token_prefix_score(language_prior, tokenizer, ids, previous_word)
                     if language_prior is not None else 0.0)
            if kenlm_prior is not None:
                value += KENLM_WEIGHT * kenlm_prior.token_score(tokenizer, ids, previous_words)
            prior_scores[ids] = value
        return prior_scores[ids]
    expansion_width = BEAM_WIDTH * 2 if language_prior is not None or kenlm_prior is not None else BEAM_WIDTH
    maximum_prior_reward = 12 * LANGUAGE_PRIOR_WEIGHT
    if kenlm_prior is not None:
        maximum_prior_reward += KENLM_WEIGHT * kenlm_prior.max_token_reward
    for frame in range(length):
        active = beams
        completed: dict[tuple[int, ...], DecoderHypothesis] = {}
        for symbols in range(MAX_SYMBOLS_PER_STEP + 1):
            if not active:
                break
            batch = len(active)
            result = decoder_session.run(None, {
                decoder_inputs[0]: np.repeat(encoded[:, :, frame:frame + 1], batch, axis=0),
                decoder_inputs[1]: np.asarray([[h.predictor[2]] for h in active], dtype=np.int32),
                decoder_inputs[2]: np.ones(batch, dtype=np.int32),
                decoder_inputs[3]: np.concatenate([h.predictor[0] for h in active], axis=1),
                decoder_inputs[4]: np.concatenate([h.predictor[1] for h in active], axis=1),
            })
            logits = result[0].reshape(batch, -1, LOCAL_VOCAB_SIZE + 1)[:, 0, :].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            for index, hypothesis in enumerate(active):
                _merge_hypothesis(completed, DecoderHypothesis(
                    hypothesis.tokens, hypothesis.token_ids, hypothesis.predictor,
                    hypothesis.score + float(log_probs[index, JOINT_BLANK_ID]),
                    hypothesis.predictor_states,
                ))
            best_completed = max(h.score for h in completed.values())
            if symbols == MAX_SYMBOLS_PER_STEP:
                break
            expanded: dict[tuple[int, ...], DecoderHypothesis] = {}
            for index, hypothesis in enumerate(active):
                # Every label extending this parent has the same newly
                # computed recurrent state. These arrays remain immutable.
                next_h = result[2][:, index:index + 1, :].copy()
                next_c = result[3][:, index:index + 1, :].copy()
                if require_word_start and not hypothesis.tokens:
                    # A settled word cannot acquire more suffix characters.
                    # Leave the new word's spelling free, but preserve its
                    # confirmed boundary even if normalization later changes.
                    labels = start_labels[np.argsort(log_probs[index, start_labels])[-expansion_width:]]
                else:
                    labels = np.argsort(log_probs[index, :LOCAL_VOCAB_SIZE])[-expansion_width:]
                for label in labels:
                    acoustic_score = hypothesis.score + float(log_probs[index, label])
                    # A token can complete one word and begin the next. The
                    # bounded prior cannot rescue a candidate below this bound.
                    if acoustic_score < best_completed - BEAM_PRUNE_LOGP - maximum_prior_reward:
                        continue
                    candidate_ids = hypothesis.token_ids + (int(label),)
                    score = (acoustic_score
                             + prior_score(candidate_ids) - prior_score(hypothesis.token_ids))
                    if score < best_completed - BEAM_PRUNE_LOGP:
                        continue
                    label = int(label)
                    predictor = (next_h, next_c, label)
                    _merge_hypothesis(expanded, DecoderHypothesis(
                        hypothesis.tokens + (TimedToken(label, frame_offset + frame),),
                        hypothesis.token_ids + (label,),
                        predictor, score, hypothesis.predictor_states + (predictor,),
                    ))
            active = sorted(expanded.values(), key=lambda h: h.score, reverse=True)[:BEAM_WIDTH]
        beams = sorted(completed.values(), key=lambda h: h.score, reverse=True)[:BEAM_WIDTH]
        top_score = beams[0].score
        beams = [DecoderHypothesis(h.tokens, h.token_ids, h.predictor, h.score - top_score, h.predictor_states)
                 for h in beams if h.score >= top_score - BEAM_PRUNE_LOGP]
    return beams



def _decode_encoder_beams_optimized(
    encoded: np.ndarray, length: int, beams: list[DecoderHypothesis], frame_offset: int,
    *, require_word_start: bool = False, previous_word: str = "", previous_words: tuple[str, ...] = (),
) -> list[DecoderHypothesis]:
    """Keep competing word interpretations alive across acoustic commits.

    Batch the predictor/joint calls across a small, probability-pruned beam.
    A blank advances acoustic time without consuming predictor state.
    """
    start_labels = np.asarray([label for label in range(LOCAL_VOCAB_SIZE)
                               if tokenizer.id_to_piece(label).startswith("▁")], dtype=np.int32) if require_word_start else None
    scorer = (IncrementalLanguageScorer(tokenizer, language_prior, kenlm_prior,
              LANGUAGE_PRIOR_WEIGHT, KENLM_WEIGHT, previous_word, previous_words)
              if language_prior is not None or kenlm_prior is not None else None)
    expansion_width = BEAM_WIDTH * 2 if language_prior is not None or kenlm_prior is not None else BEAM_WIDTH
    maximum_prior_reward = 12 * LANGUAGE_PRIOR_WEIGHT if language_prior is not None else 0.0
    if kenlm_prior is not None:
        maximum_prior_reward += KENLM_WEIGHT * kenlm_prior.max_token_reward
    for frame in range(length):
        active = beams
        completed: dict[tuple[int, ...], DecoderHypothesis] = {}
        for symbols in range(MAX_SYMBOLS_PER_STEP + 1):
            if not active:
                break
            batch = len(active)
            if isinstance(decoder_session, CachedRNNTDecoder):
                output, next_states = decoder_session.run_hypotheses(
                    encoded[:, :, frame:frame + 1], [h.predictor for h in active])
            else:
                result = decoder_session.run(None, {
                    decoder_inputs[0]: np.repeat(encoded[:, :, frame:frame + 1], batch, axis=0),
                    decoder_inputs[1]: np.asarray([[h.predictor[2]] for h in active], dtype=np.int32),
                    decoder_inputs[2]: np.ones(batch, dtype=np.int32),
                    decoder_inputs[3]: np.concatenate([h.predictor[0] for h in active], axis=1),
                    decoder_inputs[4]: np.concatenate([h.predictor[1] for h in active], axis=1),
                })
                output = result[0]
                next_states = [(result[2][:, i:i + 1].copy(), result[3][:, i:i + 1].copy())
                               for i in range(batch)]
            logits = output.reshape(batch, -1, LOCAL_VOCAB_SIZE + 1)[:, 0, :].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            for index, hypothesis in enumerate(active):
                _merge_hypothesis(completed, DecoderHypothesis(
                    hypothesis.tokens, hypothesis.token_ids, hypothesis.predictor,
                    hypothesis.score + float(log_probs[index, JOINT_BLANK_ID]),
                    hypothesis.predictor_states,
                ))
            best_completed = max(h.score for h in completed.values())
            if symbols == MAX_SYMBOLS_PER_STEP:
                break
            expanded: dict[tuple[int, ...], DecoderHypothesis] = {}
            for index, hypothesis in enumerate(active):
                parent_state = scorer.state(hypothesis.token_ids) if scorer else None
                parent_prior = scorer.score(hypothesis.token_ids) if scorer else 0.0
                # Every label extending this parent has the same newly
                # computed recurrent state. These arrays remain immutable.
                next_h, next_c = next_states[index]
                if require_word_start and not hypothesis.tokens:
                    # A settled word cannot acquire more suffix characters.
                    # Leave the new word's spelling free, but preserve its
                    # confirmed boundary even if normalization later changes.
                    labels = start_labels[np.argsort(log_probs[index, start_labels])[-expansion_width:]]
                else:
                    labels = np.argsort(log_probs[index, :LOCAL_VOCAB_SIZE])[-expansion_width:]
                if scorer:
                    bounds = scorer.upper_bounds(parent_state, labels, maximum_prior_reward)
                    labels = labels[hypothesis.score + log_probs[index, labels] >=
                                    best_completed - BEAM_PRUNE_LOGP - bounds]
                for label in labels:
                    acoustic_score = hypothesis.score + float(log_probs[index, label])
                    # A token can complete one word and begin the next. The
                    # bounded prior cannot rescue a candidate below this bound.
                    if acoustic_score < best_completed - BEAM_PRUNE_LOGP - maximum_prior_reward:
                        continue
                    state = scorer.extend(parent_state, int(label)) if scorer else None
                    if scorer:
                        candidate_prior = (state.score if state is not None else
                                           scorer.fallback(hypothesis.token_ids + (int(label),)))
                    else:
                        candidate_prior = 0.0
                    score = acoustic_score + candidate_prior - parent_prior
                    if score < best_completed - BEAM_PRUNE_LOGP:
                        continue
                    label = int(label)
                    candidate_ids = hypothesis.token_ids + (label,)
                    if scorer:
                        scorer.states[candidate_ids] = state
                    predictor = (next_h, next_c, label)
                    _merge_hypothesis(expanded, DecoderHypothesis(
                        hypothesis.tokens + (TimedToken(label, frame_offset + frame),),
                        hypothesis.token_ids + (label,),
                        predictor, score, hypothesis.predictor_states + (predictor,),
                    ))
            active = sorted(expanded.values(), key=lambda h: h.score, reverse=True)[:BEAM_WIDTH]
        beams = sorted(completed.values(), key=lambda h: h.score, reverse=True)[:BEAM_WIDTH]
        top_score = beams[0].score
        beams = [DecoderHypothesis(h.tokens, h.token_ids, h.predictor, h.score - top_score, h.predictor_states)
                 for h in beams if h.score >= top_score - BEAM_PRUNE_LOGP]
    return beams


def settled_word_prefix(beams: list[DecoderHypothesis], frame_end: int) -> int:
    """Confirm a word end through a shared next-word acoustic boundary.

    Competing spellings can describe the same completed word. Requiring
    their entire token prefixes to match would stall finalization forever.
    A false split is different: its internal word start is absent from the
    competing unsplit interpretation, so that boundary cannot settle.
    """
    boundaries = [[(index, token) for index, token in enumerate(h.tokens)
                   if index and tokenizer.id_to_piece(token.token_id).startswith("▁")]
                  for h in beams]
    for index, token in reversed(boundaries[0]):
        if token.encoder_frame > frame_end - WORD_LOOKAHEAD_FRAMES:
            continue
        if all(any(other.token_id == token.token_id and
                   abs(other.encoder_frame - token.encoder_frame) <= WORD_BOUNDARY_TOLERANCE_FRAMES
                   for _, other in starts) for starts in boundaries[1:]):
            prefix = beams[0].token_ids[:index]
            # A shorter spelling can be a token prefix of a longer word.
            # Keep that entire word open if a compatible path continues it;
            # committing its shared letters would split the longer word and
            # incorrectly advance the LM context (ਗੁਰ + ੂ instead of ਗੁਰੂ).
            if any(h.token_ids[:index] == prefix and
                   (len(h.tokens) <= index or
                    not tokenizer.id_to_piece(h.tokens[index].token_id).startswith("▁"))
                   for h in beams[1:]):
                continue
            return index
    return 0


def tokens_to_words(tokens: list[TimedToken], audio_offset: float = 0.0) -> list[dict[str, Any]]:
    groups: list[list[TimedToken]] = []
    for token in tokens:
        piece = tokenizer.id_to_piece(token.token_id)
        if piece.startswith("▁") and groups:
            groups.append([])
        if not groups:
            groups.append([])
        groups[-1].append(token)
    words = []
    for i, group in enumerate(groups):
        text = tokenizer.decode([x.token_id for x in group]).strip()
        if not text:
            continue
        start = audio_offset + group[0].encoder_frame * ENCODER_FRAME_SECONDS
        end_frame = groups[i + 1][0].encoder_frame if i + 1 < len(groups) else group[-1].encoder_frame + 1
        end = audio_offset + end_frame * ENCODER_FRAME_SECONDS
        words.append({"text": text, "start": round(start, 3), "end": round(max(start + ENCODER_FRAME_SECONDS, end), 3), "stable_count": 1})
    return words


@dataclass
class PendingDecode:
    beams: list[DecoderHypothesis]
    first_beams: list[DecoderHypothesis]
    first_cache: tuple[np.ndarray, np.ndarray, np.ndarray]
    last_cache: tuple[np.ndarray, np.ndarray, np.ndarray]
    first_frames: int
    frame_end: int


class StreamingRecognizer:
    """800 ms previews, delayed 4 s cache steps, and at most 8 s of pending PCM.

    Encoder caches and the settled-word predictor have separate frontiers.
    A word can finish after the first four seconds and be settled with its
    original predictor checkpoint, while only those first four seconds are
    cached. Later decoding skips the settled word instead of emitting it twice.
    """
    def __init__(self):
        self.cache = new_encoder_cache()
        self.beams = new_decoder_beams()
        self.history_audio = np.empty(0, dtype=np.float32)
        self.pending_audio = np.empty(0, dtype=np.float32)
        self.committed_tokens: list[TimedToken] = []
        self.committed_last_word = ""
        self.committed_lm_context: tuple[str, ...] = ()
        self._display_token_count = -1
        self._committed_display_words: list[dict[str, Any]] = []
        self._committed_display_text = ""
        self.preview_tokens: list[TimedToken] | None = None
        self.total_samples = 0
        self.encoder_frame_offset = 0
        self.decoder_frame_offset = 0
        self.peak_pending_samples = 0
        self.cache_events: list[dict[str, Any]] = []
        self.hold_reason = "collecting"
        self.finished = False
        self.last_signal_sample: int | None = None
        self.silence_suppressed = False

    def _audio_decode_allowed(self) -> bool:
        if not SILENCE_GUARD:
            self.silence_suppressed = False
            return True
        signal_end = last_audio_sample(self.pending_audio)
        if signal_end is not None:
            signal_sample = self.encoder_frame_offset * HOP_LENGTH * 4 + signal_end
            self.last_signal_sample = max(self.last_signal_sample or 0, signal_sample)
            self.silence_suppressed = False
        else:
            self.silence_suppressed = (self.last_signal_sample is None or
                                      self.total_samples - self.last_signal_sample >= SILENCE_HANGOVER_SAMPLES)
        return not self.silence_suppressed

    def _encode_pending(self) -> PendingDecode:
        initial_idle = self.last_signal_sample is None and self.silence_suppressed
        decode_allowed = self._audio_decode_allowed()
        if initial_idle and decode_allowed:
            # Normalized startup noise must not become phonetic left context
            # for the first real words. Reset once, before any signal has been
            # heard; keep PCM, history, predictor and absolute clocks intact.
            self.cache = new_encoder_cache()
        # Advance acoustic state even on idle noise. Skipping/zeroing PCM or
        # resetting after ordinary pauses would change ongoing word context.
        # Normalize the entire available window, including lookahead, before
        # slicing it into the exported graph's supported four-second blocks.
        context_audio = np.concatenate((self.history_audio, self.pending_audio))
        features = preprocess_audio(context_audio)
        history_frames = self.history_audio.size // HOP_LENGTH
        if history_frames < PREENCODE_CONTEXT_FRAMES:
            padding = PREENCODE_CONTEXT_FRAMES - history_frames
            features = np.pad(features, ((0, 0), (0, 0), (padding, 0)), mode="edge")
            history_frames += padding
        cache, beams = self.cache, self.beams
        frame_start = self.encoder_frame_offset
        for start in range(0, self.pending_audio.size, FEATURE_CHUNK_SAMPLES):
            samples = min(FEATURE_CHUNK_SAMPLES, self.pending_audio.size - start)
            block_start = history_frames + start // HOP_LENGTH
            feature_count = int(np.ceil(samples / HOP_LENGTH))
            block = features[:, :, block_start - PREENCODE_CONTEXT_FRAMES:block_start + feature_count]
            encoded, length, cache = run_encoder(block, cache)
            valid_frames = min(length, int(np.ceil(samples / (HOP_LENGTH * 4))))
            skip = min(valid_frames, max(0, self.decoder_frame_offset - frame_start))
            if decode_allowed and skip < valid_frames:
                beams = decode_encoder_beams(encoded[:, :, skip:valid_frames], valid_frames - skip,
                                             beams, frame_start + skip,
                                             require_word_start=bool(self.committed_tokens),
                                             previous_word=self.committed_last_word,
                                             previous_words=self.committed_lm_context)
            frame_start += valid_frames
            if start == 0:
                first_cache, first_beams, first_frames = cache, beams, valid_frames
        return PendingDecode(beams, first_beams, first_cache, cache, first_frames, frame_start)

    def _cache_confirmation(self, result: PendingDecode) -> tuple[bool, int, int | None]:
        boundary = self.encoder_frame_offset + result.first_frames
        if self.decoder_frame_offset >= boundary:
            return True, 0, None  # The boundary is already inside a settled word.
        best = result.beams[0]
        starts = [i for i, token in enumerate(best.tokens)
                  if i == 0 or tokenizer.id_to_piece(token.token_id).startswith("▁")]
        before_boundary = [i for i in starts if best.tokens[i].encoder_frame < boundary]
        if not before_boundary:
            self.hold_reason = "empty-lookahead"
            return result.frame_end >= boundary + WORD_LOOKAHEAD_FRAMES, 0, None
        target = before_boundary[-1]
        following = next((i for i in starts if i > target), None)
        if following is None:
            self.hold_reason = "boundary-word"
            return False, 0, None
        end = best.tokens[following]
        if result.frame_end < end.encoder_frame + WORD_LOOKAHEAD_FRAMES:
            self.hold_reason = "word-lookahead"
            return False, 0, None
        for hypothesis in result.beams[1:]:
            if not any(i and token.token_id == end.token_id and
                       tokenizer.id_to_piece(token.token_id).startswith("▁") and
                       abs(token.encoder_frame - end.encoder_frame) <= WORD_BOUNDARY_TOLERANCE_FRAMES
                       for i, token in enumerate(hypothesis.tokens)):
                self.hold_reason = "disputed-boundary"
                return False, 0, None
        if settled_word_prefix(result.beams, result.frame_end) < following:
            self.hold_reason = "unfinished-word-alternative"
            return False, 0, None
        return True, following, end.encoder_frame

    def _promote_cache(self, result: PendingDecode, settled: int, word_end: int | None,
                       reason: str) -> None:
        window_samples = self.pending_audio.size
        best = result.beams[0]
        first_end = self.encoder_frame_offset + result.first_frames
        previously_committed = len(self.committed_tokens)
        if settled:
            # This checkpoint can lie after the acoustic cache frontier.
            # All tokens of the completed boundary word settle together.
            self.committed_tokens.extend(best.tokens[:settled])
            predictor = best.predictor_states[settled - 1]
            self.beams = [DecoderHypothesis((), (), predictor)]
            self.decoder_frame_offset = word_end
        else:
            # At the hard limit retain every viable interpretation of an
            # unfinished word. Prefer the first-block path supported by the
            # complete provisional window when settling earlier whole words.
            prefix_ids = tuple(token.token_id for token in best.tokens if token.encoder_frame < first_end)
            first_beams = sorted(result.first_beams, key=lambda h: h.token_ids != prefix_ids)
            settled = settled_word_prefix(first_beams, first_end)
            prefix = first_beams[0].token_ids[:settled]
            self.committed_tokens.extend(first_beams[0].tokens[:settled])
            self.beams = [DecoderHypothesis(h.tokens[settled:], h.token_ids[settled:], h.predictor,
                                           h.score, h.predictor_states[settled:])
                          for h in first_beams if h.token_ids[:settled] == prefix]
            self.decoder_frame_offset = max(self.decoder_frame_offset, first_end)
        self.cache = result.first_cache
        if settled:
            new_words = prior_words(tokenizer.decode([t.token_id for t in self.committed_tokens[previously_committed:]]))
            context_words = kenlm_prior.order - 1 if kenlm_prior is not None else 2
            self.committed_lm_context = (self.committed_lm_context + tuple(new_words))[-context_words:]
            last = self.committed_tokens[-1:]
            for token in reversed(self.committed_tokens[:-1]):
                if tokenizer.id_to_piece(last[0].token_id).startswith("▁"):
                    break
                last.insert(0, token)
            self.committed_last_word = tokenizer.decode([t.token_id for t in last]).strip()
        self.encoder_frame_offset = first_end
        cached_audio = self.pending_audio[:FEATURE_CHUNK_SAMPLES]
        self.history_audio = np.concatenate((self.history_audio, cached_audio))[-NORMALIZATION_HISTORY_SAMPLES:]
        self.pending_audio = self.pending_audio[FEATURE_CHUNK_SAMPLES:]
        self.preview_tokens = list(best.tokens[settled:])
        self.cache_events.append({"at_seconds": self.total_samples / SAMPLE_RATE,
                                  "cached_until": first_end * ENCODER_FRAME_SECONDS,
                                  "window_seconds": window_samples / SAMPLE_RATE,
                                  "word_end_seconds": word_end * ENCODER_FRAME_SECONDS if word_end is not None else None,
                                  "reason": reason})
        self.hold_reason = "collecting"

    def _advance(self, preview: bool) -> None:
        if not preview and self.pending_audio.size < FEATURE_CHUNK_SAMPLES:
            return
        result = self._encode_pending()
        self.preview_tokens = list(result.beams[0].tokens)
        if self.pending_audio.size >= FEATURE_CHUNK_SAMPLES:
            ready, settled, word_end = self._cache_confirmation(result)
            if ready or self.pending_audio.size == MAX_PENDING_SAMPLES:
                self._promote_cache(result, settled, word_end,
                                    "word-confirmed" if ready else "window-limit")

    def update(self, audio: np.ndarray, *, preview: bool = True) -> dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.finished:
            raise RuntimeError("Cannot append audio after finish.")
        start = 0
        while start < audio.size:
            # Evaluate the same 800 ms clock regardless of packet sizes, so
            # full-file and live inference make identical lookahead decisions.
            until_tick = PARTIAL_INTERVAL_SAMPLES - self.total_samples % PARTIAL_INTERVAL_SAMPLES
            end = min(start + until_tick, audio.size)
            chunk = audio[start:end]
            self.pending_audio = np.concatenate((self.pending_audio, chunk))
            self.total_samples += chunk.size
            self.peak_pending_samples = max(self.peak_pending_samples, self.pending_audio.size)
            if self.pending_audio.size > MAX_PENDING_SAMPLES:
                raise RuntimeError("Pending audio exceeded the eight-second limit.")
            if self.total_samples % PARTIAL_INTERVAL_SAMPLES == 0:
                self._advance(preview)
            start = end
        return self._snapshot()

    def _snapshot(self, final: bool = False) -> dict[str, Any]:
        committed_display_tokens = self.committed_tokens
        partial_display_tokens = [] if final else (
            self.preview_tokens if self.preview_tokens is not None else list(self.beams[0].tokens)
        )
        if len(committed_display_tokens) != self._display_token_count:
            self._committed_display_words = tokens_to_words(committed_display_tokens)
            self._committed_display_text = tokenizer.decode([x.token_id for x in committed_display_tokens]).strip()
            self._display_token_count = len(committed_display_tokens)
        committed_words = list(self._committed_display_words)
        partial_words = tokens_to_words(partial_display_tokens)
        if committed_words and partial_display_tokens:
            # The final settled word ends at the first partial word's start.
            # Copy that record so previously returned snapshots stay immutable.
            committed_words[-1] = {**committed_words[-1], "end": round(max(
                committed_words[-1]["start"] + ENCODER_FRAME_SECONDS,
                partial_display_tokens[0].encoder_frame * ENCODER_FRAME_SECONDS), 3)}
        committed_text = self._committed_display_text
        partial_text = tokenizer.decode([x.token_id for x in partial_display_tokens]).strip()
        return {
            "type": "final" if final else "partial",
            "audio_active": not self.silence_suppressed,
            "text": " ".join(text for text in (committed_text, partial_text) if text),
            "committed_text": committed_text,
            "partial_text": partial_text,
            "committed_until": self.encoder_frame_offset * ENCODER_FRAME_SECONDS,
            "cache_committed_until": self.encoder_frame_offset * ENCODER_FRAME_SECONDS,
            "text_committed_until": committed_words[-1]["end"] if committed_words else 0.0,
            "committed_words": committed_words,
            "partial_words": partial_words,
            "window_start": 0.0,
            "window_end": self.total_samples / SAMPLE_RATE,
            "window_seconds": self.total_samples / SAMPLE_RATE,
            "samples": self.total_samples,
            "pending_window_seconds": self.pending_audio.size / SAMPLE_RATE,
            "max_window_seconds": MAX_PENDING_SAMPLES / SAMPLE_RATE,
            "cache_hold_reason": self.hold_reason,
        }

    def finish(self, audio: np.ndarray | None = None) -> dict[str, Any]:
        if audio is not None and np.asarray(audio).size:
            self.update(audio, preview=False)
        if not self.finished and self.pending_audio.size:
            result = self._encode_pending()
            best = result.beams[0]
            if kenlm_prior is not None:
                # EOF completes the last word without inventing a sentence
                # boundary or dropping audio shorter than the refresh period.
                def final_score(hypothesis):
                    ids = hypothesis.token_ids
                    completed = kenlm_prior.full_score(tokenizer.decode(ids) + " ", self.committed_lm_context)
                    partial = kenlm_prior.token_score(tokenizer, ids, self.committed_lm_context)
                    return hypothesis.score + KENLM_WEIGHT * (completed - partial)
                best = max(result.beams, key=final_score)
            self.committed_tokens.extend(best.tokens)
            self.encoder_frame_offset = result.frame_end
            self.decoder_frame_offset = result.frame_end
            self.cache = result.last_cache
            self.pending_audio = np.empty(0, dtype=np.float32)
        if not self.finished:
            self.beams = new_decoder_beams()
            self.preview_tokens = None
        self.finished = True
        self.hold_reason = "finished"
        tokens = self.committed_tokens
        text = tokenizer.decode([x.token_id for x in tokens]).strip()
        return {**self._snapshot(final=True), "final_text": text, "tokens": [x.token_id for x in tokens],
                "words": tokens_to_words(tokens), "samples": self.total_samples,
                "seconds": self.total_samples / SAMPLE_RATE,
                "feature_frames": int(np.ceil(self.total_samples / HOP_LENGTH)),
                "encoded_frames": self.encoder_frame_offset,
                "peak_pending_window_seconds": self.peak_pending_samples / SAMPLE_RATE,
                "cache_events": self.cache_events}


def new_recognizer():
    if RECOGNITION_MODE == "blocks":
        from block_recognizer import BlockRecognizer
        return BlockRecognizer(BLOCK_SECONDS)
    return StreamingRecognizer()


def transcribe_pcm_sync(audio: np.ndarray) -> dict[str, Any]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    recognizer = new_recognizer()
    recognizer.update(audio, preview=False)
    return recognizer.finish()


async def transcribe_pcm(audio: np.ndarray) -> dict[str, Any]:
    async with INFERENCE_LOCK:
        return await asyncio.to_thread(transcribe_pcm_sync, audio)


def read_wav_bytes(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected 16 kHz mono 16-bit PCM WAV audio.")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


@app.get("/")
async def index():
    page = PUBLIC_DIR / "live.html"
    return FileResponse(page) if page.exists() else {"message": "Gurbani Live Streaming ASR"}


@app.get("/health")
async def health():
    return {"status": "ok", "provider": "CPUExecutionProvider", "sample_rate": SAMPLE_RATE,
            "encoder_context": RUNTIME["att_context_size"], "encoder_frame_seconds": ENCODER_FRAME_SECONDS,
            "text_commit_seconds": 4.0, "cache_commit_seconds": 4.0,
            "word_commit_policy": "shared-word-boundary" if RECOGNITION_MODE == "blocks" else "adaptive-boundary-lookahead",
            "max_pending_window_seconds": MAX_PENDING_SAMPLES / SAMPLE_RATE,
            "encoder_update_seconds": BLOCK_SECONDS if RECOGNITION_MODE == "blocks" else PARTIAL_INTERVAL_SAMPLES / SAMPLE_RATE,
            "normalization_lookahead_seconds": 4.0 if RECOGNITION_MODE == "blocks" else None,
            "encoder_threads": ENCODER_THREADS, "decoder_beam_width": BEAM_WIDTH,
            "blas_threads": BLAS_THREADS,
            "blas_pools": [{"internal_api": p["internal_api"], "num_threads": p["num_threads"]}
                           for p in threadpool_info() if p["user_api"] == "blas"],
            "cpu_budget": CPU_BUDGET,
            "decoder_mode": DECODER_MODE,
            "recognition_mode": RECOGNITION_MODE, "block_seconds": BLOCK_SECONDS,
            "enable_language_model": enable_language_model,
            "silence_guard": SILENCE_GUARD,
            "silence_hangover_seconds": SILENCE_HANGOVER_SAMPLES / SAMPLE_RATE,
            "predictor_cache": isinstance(decoder_session, CachedRNNTDecoder),
            "predictor_cache_max_entries": getattr(decoder_session, "cache_size", 0),
            "language_prior": "repository-gurbani-word-context" if language_prior is not None else "disabled",
            "language_prior_weight": LANGUAGE_PRIOR_WEIGHT if language_prior is not None else 0.0,
            "kenlm": kenlm_prior.path.name if kenlm_prior is not None else "disabled",
            "kenlm_order": kenlm_prior.order if kenlm_prior is not None else None,
            "kenlm_minimum_ngram_order": kenlm_prior.minimum_ngram_order if kenlm_prior is not None else None,
            "kenlm_weight": KENLM_WEIGHT if kenlm_prior is not None else 0.0,
            "kenlm_mode": KENLM_MODE, "kenlm_word_bonus": KENLM_WORD_BONUS,
            "word_lookahead_seconds": WORD_LOOKAHEAD_FRAMES * ENCODER_FRAME_SECONDS}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        result = await transcribe_pcm(read_wav_bytes(await file.read()))
        return result
    except Exception as exc:
        return {"error": str(exc)}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    recognizer = new_recognizer()
    await websocket.send_json({"type": "ready", "sample_rate": SAMPLE_RATE,
                               "format": "float32-le-mono", "step_seconds": BLOCK_SECONDS if RECOGNITION_MODE == "blocks" else PARTIAL_INTERVAL_SECONDS,
                               "recognition_mode": RECOGNITION_MODE,
                               "block_seconds": BLOCK_SECONDS,
                               "normalization_lookahead_seconds": 4.0 if RECOGNITION_MODE == "blocks" else None,
                               "text_commit_seconds": 4.0, "cache_commit_seconds": 4.0,
                               "word_commit_policy": "shared-word-boundary" if RECOGNITION_MODE == "blocks" else "adaptive-boundary-lookahead",
                               "max_pending_window_seconds": MAX_PENDING_SAMPLES / SAMPLE_RATE,
                               "word_lookahead_seconds": WORD_LOOKAHEAD_FRAMES * ENCODER_FRAME_SECONDS,
                               "encoder_context": RUNTIME["att_context_size"]})
    buffered = np.empty(0, dtype=np.float32)
    last_frontier = -1
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            raw = message.get("bytes")
            if raw is None:
                try:
                    command = json.loads(message.get("text") or "null")
                except json.JSONDecodeError:
                    command = None
                if isinstance(command, dict) and command.get("type") == "finish":
                    async with INFERENCE_LOCK:
                        result = await asyncio.to_thread(recognizer.finish, buffered)
                    await websocket.send_json(result)
                    await websocket.close()
                    return
                continue
            if len(raw) % 4:
                await websocket.send_json({"type": "error", "message": "PCM byte count must be divisible by 4."})
                continue
            buffered = np.concatenate((buffered, np.frombuffer(raw, dtype="<f4").copy()))
            while buffered.size >= PARTIAL_INTERVAL_SAMPLES:
                chunk, buffered = buffered[:PARTIAL_INTERVAL_SAMPLES], buffered[PARTIAL_INTERVAL_SAMPLES:]
                async with INFERENCE_LOCK:
                    state = await asyncio.to_thread(recognizer.update, chunk)
                frontier = state.get("window_end")
                # Unchanged collection ticks must not count as fresh matching
                # evidence; send only actual acoustic progress in block mode.
                if RECOGNITION_MODE != "blocks" or frontier != last_frontier:
                    await websocket.send_json(state)
                    last_frontier = frontier
    except WebSocketDisconnect:
        pass
