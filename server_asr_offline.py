from __future__ import annotations

import asyncio
import copy
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

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"

RUNTIME_PATH = MODEL_DIR / "runtime.json"


with open(
    RUNTIME_PATH,
    "r",
    encoding="utf-8",
) as f:
    RUNTIME = json.load(f)


# ============================================================
# ASR constants
#
# These frontend values were already validated against NeMo.
# ============================================================

SAMPLE_RATE = 16000

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80

PREEMPH = 0.97

LOG_ZERO_GUARD = np.float32(
    5.960464477539063e-08
)

LOCAL_VOCAB_SIZE = 256

JOINT_BLANK_ID = 256

PREDICTOR_START_ID = 5632

MAX_SYMBOLS_PER_STEP = int(
    RUNTIME.get(
        "max_symbols_per_step",
        5,
    )
)


# ============================================================
# Streaming configuration
# ============================================================

# Decode a bounded recent window for live updates. Keep the complete recording
# separately in audio_chunks for the authoritative /transcribe-style flush.
LIVE_STEP_SECONDS = 0.75

# Long enough to cover slow sung phrases while keeping per-update work bounded.
LIVE_WINDOW_SECONDS = 6.0
LEFT_CONTEXT_SECONDS = 4.0
# A sustained sung word can outlast the normal rolling window. Preserve that
# trailing hypothesis for up to four additional seconds while waiting to hear
# the next word's onset.
LIVE_WINDOW_EXTENSION_SECONDS = 4.0
# Keep up to five seconds before a still-uncommitted trailing word, capped by
# the ten-second live audio snapshot.
TRAILING_WORD_PREROLL_SECONDS = 5.0
MAX_LIVE_WINDOW_SECONDS = LIVE_WINDOW_SECONDS + LIVE_WINDOW_EXTENSION_SECONDS

# Kirtan sustains words across long notes. Keep a longer acoustic tail before
# committing words so a partly sung word is not frozen too early.
RIGHT_EDGE_GUARD_SECONDS = 1.25
NEXT_WORD_ONSET_GUARD_SECONDS = 0.15

# Confirm once across successive decodes. A one-second-plus right-edge guard
# still prevents words at the newest audio edge from being committed early.
STABLE_REPEATS = 2

# Boundary for removing old committed left-context words.
TIMESTAMP_TOLERANCE_SECONDS = 0.20

# Singing can move word timestamps between successive rolling windows while
# the decoded word remains stable. Keep this separate from the commit boundary.
STABILITY_MATCH_TOLERANCE_SECONDS = 0.75


LIVE_STEP_SAMPLES = int(
    SAMPLE_RATE
    *
    LIVE_STEP_SECONDS
)

LIVE_WINDOW_SAMPLES = int(
    SAMPLE_RATE
    *
    LIVE_WINDOW_SECONDS
)
MAX_LIVE_WINDOW_SAMPLES = int(
    SAMPLE_RATE
    * MAX_LIVE_WINDOW_SECONDS
)
LEFT_CONTEXT_SAMPLES = int(
    SAMPLE_RATE
    *
    LEFT_CONTEXT_SECONDS
)


# ============================================================
# Encoder timing
#
# Frontend frames are 10 ms:
#
#   hop 160 / 16000 = 0.01 sec
#
# IndicConformer encoder output observed:
#
#   298 frontend frames -> 75 encoder frames
#
# ~= 4x subsampling.
#
# Therefore one encoder frame is approximately 40 ms.
# ============================================================

ENCODER_SUBSAMPLING = 4

ENCODER_FRAME_SECONDS = (
    HOP_LENGTH
    *
    ENCODER_SUBSAMPLING
    /
    SAMPLE_RATE
)


# ============================================================
# CPU ONLY
# ============================================================

PROVIDERS = [
    "CPUExecutionProvider",
]


# ============================================================
# Model files
# ============================================================

ENCODER_PATH = (
    MODEL_DIR
    /
    RUNTIME["encoder"]
)

DECODER_JOINT_PATH = (
    MODEL_DIR
    /
    RUNTIME["decoder_joint"]
)

TOKENIZER_PATH = (
    MODEL_DIR
    /
    RUNTIME["tokenizer"]
)

MEL_PATH = (
    MODEL_DIR
    /
    RUNTIME["mel_filterbank"]
)

WINDOW_PATH = (
    MODEL_DIR
    /
    RUNTIME["stft_window"]
)


# ============================================================
# Frontend assets
# ============================================================

mel_filterbank = np.load(
    MEL_PATH
).astype(
    np.float32
)


if (
    mel_filterbank.ndim == 3
    and
    mel_filterbank.shape[0] == 1
):
    mel_filterbank = (
        mel_filterbank[0]
    )


expected_mel_shape = (
    N_MELS,
    N_FFT // 2 + 1,
)


if (
    mel_filterbank.shape
    !=
    expected_mel_shape
):
    raise RuntimeError(
        "Unexpected mel filterbank shape: "
        f"{mel_filterbank.shape}. "
        f"Expected {expected_mel_shape}."
    )


stft_window = np.load(
    WINDOW_PATH
).astype(
    np.float32
).reshape(-1)


if (
    stft_window.shape[0]
    !=
    WIN_LENGTH
):
    raise RuntimeError(
        "Unexpected STFT window length: "
        f"{stft_window.shape[0]}"
    )


# ============================================================
# Tokenizer
# ============================================================

tokenizer = (
    spm.SentencePieceProcessor()
)


if not tokenizer.load(
    str(TOKENIZER_PATH)
):
    raise RuntimeError(
        f"Could not load tokenizer: "
        f"{TOKENIZER_PATH}"
    )


if (
    tokenizer.get_piece_size()
    !=
    LOCAL_VOCAB_SIZE
):
    raise RuntimeError(
        "Unexpected Punjabi tokenizer vocabulary: "
        f"{tokenizer.get_piece_size()}"
    )


# ============================================================
# ONNX Runtime
# ============================================================

session_options = (
    ort.SessionOptions()
)

session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel
    .ORT_ENABLE_ALL
)

# CPU only.
#
# 0 lets ORT choose an appropriate intra-op CPU thread count.
session_options.intra_op_num_threads = 0
session_options.inter_op_num_threads = 1

session_options.execution_mode = (
    ort.ExecutionMode
    .ORT_SEQUENTIAL
)


encoder_session = (
    ort.InferenceSession(
        str(ENCODER_PATH),
        sess_options=session_options,
        providers=PROVIDERS,
    )
)


decoder_session = (
    ort.InferenceSession(
        str(DECODER_JOINT_PATH),
        sess_options=session_options,
        providers=PROVIDERS,
    )
)


encoder_input_names = [
    item.name
    for item
    in encoder_session.get_inputs()
]


decoder_input_names = [
    item.name
    for item
    in decoder_session.get_inputs()
]


# ============================================================
# Serialize heavy CPU inference
# ============================================================

INFERENCE_LOCK = asyncio.Lock()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Gurbani Live ASR",
)


if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(
                STATIC_DIR
            )
        ),
        name="static",
    )


# ============================================================
# Feature normalization
# ============================================================

def normalize_features(
    features: np.ndarray,
) -> np.ndarray:

    features = np.asarray(
        features,
        dtype=np.float32,
    )


    mean = np.mean(
        features,
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )


    if (
        features.shape[1] > 1
    ):

        std = np.std(
            features,
            axis=1,
            keepdims=True,
            ddof=1,
            dtype=np.float32,
        )

    else:

        std = np.zeros(
            (
                features.shape[0],
                1,
            ),
            dtype=np.float32,
        )


    return (
        (
            features
            -
            mean
        )
        /
        (
            std
            +
            np.float32(1e-5)
        )
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# NumPy frontend
# ============================================================

def preprocess_audio(
    audio: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)


    if audio.size == 0:

        raise ValueError(
            "Audio is empty."
        )


    # --------------------------------------------------------
    # Pre-emphasis
    # --------------------------------------------------------

    emphasized = np.empty_like(
        audio
    )

    emphasized[0] = audio[0]


    if audio.size > 1:

        emphasized[1:] = (
            audio[1:]
            -
            np.float32(
                PREEMPH
            )
            *
            audio[:-1]
        )


    # --------------------------------------------------------
    # torch.stft(center=True) equivalent
    # --------------------------------------------------------

    pad = N_FFT // 2


    if (
        emphasized.size > pad
    ):

        padded = np.pad(
            emphasized,
            (
                pad,
                pad,
            ),
            mode="reflect",
        )

    else:

        padded = np.pad(
            emphasized,
            (
                pad,
                pad,
            ),
            mode="constant",
        )


    if (
        padded.size < N_FFT
    ):

        padded = np.pad(
            padded,
            (
                0,
                N_FFT
                -
                padded.size,
            ),
            mode="constant",
        )


    num_frames = (
        (
            padded.size
            -
            N_FFT
        )
        //
        HOP_LENGTH
        +
        1
    )


    # --------------------------------------------------------
    # Framing
    # --------------------------------------------------------

    windows = (
        np.lib.stride_tricks
        .sliding_window_view(
            padded,
            N_FFT,
        )
    )


    frames = windows[
        ::HOP_LENGTH
    ][
        :num_frames
    ]


    frames = np.asarray(
        frames,
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Exported 400 sample window centered in 512
    # --------------------------------------------------------

    fft_window = np.zeros(
        N_FFT,
        dtype=np.float32,
    )


    window_start = (
        N_FFT
        -
        WIN_LENGTH
    ) // 2


    fft_window[
        window_start:
        window_start + WIN_LENGTH
    ] = stft_window


    frames = (
        frames
        *
        fft_window[None, :]
    )


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    spectrum = np.fft.rfft(
        frames,
        n=N_FFT,
        axis=-1,
    )


    power = (
        np.abs(
            spectrum
        )
        ** 2
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Mel
    # --------------------------------------------------------

    mel = np.matmul(
        power,
        mel_filterbank.T,
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    mel = np.log(
        mel
        +
        LOG_ZERO_GUARD
    ).astype(
        np.float32
    )


    # [80, time]
    features = mel.T


    features = normalize_features(
        features
    )


    # [1, 80, time]
    features = features[
        None,
        :,
        :
    ].astype(
        np.float32,
        copy=False,
    )


    feature_length = np.asarray(
        [
            features.shape[2]
        ],
        dtype=np.int64,
    )


    return (
        features,
        feature_length,
    )


# ============================================================
# Timed RNNT token
# ============================================================

@dataclass
class TimedToken:

    token_id: int

    encoder_frame: int


# ============================================================
# RNNT decode WITH encoder-frame timestamps
# ============================================================

def rnnt_decode_greedy_timed(
    encoder_outputs: np.ndarray,
    encoded_length: int,
) -> list[TimedToken]:

    encoder_outputs = np.asarray(
        encoder_outputs,
        dtype=np.float32,
    )


    hidden_size = 640


    current_h = np.zeros(
        (
            1,
            1,
            hidden_size,
        ),
        dtype=np.float32,
    )


    current_c = np.zeros(
        (
            1,
            1,
            hidden_size,
        ),
        dtype=np.float32,
    )


    current_label = (
        PREDICTOR_START_ID
    )


    target_length = np.asarray(
        [1],
        dtype=np.int32,
    )


    output: list[
        TimedToken
    ] = []


    for t in range(
        int(
            encoded_length
        )
    ):

        symbols_this_frame = 0


        while (
            symbols_this_frame
            <
            MAX_SYMBOLS_PER_STEP
        ):

            encoder_frame = (
                encoder_outputs[
                    :,
                    :,
                    t:t + 1,
                ]
            )


            target = np.asarray(
                [
                    [
                        current_label
                    ]
                ],
                dtype=np.int32,
            )


            result = (
                decoder_session.run(
                    None,
                    {
                        decoder_input_names[0]:
                            encoder_frame,

                        decoder_input_names[1]:
                            target,

                        decoder_input_names[2]:
                            target_length,

                        decoder_input_names[3]:
                            current_h,

                        decoder_input_names[4]:
                            current_c,
                    },
                )
            )


            logits = result[0]

            candidate_h = result[2]

            candidate_c = result[3]


            local_token = int(
                np.argmax(
                    logits.reshape(
                        -1,
                        logits.shape[-1],
                    )[0]
                )
            )


            # Blank -> advance encoder frame.
            #
            # Candidate predictor state is NOT committed.
            if (
                local_token
                ==
                JOINT_BLANK_ID
            ):

                break


            if not (
                0
                <=
                local_token
                <
                LOCAL_VOCAB_SIZE
            ):

                raise RuntimeError(
                    "RNNT emitted invalid local token: "
                    f"{local_token}"
                )


            output.append(
                TimedToken(
                    token_id=local_token,
                    encoder_frame=t,
                )
            )


            # Commit predictor state on nonblank.
            current_h = candidate_h
            current_c = candidate_c


            # CRITICAL:
            # predictor feedback is Punjabi LOCAL ID.
            current_label = local_token


            symbols_this_frame += 1


    return output


def rnnt_decode_timed(
    encoder_outputs: np.ndarray,
    encoded_length: int,
) -> list[TimedToken]:
    return rnnt_decode_greedy_timed(encoder_outputs, encoded_length)


# ============================================================
# Timed word
# ============================================================

@dataclass
class TimedWord:

    text: str

    start: float

    end: float

    token_ids: list[int]

    stable_count: int = 1


    def to_json(
        self,
    ) -> dict[str, Any]:

        return {
            "text":
                self.text,

            "start":
                round(
                    self.start,
                    3,
                ),

            "end":
                round(
                    self.end,
                    3,
                ),

            "stable_count":
                self.stable_count,
        }


# ============================================================
# Convert SentencePiece timed tokens -> words
#
# SentencePiece normally marks a new word using ▁
# ============================================================

def timed_tokens_to_words(
    timed_tokens: list[TimedToken],
    window_start_seconds: float,
) -> list[TimedWord]:

    if not timed_tokens:
        return []


    groups: list[
        dict[str, Any]
    ] = []


    current_ids: list[int] = []
    current_frames: list[int] = []


    for token in timed_tokens:

        piece = tokenizer.id_to_piece(
            token.token_id
        )


        starts_new_word = (
            piece.startswith("▁")
        )


        if (
            starts_new_word
            and
            current_ids
        ):

            groups.append({
                "ids":
                    current_ids,

                "frames":
                    current_frames,
            })


            current_ids = []
            current_frames = []


        current_ids.append(
            token.token_id
        )

        current_frames.append(
            token.encoder_frame
        )


    if current_ids:

        groups.append({
            "ids":
                current_ids,

            "frames":
                current_frames,
        })


    words: list[
        TimedWord
    ] = []


    for index, group in enumerate(
        groups
    ):

        ids = group["ids"]
        frames = group["frames"]


        word_text = (
            tokenizer.decode(
                ids
            )
            .strip()
        )


        if not word_text:
            continue


        first_frame = (
            frames[0]
        )


        start_seconds = (
            window_start_seconds
            +
            first_frame
            *
            ENCODER_FRAME_SECONDS
        )


        # Prefer next word's first token as this word's end.
        if (
            index + 1
            <
            len(groups)
        ):

            next_first_frame = (
                groups[
                    index + 1
                ]["frames"][0]
            )


            end_seconds = (
                window_start_seconds
                +
                next_first_frame
                *
                ENCODER_FRAME_SECONDS
            )

        else:

            last_frame = (
                frames[-1]
            )


            end_seconds = (
                window_start_seconds
                +
                (
                    last_frame
                    + 1
                )
                *
                ENCODER_FRAME_SECONDS
            )


        if (
            end_seconds
            <=
            start_seconds
        ):

            end_seconds = (
                start_seconds
                +
                ENCODER_FRAME_SECONDS
            )


        words.append(
            TimedWord(
                text=word_text,
                start=start_seconds,
                end=end_seconds,
                token_ids=list(
                    ids
                ),
            )
        )


    return words


# ============================================================
# Core transcription
# ============================================================

def transcribe_pcm_sync(
    audio: np.ndarray,
    window_start_seconds: float = 0.0,
) -> dict[str, Any]:

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)


    (
        features,
        feature_length,
    ) = preprocess_audio(
        audio
    )


    encoder_result = (
        encoder_session.run(
            None,
            {
                encoder_input_names[0]:
                    features,

                encoder_input_names[1]:
                    feature_length,
            },
        )
    )


    encoder_outputs = np.asarray(
        encoder_result[0],
        dtype=np.float32,
    )


    encoded_length = int(
        np.asarray(
            encoder_result[1]
        ).reshape(-1)[0]
    )


    timed_tokens = (
        rnnt_decode_timed(
            encoder_outputs,
            encoded_length,
        )
    )


    token_ids = [
        item.token_id
        for item
        in timed_tokens
    ]


    text = (
        tokenizer.decode(
            token_ids
        )
        if token_ids
        else ""
    )


    words = (
        timed_tokens_to_words(
            timed_tokens,
            window_start_seconds,
        )
    )


    return {
        "text":
            text,

        "tokens":
            token_ids,

        "words":
            words,

        "samples":
            int(
                audio.size
            ),

        "seconds":
            float(
                audio.size
                /
                SAMPLE_RATE
            ),

        "window_start":
            float(
                window_start_seconds
            ),

        "window_end":
            float(
                window_start_seconds
                +
                audio.size
                /
                SAMPLE_RATE
            ),

        "feature_frames":
            int(
                features.shape[2]
            ),

        "encoded_frames":
            encoded_length,
    }


async def transcribe_pcm(
    audio: np.ndarray,
    window_start_seconds: float = 0.0,
) -> dict[str, Any]:

    async with INFERENCE_LOCK:

        return await asyncio.to_thread(
            transcribe_pcm_sync,
            audio,
            window_start_seconds,
        )


# ============================================================
# Transcript stabilizer
# ============================================================

class TimestampTranscript:

    def __init__(
        self,
        history_limit: int | None = None,
    ):

        self.history_limit = history_limit

        self.committed_words: list[
            TimedWord
        ] = []

        self.previous_words: list[
            TimedWord
        ] = []

        self.current_words: list[
            TimedWord
        ] = []

        self.committed_until: float = 0.0


    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.committed_words = []
        self.previous_words = []
        self.current_words = []

        self.committed_until = 0.0


    # ========================================================
    # Same word across sliding hypotheses
    # ========================================================

    @staticmethod
    def same_word(
        old: TimedWord,
        new: TimedWord,
    ) -> bool:

        if (
            old.text
            !=
            new.text
        ):
            return False


        start_difference = abs(
            old.start
            -
            new.start
        )


        end_difference = abs(
            old.end
            -
            new.end
        )


        return (
            start_difference
            <=
            STABILITY_MATCH_TOLERANCE_SECONDS
            or
            end_difference
            <=
            STABILITY_MATCH_TOLERANCE_SECONDS
        )


    # ========================================================
    # Determine whether decoded word belongs to old committed
    # context.
    #
    # This is the important duplicate protection.
    # ========================================================

    def is_already_committed(
        self,
        word: TimedWord,
    ) -> bool:

        if (
            not self.committed_words
        ):

            return False


        # ----------------------------------------------------
        # Anything ending clearly before the committed boundary
        # is old context.
        # ----------------------------------------------------

        if (
            word.end
            <=
            self.committed_until
            +
            TIMESTAMP_TOLERANCE_SECONDS
        ):

            return True


        # ----------------------------------------------------
        # Also explicitly compare against recently committed
        # words.
        #
        # We only need to inspect the newest few words.
        # ----------------------------------------------------

        recent_committed = (
            self.committed_words[
                -8:
            ]
        )


        for committed in recent_committed:

            if (
                committed.text
                !=
                word.text
            ):

                continue


            start_difference = abs(
                committed.start
                -
                word.start
            )


            end_difference = abs(
                committed.end
                -
                word.end
            )


            if (
                start_difference
                <=
                0.35
                or
                end_difference
                <=
                0.35
            ):

                return True


        return False


    # ========================================================
    # Update stability counts
    # ========================================================

    def update_stability(
        self,
        words: list[TimedWord],
    ) -> list[TimedWord]:

        used_previous: set[int] = set()


        for word in words:

            # Don't stability-track words from committed
            # left context.
            if (
                self.is_already_committed(
                    word
                )
            ):

                word.stable_count = 0

                continue


            best_index = None
            best_distance = None


            for index, old in enumerate(
                self.previous_words
            ):

                if (
                    index
                    in
                    used_previous
                ):

                    continue


                if (
                    self.is_already_committed(
                        old
                    )
                ):

                    continue


                if not self.same_word(
                    old,
                    word,
                ):

                    continue


                distance = abs(
                    old.start
                    -
                    word.start
                )


                if (
                    best_distance is None
                    or
                    distance
                    <
                    best_distance
                ):

                    best_distance = distance

                    best_index = index


            if (
                best_index
                is not None
            ):

                old = (
                    self.previous_words[
                        best_index
                    ]
                )


                word.stable_count = (
                    old.stable_count
                    +
                    1
                )


                used_previous.add(
                    best_index
                )

            else:

                word.stable_count = 1


        return words


    # ========================================================
    # Only return genuinely new/uncommitted words
    # ========================================================

    def uncommitted_words(
        self,
        words: list[TimedWord],
    ) -> list[TimedWord]:

        result = []


        for word in words:

            if (
                self.is_already_committed(
                    word
                )
            ):

                continue


            result.append(
                word
            )


        return result


    # ========================================================
    # Commit stable prefix
    # ========================================================

    def commit_stable_words(
        self,
        words: list[TimedWord],
        window_end: float,
    ) -> None:

        uncommitted = (
            self.uncommitted_words(
                words
            )
        )


        safe_before = (
            window_end
            -
            RIGHT_EDGE_GUARD_SECONDS
        )


        for word_index, word in enumerate(uncommitted):

            # -----------------------------------------------
            # Needs repeated confirmation.
            # -----------------------------------------------

            if (
                word.stable_count
                <
                STABLE_REPEATS
            ):

                break


            # -----------------------------------------------
            # Do not commit newest edge of current window.
            # -----------------------------------------------

            if (
                word.end
                >
                safe_before
            ):

                break


            # A sustained sung word can look complete in one rolling decode
            # and then change when more of the note arrives. Keep the
            # trailing word revisable until a following word has started in
            # the captured audio. Its snapshot can grow by up to two seconds.
            if word_index + 1 >= len(uncommitted):
                break

            following = uncommitted[word_index + 1]
            if (
                following.start < word.start + NEXT_WORD_ONSET_GUARD_SECONDS
                or following.start > window_end - NEXT_WORD_ONSET_GUARD_SECONDS
            ):
                break


            # -----------------------------------------------
            # Final duplicate guard before adding.
            # -----------------------------------------------

            if (
                self.is_already_committed(
                    word
                )
            ):

                continue


            self.committed_words.append(
                TimedWord(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    token_ids=list(
                        word.token_ids
                    ),
                    stable_count=word.stable_count,
                )
            )


            self.committed_until = max(
                self.committed_until,
                word.end,
            )


    # ========================================================
    # Process newest sliding hypothesis
    # ========================================================

    def update(
        self,
        words: list[TimedWord],
        window_end: float,
        window_start: float | None = None,
    ) -> dict[str, Any]:

        # An unresolved word must not be discarded when the bounded audio
        # window slides past it. Promote only words wholly before the new
        # window edge; words that still overlap the window remain revisable.
        if window_start is None:
            window_start = max(0.0, window_end - LIVE_WINDOW_SECONDS)
        expired: dict[tuple[str, float, float], TimedWord] = {}
        retained: dict[tuple[str, float, float], TimedWord] = {}
        for word in self.previous_words + self.current_words:
            key = (word.text, round(word.start, 3), round(word.end, 3))
            target = expired if word.end <= window_start else retained
            old = target.get(key)
            if old is None or word.stable_count > old.stable_count:
                target[key] = word
        self.current_words = sorted(
            retained.values(), key=lambda word: (word.start, word.end)
        )
        self.previous_words = [
            TimedWord(
                text=word.text,
                start=word.start,
                end=word.end,
                token_ids=list(word.token_ids),
                stable_count=word.stable_count,
            )
            for word in self.current_words
        ]
        pending_tail = max(
            retained.values(),
            key=lambda word: (word.start, word.end),
            default=None,
        )
        for word in sorted(expired.values(), key=lambda item: (item.start, item.end)):
            if self.is_already_committed(word):
                continue
            self.committed_words.append(
                TimedWord(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    token_ids=list(word.token_ids),
                    stable_count=word.stable_count,
                )
            )
            self.committed_until = max(self.committed_until, word.end)

        # A word only enters committed_words after repeated confirmation, a
        # safe distance from the current audio edge, and a following-word
        # onset. Once it meets those checks it is final and must not be
        # reopened by a later rolling decode.
        previous_hypothesis = list(self.previous_words)
        if self.committed_words:
            self.committed_until = max(
                self.committed_until,
                self.committed_words[-1].end,
            )

        # The hypothesis list is timestamp ordered. Keep one copy if the
        # revisable committed word was already present in the previous tail.
        previous_hypothesis.sort(key=lambda word: (word.start, word.end))
        deduped_previous: list[TimedWord] = []
        for word in previous_hypothesis:
            if (
                deduped_previous
                and deduped_previous[-1].text == word.text
                and abs(deduped_previous[-1].start - word.start) <= 0.001
                and abs(deduped_previous[-1].end - word.end) <= 0.001
            ):
                if word.stable_count > deduped_previous[-1].stable_count:
                    deduped_previous[-1] = word
                continue
            deduped_previous.append(word)
        self.previous_words = deduped_previous

        # -----------------------------------------------
        # Immediately remove words belonging to left context
        # before doing stability matching.
        # -----------------------------------------------

        words = [
            word
            for word
            in words
            if not self.is_already_committed(
                word
            )
        ]


        words = self.update_stability(
            words
        )

        # A rolling decode can briefly withdraw its trailing partial while it
        # re-evaluates a sustained note. Keep that one hypothesis in the
        # revisable tail when no new word occupies the same time span. If the
        # next onset is already present, this lets the normal commit checks
        # finalize it; otherwise it remains partial and keeps its source audio
        # in the adaptive window.
        if pending_tail is not None and pending_tail.end > window_start:
            replacement_present = any(
                word.start >= pending_tail.start - 0.35
                and word.start <= pending_tail.end + 0.35
                and word.end > pending_tail.start + 0.05
                for word in words
            )
            if not replacement_present:
                carried = TimedWord(
                    text=pending_tail.text,
                    start=pending_tail.start,
                    end=pending_tail.end,
                    token_ids=list(pending_tail.token_ids),
                    stable_count=min(
                        STABLE_REPEATS,
                        pending_tail.stable_count + 1,
                    ),
                )
                words.append(carried)
                words.sort(key=lambda word: (word.start, word.end))

        self.commit_stable_words(
            words,
            window_end,
        )


        self.current_words = (
            self.uncommitted_words(
                words
            )
        )

        # Only retain genuinely uncommitted previous words.
        self.previous_words = [

            TimedWord(
                text=word.text,
                start=word.start,
                end=word.end,
                token_ids=list(
                    word.token_ids
                ),
                stable_count=word.stable_count,
            )

            for word
            in self.current_words
        ]

        self._trim_committed_history()


        return self.result()


    def _trim_committed_history(self) -> None:
        if self.history_limit and len(self.committed_words) > self.history_limit:
            self.committed_words = self.committed_words[-self.history_limit:]


    # ========================================================
    # Current UI state
    # ========================================================

    def result(
        self,
    ) -> dict[str, Any]:

        committed_text = " ".join(
            word.text
            for word
            in self.committed_words
        ).strip()


        partial_text = " ".join(
            word.text
            for word
            in self.current_words
        ).strip()


        combined_text = " ".join(
            part
            for part
            in (
                committed_text,
                partial_text,
            )
            if part
        ).strip()


        return {
            "committed_text":
                committed_text,

            "partial_text":
                partial_text,

            "text":
                combined_text,

            "committed_until":
                self.committed_until,

            "committed_words":
                [
                    word.to_json()
                    for word
                    in self.committed_words
                ],

            "partial_words":
                [
                    word.to_json()
                    for word
                    in self.current_words
                ],
        }


    # ========================================================
    # Stop
    #
    # Promote only genuinely NEW words.
    # ========================================================

    def finalize_current(
        self,
    ) -> dict[str, Any]:

        for word in self.current_words:

            if (
                self.is_already_committed(
                    word
                )
            ):

                continue


            self.committed_words.append(
                TimedWord(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    token_ids=list(
                        word.token_ids
                    ),
                    stable_count=word.stable_count,
                )
            )


            self.committed_until = max(
                self.committed_until,
                word.end,
            )


        self.current_words = []

        self.previous_words = []

        self._trim_committed_history()


        return self.result()


# ============================================================
# WAV reader
# ============================================================

def read_wav_bytes(
    wav_bytes: bytes,
) -> np.ndarray:

    with wave.open(
        io.BytesIO(
            wav_bytes
        ),
        "rb",
    ) as wav_file:

        channels = (
            wav_file.getnchannels()
        )

        sample_width = (
            wav_file.getsampwidth()
        )

        sample_rate = (
            wav_file.getframerate()
        )

        frame_count = (
            wav_file.getnframes()
        )


        if (
            sample_rate
            !=
            SAMPLE_RATE
        ):

            raise ValueError(
                "Expected WAV sample rate "
                f"{SAMPLE_RATE}, got "
                f"{sample_rate}."
            )


        raw = wav_file.readframes(
            frame_count
        )


    if (
        sample_width == 2
    ):

        audio = np.frombuffer(
            raw,
            dtype="<i2",
        ).astype(
            np.float32
        )

        audio /= np.float32(
            32768.0
        )


    elif (
        sample_width == 4
    ):

        audio = np.frombuffer(
            raw,
            dtype="<i4",
        ).astype(
            np.float32
        )

        audio /= np.float32(
            2147483648.0
        )


    else:

        raise ValueError(
            "Only 16-bit and 32-bit integer "
            "PCM WAV files are supported."
        )


    if (
        channels > 1
    ):

        audio = audio.reshape(
            -1,
            channels,
        )


        audio = np.mean(
            audio,
            axis=1,
            dtype=np.float32,
        )


    return audio.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Root
# ============================================================

@app.get("/")
async def index():

    index_path = (
        STATIC_DIR
        /
        "index.html"
    )


    if (
        index_path.exists()
    ):

        return FileResponse(
            index_path
        )


    return {
        "message":
            "Gurbani Live ASR"
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health():

    return {
        "status":
            "ok",

        "provider":
            "CPUExecutionProvider",

        "sample_rate":
            SAMPLE_RATE,

        "live_step_seconds":
            LIVE_STEP_SECONDS,

        "live_window_seconds":
            LIVE_WINDOW_SECONDS,

        "max_live_window_seconds":
            MAX_LIVE_WINDOW_SECONDS,

        "window_mode":
            "bounded_live_full_final",

        "left_context_seconds":
            LEFT_CONTEXT_SECONDS,

        "stable_repeats":
            STABLE_REPEATS,

        "timestamp_tolerance_seconds":
            TIMESTAMP_TOLERANCE_SECONDS,

        "stability_match_tolerance_seconds":
            STABILITY_MATCH_TOLERANCE_SECONDS,

        "right_edge_guard_seconds":
            RIGHT_EDGE_GUARD_SECONDS,

        "encoder_frame_seconds":
            ENCODER_FRAME_SECONDS,

        "joint_blank_id":
            JOINT_BLANK_ID,

        "predictor_start_id":
            PREDICTOR_START_ID,

        "predictor_feedback":
            "local",
    }


# ============================================================
# HTTP WAV transcription
# ============================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
):

    try:

        wav_bytes = (
            await file.read()
        )


        audio = read_wav_bytes(
            wav_bytes
        )


        result = await transcribe_pcm(
            audio,
            0.0,
        )


        return {
            "text":
                result["text"],

            "tokens":
                result["tokens"],

            "words":
                [
                    word.to_json()
                    for word
                    in result["words"]
                ],

            "samples":
                result["samples"],

            "seconds":
                result["seconds"],

            "feature_frames":
                result["feature_frames"],

            "encoded_frames":
                result["encoded_frames"],
        }


    except Exception as exc:

        return {
            "error":
                str(exc)
        }


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
):

    await websocket.accept()

    # The always-on site stream only needs a recent rolling window. Keep the
    # historical full-capture behavior for ordinary clients that may request
    # an authoritative flush at the end of a recording.
    live_mode = websocket.query_params.get("mode") == "live"


    await websocket.send_json({
        "type":
            "ready",

        "sample_rate":
            SAMPLE_RATE,

        "format":
            "float32-le-mono",

        "step_seconds":
            LIVE_STEP_SECONDS,

        "window_seconds":
            LIVE_WINDOW_SECONDS,

        "max_window_seconds":
            MAX_LIVE_WINDOW_SECONDS,

        "window_mode":
            "bounded_live" if live_mode else "bounded_live_full_final",

        "left_context_seconds":
            LEFT_CONTEXT_SECONDS,

        "stability_repeats":
            STABLE_REPEATS,

        "timestamp_tolerance_seconds":
            TIMESTAMP_TOLERANCE_SECONDS,

        "stability_match_tolerance_seconds":
            STABILITY_MATCH_TOLERANCE_SECONDS,

        "message":
            (
                "Send raw 16 kHz mono "
                "Float32 PCM bytes"
            ),
    })


    # ========================================================
    # Audio buffer
    #
    # buffer_start_sample is the absolute stream sample index corresponding to
    # the first retained chunk.
    #
    # total_samples_received is absolute stream position.
    # ========================================================

    # Store incoming frames as chunks. Repeated np.concatenate on a growing
    # recording would copy all earlier samples for every browser packet.
    audio_chunks: list[tuple[int, np.ndarray]] = []


    buffer_start_sample = 0

    total_samples_received = 0

    samples_since_inference = 0


    transcript = TimestampTranscript(history_limit=120 if live_mode else None)


    inference_running = False

    inference_pending = False

    connection_alive = True

    generation = 0


    # ========================================================
    # Ordinary websocket clients retain a complete capture for final decode.
    # The always-on live endpoint prunes to a short rolling buffer instead.
    # ========================================================

    def retain_audio_chunk(
        chunk: np.ndarray,
        chunk_start_sample: int,
    ):
        audio_chunks.append((chunk_start_sample, chunk))


    def trim_audio_buffer():
        nonlocal buffer_start_sample
        if not live_mode:
            return
        keep_samples = (
            MAX_LIVE_WINDOW_SAMPLES
            + LEFT_CONTEXT_SAMPLES
            + LIVE_STEP_SAMPLES
        )
        cutoff_sample = total_samples_received - keep_samples
        while audio_chunks:
            first_start, first_chunk = audio_chunks[0]
            if first_start + first_chunk.size > cutoff_sample:
                break
            audio_chunks.pop(0)
        buffer_start_sample = (
            audio_chunks[0][0] if audio_chunks else total_samples_received
        )


    # ========================================================
    # Build the newest bounded snapshot for live inference.
    # ========================================================

    def make_live_snapshot() -> tuple[np.ndarray, int]:
        if not audio_chunks:
            return np.empty(0, dtype=np.float32), buffer_start_sample

        stream_end_sample = total_samples_received
        rolling_start = max(
            buffer_start_sample,
            stream_end_sample - LIVE_WINDOW_SAMPLES,
        )
        max_extension_start = max(
            buffer_start_sample,
            stream_end_sample - MAX_LIVE_WINDOW_SAMPLES,
        )
        trailing_word_start: int | None = None
        if transcript.current_words:
            trailing_word_start = max(
                0,
                int(transcript.current_words[-1].start * SAMPLE_RATE),
            )
            trailing_context_start = trailing_word_start - int(
                TRAILING_WORD_PREROLL_SECONDS * SAMPLE_RATE
            )
            if trailing_context_start < rolling_start:
                rolling_start = max(
                    max_extension_start,
                    trailing_context_start,
                )
        context_start = int(
            max(
                0.0,
                transcript.committed_until - LEFT_CONTEXT_SECONDS,
            )
            * SAMPLE_RATE
        )
        start_sample = min(
            stream_end_sample,
            max(rolling_start, context_start),
        )
        if trailing_word_start is not None:
            trailing_context_start = trailing_word_start - int(
                TRAILING_WORD_PREROLL_SECONDS * SAMPLE_RATE
            )
            if trailing_context_start < start_sample:
                start_sample = max(
                    max_extension_start,
                    trailing_context_start,
                )
        pieces = []
        for chunk_start, chunk in audio_chunks:
            chunk_end = chunk_start + chunk.size
            overlap_start = max(start_sample, chunk_start)
            overlap_end = min(stream_end_sample, chunk_end)
            if overlap_start < overlap_end:
                pieces.append(
                    chunk[overlap_start - chunk_start:overlap_end - chunk_start]
                )
        snapshot = np.concatenate(pieces).astype(np.float32, copy=False)
        return snapshot, start_sample


    # Final inference must see every sample captured since the last reset.
    def make_final_snapshot() -> tuple[np.ndarray, int]:
        if not audio_chunks:
            return np.empty(0, dtype=np.float32), buffer_start_sample

        return np.concatenate([chunk for _, chunk in audio_chunks]).astype(
            np.float32,
            copy=False,
        ), buffer_start_sample


    # ========================================================
    # Send partial response
    # ========================================================

    async def send_partial(
        state: dict[str, Any],
        result: dict[str, Any],
    ):

        await websocket.send_json({
            "type":
                "partial",

            "text":
                state[
                    "text"
                ],

            "committed_text":
                state[
                    "committed_text"
                ],

            "partial_text":
                state[
                    "partial_text"
                ],

            "committed_until":
                state[
                    "committed_until"
                ],

            "committed_words":
                state[
                    "committed_words"
                ],

            "partial_words":
                state[
                    "partial_words"
                ],

            "window_start":
                result[
                    "window_start"
                ],

            "window_end":
                result[
                    "window_end"
                ],

            "window_seconds":
                result[
                    "seconds"
                ],

            "samples":
                result[
                    "samples"
                ],

            "feature_frames":
                result[
                    "feature_frames"
                ],

            "encoded_frames":
                result[
                    "encoded_frames"
                ],
        })


    # ========================================================
    # Partial worker
    # ========================================================

    async def run_partial():

        nonlocal inference_running
        nonlocal inference_pending
        nonlocal samples_since_inference


        if inference_running:

            inference_pending = True

            return


        inference_running = True


        try:

            while connection_alive:

                inference_pending = False


                this_generation = (
                    generation
                )


                (
                    snapshot,
                    snapshot_start_sample,
                ) = make_live_snapshot()


                samples_since_inference = 0


                if (
                    snapshot.size == 0
                ):

                    break


                snapshot_start_seconds = (
                    snapshot_start_sample
                    /
                    SAMPLE_RATE
                )


                try:

                    result = (
                        await transcribe_pcm(
                            snapshot,
                            snapshot_start_seconds,
                        )
                    )


                    # Reset/flush happened while CPU was busy.
                    if (
                        this_generation
                        !=
                        generation
                    ):

                        break


                    state = transcript.update(
                        result["words"],
                        result["window_end"],
                        result["window_start"],
                    )


                    # Keep all audio for context and the authoritative flush.
                    trim_audio_buffer()


                    if connection_alive:

                        await send_partial(
                            state,
                            result,
                        )


                except Exception as exc:

                    if connection_alive:

                        await websocket.send_json({
                            "type":
                                "error",

                            "message":
                                str(exc),
                        })


                # If more audio arrived during CPU inference,
                # immediately process newest state instead of
                # queueing obsolete intermediate jobs.
                if (
                    inference_pending
                    or
                    samples_since_inference
                    >=
                    LIVE_STEP_SAMPLES
                ):

                    continue


                break


        finally:

            inference_running = False


    # ========================================================
    # Receive loop
    # ========================================================

    try:

        while True:

            message = (
                await websocket.receive()
            )


            if (
                message.get("type")
                ==
                "websocket.disconnect"
            ):

                break


            # =================================================
            # Binary Float32 PCM
            # =================================================

            binary_data = (
                message.get("bytes")
            )


            if (
                binary_data
                is not None
            ):

                if (
                    len(binary_data)
                    %
                    4
                    !=
                    0
                ):

                    await websocket.send_json({
                        "type":
                            "error",

                        "message":
                            (
                                "PCM byte count must "
                                "be divisible by 4."
                            ),
                    })

                    continue


                chunk = np.frombuffer(
                    binary_data,
                    dtype="<f4",
                ).astype(
                    np.float32,
                    copy=True,
                )


                if (
                    chunk.size == 0
                ):

                    continue


                retain_audio_chunk(chunk, total_samples_received)


                total_samples_received += (
                    chunk.size
                )


                samples_since_inference += (
                    chunk.size
                )


                # Prevent buffer from growing forever.
                trim_audio_buffer()


                if (
                    samples_since_inference
                    >=
                    LIVE_STEP_SAMPLES
                ):

                    samples_since_inference = 0


                    if inference_running:

                        inference_pending = True

                    else:

                        asyncio.create_task(
                            run_partial()
                        )


                continue


            # =================================================
            # JSON control message
            # =================================================

            text_data = (
                message.get("text")
            )


            if (
                text_data is None
            ):

                continue


            try:

                payload = json.loads(
                    text_data
                )

            except json.JSONDecodeError:

                await websocket.send_json({
                    "type":
                        "error",

                    "message":
                        "Invalid JSON message.",
                })

                continue


            control_type = (
                payload.get("type")
            )


            # =================================================
            # FLUSH
            #
            # Decode the entire retained capture once, exactly as the
            # HTTP /transcribe endpoint does. This authoritative final
            # replaces the provisional committed/partial display and cannot
            # duplicate or omit text by joining overlapping windows.
            # =================================================

            if control_type == "flush":
                generation += 1
                inference_pending = False

                snapshot, snapshot_start_sample = make_final_snapshot()

                if snapshot.size == 0:
                    result = {
                        "text": "",
                        "words": [],
                        "window_end": 0.0,
                        "samples": 0,
                        "seconds": 0.0,
                        "feature_frames": 0,
                        "encoded_frames": 0,
                    }
                else:
                    try:
                        result = await transcribe_pcm(
                            snapshot,
                            snapshot_start_sample / SAMPLE_RATE,
                        )
                    except Exception as exc:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(exc),
                        })
                        continue

                final_words = [word.to_json() for word in result["words"]]
                final_text = result["text"]

                await websocket.send_json({
                    "type": "final",
                    "text": final_text,
                    "committed_text": final_text,
                    "partial_text": "",
                    "committed_until": result["window_end"],
                    "committed_words": final_words,
                    "partial_words": [],
                    "samples": result["samples"],
                    "seconds": result["seconds"],
                    "feature_frames": result["feature_frames"],
                    "encoded_frames": result["encoded_frames"],
                })

                continue


            # =================================================
            # RESET
            # =================================================

            if (
                control_type
                ==
                "reset"
            ):

                generation += 1

                inference_pending = False


                transcript.reset()


                audio_chunks = []


                buffer_start_sample = (
                    total_samples_received
                )


                samples_since_inference = 0


                await websocket.send_json({
                    "type":
                        "reset",

                    "committed_text":
                        "",

                    "partial_text":
                        "",
                })


                continue


            await websocket.send_json({
                "type":
                    "error",

                "message":
                    (
                        "Unknown control message: "
                        f"{control_type}"
                    ),
            })


    except WebSocketDisconnect:

        pass


    except Exception as exc:

        try:

            await websocket.send_json({
                "type":
                    "error",

                "message":
                    str(exc),
            })

        except Exception:

            pass


    finally:

        connection_alive = False


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()

    print(
        "=========================================="
    )

    print(
        " Gurbani Live RNNT ASR"
    )

    print(
        "=========================================="
    )

    print(
        "Provider: CPUExecutionProvider"
    )

    print(
        "GPU: disabled"
    )

    print(
        f"Sample rate: "
        f"{SAMPLE_RATE}"
    )

    print(
        f"Live step: "
        f"{LIVE_STEP_SECONDS:.2f}s"
    )

    print(
        f"Live window: "
        f"{LIVE_WINDOW_SECONDS:.2f}s"
        f" (up to {MAX_LIVE_WINDOW_SECONDS:.2f}s with 5s of trailing-word pre-roll)"
    )

    print("Decode mode: full-capture re-decode")

    print(
        f"Right-edge guard: "
        f"{RIGHT_EDGE_GUARD_SECONDS:.2f}s"
    )

    print(
        f"Stable repeats: "
        f"{STABLE_REPEATS}"
    )

    print(
        f"Timestamp tolerance: "
        f"{TIMESTAMP_TOLERANCE_SECONDS:.2f}s"
    )

    print(
        f"Stability timestamp tolerance: "
        f"{STABILITY_MATCH_TOLERANCE_SECONDS:.2f}s"
    )

    print(
        f"Encoder frame duration: "
        f"{ENCODER_FRAME_SECONDS:.3f}s"
    )

    print(
        "Timestamp transcript stabilization: enabled"
    )

    print(
        "RNNT local tokens: 0..255"
    )

    print(
        f"Joint blank: "
        f"{JOINT_BLANK_ID}"
    )

    print(
        f"Predictor SOS: "
        f"{PREDICTOR_START_ID}"
    )

    print(
        "Predictor feedback: local 0..256"
    )

    print(
        "=========================================="
    )

    print()
