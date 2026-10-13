"""Standalone public Gurbani stream and pankti server.

Run with: uvicorn server:app --host 127.0.0.1 --port 8000
The cache-aware NeMo streaming ONNX recognizer is loaded from
server_asr_stream.py. Model assets are expected in kirtan_stream_model/.
"""
from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections import deque
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bani_matching import OrderedBaniMatcher, bani_family
from delayed_audio import DelayedAudio, FrameParser


BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load a small .env file without adding a runtime dependency."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        os.environ.setdefault(key.strip(), value)


_load_dotenv(BASE_DIR / ".env")

LIVE_STREAM_URL = os.getenv(
    "LIVE_STREAM_URL", "https://live.sgpc.net:8443/;stream.mp3"
).strip()
ASR_REPLAY_SECONDS = max(2.0, min(8.0, float(os.getenv("ASR_REPLAY_SECONDS", "6"))))
PLAYBACK_DELAY_SECONDS = max(0.0, min(120.0, float(os.getenv("PLAYBACK_DELAY_SECONDS", "30"))))
# Absorb source startup bursts without dropping PCM while recognition is
# ahead of the delayed listener. The recognizer itself still holds <=8s.
ASR_PCM_QUEUE_SECONDS = max(8.0, min(128.0, float(os.getenv("ASR_PCM_QUEUE_SECONDS", str(PLAYBACK_DELAY_SECONDS + 8)))))
ASR_SAMPLE_RATE = 16000
PCM_READ_BYTES = 12800
ASR_PCM_QUEUE_CHUNKS = max(8, int(ASR_PCM_QUEUE_SECONDS * ASR_SAMPLE_RATE * 4 / PCM_READ_BYTES))
SOURCE_READ_TIMEOUT_SECONDS = max(
    15.0, min(600.0, float(os.getenv("SOURCE_READ_TIMEOUT_SECONDS", "60")))
)
SOURCE_RETRY_MAX_SECONDS = max(
    60.0, min(3600.0, float(os.getenv("SOURCE_RETRY_MAX_SECONDS", "300")))
)
SOURCE_STABLE_SECONDS = max(
    30.0, min(600.0, float(os.getenv("SOURCE_STABLE_SECONDS", "120")))
)
_db_setting = Path(os.getenv("BANI_DB_PATH", "data/bani.db"))
BANI_DB_PATH = _db_setting if _db_setting.is_absolute() else BASE_DIR / _db_setting
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "/usr/bin/ffmpeg")
ASR_INTERNAL_WS_URL = os.getenv(
    "ASR_INTERNAL_WS_URL",
    f"ws://127.0.0.1:{os.getenv('PORT', '8000')}/asr/ws/transcribe",
).strip()
if "mode=live" not in ASR_INTERNAL_WS_URL:
    ASR_INTERNAL_WS_URL += "&mode=live" if "?" in ASR_INTERNAL_WS_URL else "?mode=live"
PUNJABI_LANGUAGE_SOURCE_ID = int(os.getenv("PUNJABI_LANGUAGE_SOURCE_ID", "6"))
ENGLISH_LANGUAGE_SOURCE_ID = int(os.getenv("ENGLISH_LANGUAGE_SOURCE_ID", "1"))
MATCH_HOLD_SECONDS = max(1.0, min(300.0, float(os.getenv("MATCH_HOLD_SECONDS", "300"))))
PUBLIC_DIR = BASE_DIR / "public"
MODEL_DIR = BASE_DIR / "model"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gurbani-live")

_speech_column: str | None = None
_pankti_rows: list[dict[str, Any]] | None = None
_pankti_token_index: dict[str, tuple[int, ...]] = {}
_pankti_shabad_index: dict[str, tuple[int, ...]] = {}
_bani_matcher: OrderedBaniMatcher | None = None
_live_match: dict[str, Any] | None = None
_live_match_at = 0.0
_active_shabad_id: str | None = None
_pending_shabad_id: str | None = None
_pending_shabad_hits = 0
_pending_pankti_id: str | None = None
_pending_pankti_hits = 0
_pending_pankti_audio_time: float | None = None
_stream_started_at: float | None = None
_stream_wall_started_at: float | None = None
_stream_audio_samples_total = 0
_source_connected = False
_source_last_chunk_at: float | None = None
_source_last_chunk_monotonic: float | None = None
_source_audio_chunk_count = 0
_pankti_timeline: list[tuple[float, dict[str, Any]]] = []
_audio_sessions: dict[str, dict[str, Any]] = {}
_stream_connected = False
_live_task: asyncio.Task | None = None
_audio_delay_task: asyncio.Task | None = None
_asr_module: Any = None
_debug_clients: set[WebSocket] = set()
_debug_final_text = ""
_debug_partial_text = ""
_debug_final_words: list[str] = []
_debug_partial_words: list[str] = []
_debug_timed_final_words: list[dict[str, Any]] = []
_debug_timed_partial_words: list[dict[str, Any]] = []
_debug_recompute_from: float | None = None
_audio_subscribers: set[asyncio.Queue[tuple[float, bytes] | None]] = set()
_live_audio_media_type = "audio/aac"
_delayed_audio = DelayedAudio(PLAYBACK_DELAY_SECONDS)

app = FastAPI(title="BaniGuru Live Gurbani")
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="public")


def _sqlite_ro() -> sqlite3.Connection:
    if not BANI_DB_PATH.is_file():
        raise RuntimeError(f"Bani database is missing: {BANI_DB_PATH}")
    conn = sqlite3.connect(f"file:{BANI_DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_database(path: Path) -> str:
    """Check the current database and require its normalized speech field."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"panktis", "lines"}.issubset(tables):
            raise RuntimeError("Database must contain panktis and lines tables.")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(panktis)")}
        if "unicode_speech" not in columns:
            raise RuntimeError("The configured database must have panktis.unicode_speech.")
        return "unicode_speech"
    finally:
        conn.close()


def _ensure_database() -> str:
    if not BANI_DB_PATH.is_file():
        raise FileNotFoundError(f"Bani database is missing: {BANI_DB_PATH}")
    speech_column = _validate_database(BANI_DB_PATH)
    logger.info("Using current Gurbani database at %s", BANI_DB_PATH)
    return speech_column


def _normalise_query(text: str) -> str:
    text = re.sub(r"[^\u0A00-\u0A7F\s]", " ", text or "")
    return " ".join(text.split())


@lru_cache(maxsize=65536)
def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return SequenceMatcher(None, left, right).ratio()


def _ordered_alignment_score(expected: list[str], heard: list[str]) -> tuple[float, int, int]:
    """Score one pankti against in-order ASR words with insertions/deletions."""
    best = (0.0, 0, -1)
    if not expected or not heard:
        return best
    for end in range(1, len(heard) + 1):
        for size in range(max(1, len(expected) - 2), min(len(heard), len(expected) + 5) + 1):
            start = end - size
            if start < 0:
                continue
            window = heard[start:end]
            # Extra ASR words are cheap; missing pankti words cost more.
            dp = [[0.0] * (size + 1) for _ in range(len(expected) + 1)]
            matched = [[0] * (size + 1) for _ in range(len(expected) + 1)]
            for i in range(1, len(expected) + 1):
                dp[i][0] = dp[i - 1][0] + 0.68
                matched[i][0] = matched[i - 1][0]
            for j in range(1, size + 1):
                dp[0][j] = dp[0][j - 1] + 0.04
                matched[0][j] = matched[0][j - 1]
            for i, pword in enumerate(expected, 1):
                for j, qword in enumerate(window, 1):
                    similarity = _token_similarity(pword, qword)
                    match_cost = 1.0 - similarity if similarity >= 0.35 else 1.0
                    options = (
                        (dp[i - 1][j - 1] + match_cost, matched[i - 1][j - 1] + int(similarity >= 0.55)),
                        (dp[i - 1][j] + 0.68, matched[i - 1][j]),
                        (dp[i][j - 1] + 0.04, matched[i][j - 1]),
                    )
                    dp[i][j], matched[i][j] = min(options, key=lambda value: value[0])
            score = max(0.0, 1.0 - dp[-1][-1] / len(expected))
            # A verse that matched earlier in the rolling transcript should
            # yield to words that align near the newest ASR hypothesis.
            score = max(0.0, score - min(0.60, (len(heard) - end) * 0.05))
            candidate = (score, matched[-1][-1], end)
            if candidate > best:
                best = candidate
    return best


def _best_line_alignment(expected: list[str], heard: list[str]) -> tuple[float, int, int]:
    """Align near shared words, avoiding every possible substring scan."""
    best = _ordered_alignment_score(expected, heard)
    if len(expected) <= 3 or not heard:
        return best

    # Exact and near-exact word pairs indicate where a short ASR window lies
    # inside this full pankti. Test only the strongest alignment offsets.
    offsets: dict[int, float] = {}
    for line_index, line_word in enumerate(expected):
        for heard_index, heard_word in enumerate(heard):
            similarity = _token_similarity(line_word, heard_word)
            if similarity >= 0.35:
                offset = line_index - heard_index
                offsets[offset] = offsets.get(offset, 0.0) + similarity
    if not offsets:
        return best

    offset = max(offsets, key=offsets.get)
    for width in range(3, min(8, len(expected)) + 1):
        start = min(max(offset, 0), len(expected) - width)
        query_start = start - offset
        query_left = max(0, query_start - 1)
        query_right = min(len(heard), query_start + width + 2)
        heard_window = heard[query_left:query_right]
        if not heard_window:
            continue
        score, matched, end = _ordered_alignment_score(
            expected[start:start + width], heard_window
        )
        global_end = query_left + end
        # Account for newer transcript words outside this focused window so
        # old lines still yield to the live phrase tail.
        score = max(0.0, score - min(0.60, (len(heard) - global_end) * 0.05))
        candidate = (score, matched, global_end)
        if candidate > best:
            best = candidate
    return best


def _matched_word_offset(message: dict[str, Any], transcript: str, query_end: int,
                         line: str | None = None) -> float | None:
    """Map the matched query word to the ASR word's source-audio timestamp."""
    if query_end <= 0:
        return None
    timed_words = []
    for key in ("committed_words", "partial_words"):
        words = message.get(key)
        if isinstance(words, list):
            timed_words.extend(word for word in words if isinstance(word, dict))
    normalized = []
    for word in timed_words:
        token = _normalise_query(str(word.get("text") or ""))
        try:
            end = float(word.get("start" if line else "end"))
        except (TypeError, ValueError):
            continue
        if token:
            normalized.append((token.split()[-1], end))
    query_words = _normalise_query(transcript).split()[-12:]
    if not query_words or not normalized:
        return None
    # Stabilized words may be monotonic-merged by the server. Align backwards
    # to tolerate a clipped/rolling ASR word list and repeated words.
    qi = len(query_words) - 1
    wi = len(normalized) - 1
    found: dict[int, float] = {}
    while qi >= 0 and wi >= 0:
        if _token_similarity(query_words[qi], normalized[wi][0]) >= 0.62:
            found[qi] = normalized[wi][1]
            qi -= 1
        wi -= 1
    query_index = min(len(query_words) - 1, query_end - 1)
    if line:
        # Recognition arrives before playback: backdate a confirmed pankti to
        # the beginning of its supported phrase, not its second/third word.
        line_words = _normalise_query(line).split()
        blocks = SequenceMatcher(None, query_words[:query_index+1], line_words, autojunk=False).get_matching_blocks()
        supported = [b for b in blocks if b.size >= 2]
        if supported:
            query_index = supported[0].a
    return found.get(query_index)


def _merge_committed_text(previous: str, incoming: str) -> str:
    """Keep committed ASR words monotonic across rolling-window updates."""
    old_words = previous.split()
    new_words = incoming.split()
    if not new_words:
        return previous
    if not old_words:
        return " ".join(new_words)
    if len(new_words) >= len(old_words) and new_words[:len(old_words)] == old_words:
        return " ".join(new_words)
    if len(old_words) >= len(new_words) and old_words[:len(new_words)] == new_words:
        return previous

    # Rolling hypotheses may start at the current context window rather than
    # word zero. Join at the largest overlap and preserve every earlier final.
    max_overlap = min(len(old_words), len(new_words), 80)
    for overlap in range(max_overlap, 0, -1):
        tail = old_words[-overlap:]
        head = new_words[:overlap]
        if all(_token_similarity(a, b) >= 0.78 for a, b in zip(tail, head)):
            suffix = new_words[overlap:]
            return " ".join(old_words + suffix)

    # Ignore a delayed replay of already committed text; otherwise append it
    # so a short replacement hypothesis cannot erase prior committed words.
    if any(
        old_words[i:i + len(new_words)] == new_words
        for i in range(max(0, len(old_words) - len(new_words) - 8), len(old_words) - len(new_words) + 1)
    ):
        return previous
    return " ".join(old_words + new_words)


def _timed_word_records(
    items: Any,
    offset_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Validate word timestamps and map them to the source stream clock."""
    if not isinstance(items, list):
        return []
    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        try:
            start = float(item.get("start")) + offset_seconds
            end = float(item.get("end")) + offset_seconds
        except (TypeError, ValueError):
            continue
        if not text or end < start:
            continue
        records.append({"text": text, "start": start, "end": end})
    return sorted(records, key=lambda word: (word["start"], word["end"]))


def _recompute_timed_final_words(
    previous: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    rewind_from: float | None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Keep the stable prefix and rebuild from the latest word's start time.

    Rewinding one word lets the next rolling ASR hypothesis replace a word
    that was previously shown as committed, instead of text-merging a stale
    version or duplicating words when a partial is promoted.
    """
    if rewind_from is None and previous:
        rewind_from = float(previous[-1]["start"])

    if rewind_from is None:
        merged = incoming
    else:
        # A word ending at the rewind point belongs to the preserved prefix;
        # the word starting there is regenerated from the new hypothesis.
        prefix = [word for word in previous if float(word["end"]) <= rewind_from]
        suffix = [
            word for word in incoming
            if float(word["end"]) > rewind_from or float(word["start"]) >= rewind_from
        ]
        merged = prefix + suffix

    merged.sort(key=lambda word: (float(word["start"]), float(word["end"])))
    deduped: list[dict[str, Any]] = []
    for word in merged:
        if (
            deduped
            and deduped[-1]["text"] == word["text"]
            and abs(float(deduped[-1]["start"]) - float(word["start"])) <= 0.001
            and abs(float(deduped[-1]["end"]) - float(word["end"])) <= 0.001
        ):
            continue
        deduped.append(word)

    # Do not move the rewind point backward if the latest hypothesis
    # temporarily demotes the anchor word to partial. Keep revising from the
    # same point until a later committed word advances it.
    if deduped:
        latest_start = float(deduped[-1]["start"])
        rewind_from = max(rewind_from, latest_start) if rewind_from is not None else latest_start
    return deduped, rewind_from


def _translation_expression(
    conn: sqlite3.Connection, language_source_id: int
) -> tuple[str, list[Any]]:
    """Build a line translation lookup for current and legacy DB schemas."""
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "translations" in tables:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(translations)")}
        if {"line_id", "language_source_id", "translation"}.issubset(columns):
            return (
                "COALESCE((SELECT t.translation FROM translations t "
                "WHERE t.line_id=l.id AND t.language_source_id=? "
                "AND trim(COALESCE(t.translation,''))<>'' LIMIT 1), '')",
                [language_source_id],
            )
        if (
            {"line_id", "translation_source_id", "translation"}.issubset(columns)
            and "translation_sources" in tables
        ):
            source_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(translation_sources)")
            }
            if "language_source_id" in source_columns:
                return (
                    "COALESCE((SELECT t.translation FROM translations t "
                    "JOIN translation_sources ts ON ts.id=t.translation_source_id "
                    "WHERE t.line_id=l.id AND ts.language_source_id=? "
                    "AND trim(COALESCE(t.translation,''))<>'' LIMIT 1), '')",
                    [language_source_id],
                )
        if {"line_id", "translation_source_id", "translation"}.issubset(columns):
            return (
                "COALESCE((SELECT t.translation FROM translations t "
                "WHERE t.line_id=l.id AND t.translation_source_id=? "
                "AND trim(COALESCE(t.translation,''))<>'' LIMIT 1), '')",
                [language_source_id],
            )
    if "languages" in tables:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(languages)")}
        content_column = next(
            (name for name in ("translation", "translation_text", "text", "value", "content", "gurmukhi") if name in columns),
            None,
        )
        if {"line_id", "language_source_id"}.issubset(columns) and content_column:
            return (
                f"COALESCE((SELECT lang.{content_column} FROM languages lang "
                "WHERE lang.line_id=l.id AND lang.language_source_id=? "
                f"AND trim(COALESCE(lang.{content_column},''))<>'' LIMIT 1), '')",
                [language_source_id],
            )
    return "''", []


def _build_pankti_index() -> int:
    """Load pankti text once and index normalized words for live lookup."""
    global _pankti_rows, _pankti_token_index, _pankti_shabad_index, _speech_column, _bani_matcher
    if _speech_column is None:
        _speech_column = _validate_database(BANI_DB_PATH)
    speech_expr = f"REPLACE(COALESCE(p.{_speech_column}, ''), ',', '')"
    with _sqlite_ro() as conn:
        punjabi_sql, punjabi_params = _translation_expression(
            conn, PUNJABI_LANGUAGE_SOURCE_ID
        )
        english_sql, english_params = _translation_expression(
            conn, ENGLISH_LANGUAGE_SOURCE_ID
        )
        source_rows = conn.execute(
            f"""SELECT CAST(p.id AS TEXT) AS pankti_id,
                       CAST(p.shabad_id AS TEXT) AS shabad_id,
                       {speech_expr} AS speech,
                       COALESCE(NULLIF({punjabi_sql}, ''), p.gurmukhi, '') AS punjabi,
                       COALESCE(l.gurmukhi, '') AS legacy_gurmukhi,
                       {english_sql} AS english,
                       l.order_id AS order_id,
                       l.type_id AS _line_type
                FROM panktis p
                LEFT JOIN lines l ON l.id=p.id
                WHERE p.shabad_id IS NOT NULL
                  AND COALESCE(l.type_id, 0)<>2""",
            punjabi_params + english_params,
        ).fetchall()

    rows: list[dict[str, Any]] = []
    postings: dict[str, list[int]] = {}
    shabad_rows: dict[str, list[int]] = {}
    for source_row in source_rows:
        record = dict(source_row)
        tokens = _normalise_query(str(record["speech"] or "")).split()
        if len(tokens) < 2:
            continue
        record["_tokens"] = tokens
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        record["_token_counts"] = token_counts
        row_index = len(rows)
        rows.append(record)
        shabad_rows.setdefault(str(record["shabad_id"]), []).append(row_index)
        for token in set(tokens):
            postings.setdefault(token, []).append(row_index)

    _pankti_rows = rows
    _pankti_token_index = {token: tuple(indexes) for token, indexes in postings.items()}
    _pankti_shabad_index = {
        shabad_id: tuple(indexes) for shabad_id, indexes in shabad_rows.items()
    }
    by_id = {row["pankti_id"]: row for row in rows}
    routes: dict[int, dict[str, Any]] = {}
    with _sqlite_ro() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        # The shipped schema calls this table banis; accept the alternate
        # spelling used by some database exports as well.
        bani_table = next((name for name in ("banis", "banies") if name in tables), None)
        if bani_table and "bani_lines" in tables:
            for entry in conn.execute(f"""
                SELECT b.id AS bani_id, b.name_english, bl.line_group, p.id AS pankti_id
                FROM {bani_table} b
                JOIN bani_lines bl ON bl.bani_id=b.id
                JOIN panktis p ON p.id=bl.line_id
                JOIN lines l ON l.id=p.id
                ORDER BY b.id, bl.line_group, l.order_id, p.id
            """):
                family = bani_family(entry["name_english"])
                row = by_id.get(str(entry["pankti_id"]))
                if family and row is not None:
                    route = routes.setdefault(entry["bani_id"], {
                        "name": entry["name_english"], "family": family, "lines": []})
                    route["lines"].append({**row, "line_group": entry["line_group"]})
    _bani_matcher = OrderedBaniMatcher(routes, _token_similarity, _normalise_query)
    logger.info(
        "Loaded %s panktis into the live word index (%s distinct words)",
        len(rows),
        len(postings),
    )
    return len(rows)


def _pankti_candidates(
    query_words: list[str], preferred_shabad_id: str | None
) -> list[dict[str, Any]]:
    """Retrieve likely lines by exact word postings, then score them fuzzily."""
    if _pankti_rows is None:
        _build_pankti_index()
    assert _pankti_rows is not None

    last_position: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    for position, word in enumerate(query_words):
        last_position[word] = position
        if len(word) >= 2 and word in _pankti_token_index:
            query_counts[word] = query_counts.get(word, 0) + 1
    indexed_words = list(query_counts)
    indexed_words.sort(
        key=lambda word: (
            len(_pankti_token_index[word]),
            -last_position[word],
        )
    )
    indexed_words = indexed_words[:12]

    hits: dict[int, tuple[int, int]] = {}
    for word in indexed_words:
        position = last_position[word]
        for row_index in _pankti_token_index[word]:
            record = _pankti_rows[row_index]
            if preferred_shabad_id and record["shabad_id"] != preferred_shabad_id:
                continue
            count, latest = hits.get(row_index, (0, -1))
            occurrence_weight = min(
                query_counts[word],
                int(record.get("_token_counts", {}).get(word, 1)),
            )
            hits[row_index] = (
                count + occurrence_weight,
                max(latest, position),
            )

    ranked = sorted(
        hits,
        key=lambda row_index: (
            hits[row_index][0],
            hits[row_index][1],
        ),
        reverse=True,
    )[:32]
    return [_pankti_rows[row_index] for row_index in ranked]


def _match_active_shabad_fragment(
    query_words: list[str], shabad_id: str, min_score: float
) -> dict[str, Any] | None:
    """Find a current-shabad pankti from a short, possibly noisy ASR tail.

    Two ordered words may advance the displayed line when their fuzzy
    similarities are strong. A single word is accepted only when it is long
    and distinctive within this shabad, so common words cannot make the line
    jump around.
    """
    if _pankti_rows is None:
        _build_pankti_index()
    assert _pankti_rows is not None
    row_indexes = _pankti_shabad_index.get(shabad_id, ())
    if not row_indexes or not query_words:
        return None

    tail = query_words[-5:]
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    one_word_hits: list[tuple[float, int, dict[str, Any]]] = []
    query_end = len(query_words)
    for row_index in row_indexes:
        row = _pankti_rows[row_index]
        expected = row["_tokens"]
        # Ordered pairs are the normal short-context path. Permit a single
        # inserted/noisy word on either side while retaining word order.
        for qend in range(1, len(tail)):
            for qgap in (1, 2):
                qstart = qend - qgap
                if qstart < 0:
                    continue
                heard = (tail[qstart], tail[qend])
                global_end = query_end - len(tail) + qend + 1
                for pstart in range(len(expected) - 1):
                    for pgap in (1, 2):
                        if pstart + pgap >= len(expected):
                            continue
                        similarities = (
                            _token_similarity(expected[pstart], heard[0]),
                            _token_similarity(expected[pstart + pgap], heard[1]),
                        )
                        average = sum(similarities) / 2
                        if min(similarities) < 0.45 or average < 0.70:
                            continue
                        # Prefer a fresh phrase and an uninterrupted word
                        # sequence, while allowing one inserted ASR error.
                        recency_penalty = max(0, len(tail) - qend - 1) * 0.08
                        gap_penalty = (qgap - 1) * 0.08 + (pgap - 1) * 0.04
                        score = average - recency_penalty - gap_penalty
                        candidates.append((score, global_end, row))

        # A one-word fragment is only useful if it is distinctive in this
        # shabad and sufficiently similar. Track hits across lines below.
        newest = tail[-1]
        if len(newest) >= 3:
            for ppos, pword in enumerate(expected):
                similarity = _token_similarity(pword, newest)
                if similarity >= 0.80:
                    one_word_hits.append((similarity, query_end, row))

    best_pair = max(candidates, key=lambda item: (item[0], item[1]), default=None)
    if best_pair and best_pair[0] >= max(0.68, min_score):
        score, end, row = best_pair
        return _active_match_payload(row, score, end)

    # Only allow the one-word path when it identifies exactly one pankti;
    # repeat occurrences in that line are still one distinct pankti.
    unique_rows = {str(item[2]["pankti_id"]): item for item in one_word_hits}
    if len(unique_rows) == 1:
        score, end, row = next(iter(unique_rows.values()))
        if score >= max(0.86, min_score):
            return _active_match_payload(row, score, end)
    return None


def _active_match_payload(row: dict[str, Any], score: float, query_end: int) -> dict[str, Any]:
    return {
        "score": round(score, 4),
        "pankti_id": row["pankti_id"],
        "shabad_id": row["shabad_id"],
        "order_id": row["order_id"],
        "punjabi": row["punjabi"] or row["speech"],
        "legacy_gurmukhi": row["legacy_gurmukhi"],
        "english": row["english"],
        "_query_end": query_end,
    }


def match_live_pankti(
    transcript: str,
    preferred_shabad_id: str | None = None,
    min_score: float = 0.66,
) -> dict[str, Any] | None:
    """Match rolling ASR text to one pankti; never return a whole shabad."""
    query = _normalise_query(transcript)
    # Search only the current phrase tail. The rolling ASR final transcript is
    # cumulative; older exact lines must not keep winning over today's pankti.
    qtokens = query.split()[-12:]
    if not qtokens:
        return None
    if preferred_shabad_id:
        return _match_active_shabad_fragment(qtokens, str(preferred_shabad_id), min_score)
    rows = _pankti_candidates(qtokens, preferred_shabad_id)

    best: tuple[float, int, dict[str, Any]] | None = None
    for row in rows:
        ptokens = row["_tokens"]
        recent_qtokens = qtokens[-12:]
        qcompact = "".join(recent_qtokens)
        pcompact = "".join(ptokens)
        if pcompact and qcompact.endswith(pcompact):
            score = 1.0
            latest_end = len(recent_qtokens)
        else:
            score, matched_words, latest_end = _best_line_alignment(ptokens, recent_qtokens)
            if matched_words < min(3, len(ptokens)):
                continue
        if best is None or (score, latest_end) > (best[0], best[1]):
            best = (score, latest_end, row)

    if best is None or best[0] < min_score:
        return None
    row = best[2]
    return {
        "score": round(best[0], 4),
        "pankti_id": row["pankti_id"],
        "shabad_id": row["shabad_id"],
        "order_id": row["order_id"],
        "punjabi": row["punjabi"] or row["speech"],
        "legacy_gurmukhi": row["legacy_gurmukhi"],
        "english": row["english"],
        "_query_end": best[1],
    }


async def _handle_asr_messages(socket: Any, audio_offset_seconds: float = 0.0) -> None:
    global _live_match, _live_match_at
    global _active_shabad_id, _pending_shabad_id, _pending_shabad_hits
    global _pending_pankti_id, _pending_pankti_hits
    global _pending_pankti_audio_time
    global _debug_final_text, _debug_partial_text
    global _debug_final_words, _debug_partial_words
    global _debug_timed_final_words, _debug_timed_partial_words, _debug_recompute_from
    async for raw in socket:
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        message_type = message.get("type")
        if message_type == "reset":
            # Partial words are provisional; clear them on decoder reset so
            # the page never presents stale hypotheses as live text.
            if _debug_partial_text:
                _debug_partial_text = ""
                _debug_partial_words = []
                await _broadcast_debug_words()
            continue
        if message_type not in {"partial", "final"}:
            continue
        try:
            lag = _stream_audio_samples_total / ASR_SAMPLE_RATE - (audio_offset_seconds + float(message.get("window_end", 0.0)))
            logger.info(
                "ASR_TEST timing lag=%.2fs type=%s mode=%s processing=%.3fs lock_wait=%.3fs",
                lag, message_type, message.get("recognition_mode", "legacy"),
                float(message.get("processing_seconds") or 0.0),
                float(message.get("inference_lock_wait_seconds") or 0.0),
            )
        except (TypeError, ValueError):
            pass
        if "final_text" in message:
            incoming_final = str(message.get("final_text") or "").strip()
        elif "committed_text" in message:
            incoming_final = str(message.get("committed_text") or "").strip()
        elif message_type == "final":
            incoming_final = str(message.get("text") or "").strip()
        else:
            incoming_final = _debug_final_text

        if isinstance(message.get("committed_words"), list):
            incoming_timed = _timed_word_records(
                message.get("committed_words"), audio_offset_seconds
            )
            _debug_timed_final_words, _debug_recompute_from = (
                _recompute_timed_final_words(
                    _debug_timed_final_words,
                    incoming_timed,
                    _debug_recompute_from,
                )
            )
            # Live matching only needs the recent phrase tail. Bound this
            # history so a continuously running service does not keep
            # rebuilding and serializing the entire stream transcript.
            latest = max((w["start"] for w in _debug_timed_final_words), default=0.0)
            _debug_timed_final_words = [w for w in _debug_timed_final_words
                                        if w["start"] >= latest-PLAYBACK_DELAY_SECONDS-90][-2000:]
            final_text = " ".join(
                word["text"] for word in _debug_timed_final_words
            ).strip()
        else:
            # Compatibility with ASR clients that don't send timed words.
            final_text = _merge_committed_text(_debug_final_text, incoming_final)
        # The recognizer owns partial replacement. In particular an empty
        # partial means that its current hypothesis was withdrawn/promoted.
        partial_text = str(message.get("partial_text") or "").strip()
        _debug_timed_partial_words = _timed_word_records(message.get("partial_words"), audio_offset_seconds)
        changed = final_text != _debug_final_text or partial_text != _debug_partial_text
        _debug_final_text = final_text
        _debug_partial_text = partial_text
        _debug_final_words = final_text.split()[-30:]
        _debug_partial_words = partial_text.split()
        transcript = " ".join(part for part in (final_text, partial_text) if part).strip()
        if changed:
            await _broadcast_debug_words()
        # Idle updates retain transcript history, but are not fresh evidence
        # for matching or extending the lifetime of a previous pankti.
        if not transcript or message.get("audio_active") is False:
            continue
        # `match` must be local to this ASR update. Reusing its previous value
        # makes a no-match update look like a fresh repeat of the old pankti.
        match: dict[str, Any] | None = None
        try:
            now = time.time()
            if _active_shabad_id and now - _live_match_at > MATCH_HOLD_SECONDS:
                _active_shabad_id = None
                _pending_shabad_id = None
                _pending_shabad_hits = 0
                if _bani_matcher is not None:
                    _bani_matcher.family = None

            if _bani_matcher is not None:
                match = _bani_matcher.match(transcript, message)

            if match is not None:
                _pending_shabad_id = None
                _pending_shabad_hits = 0
                _active_shabad_id = str(match["_bani_observed_shabad_id"])
            elif _active_shabad_id:
                active_id = _active_shabad_id
                match = await asyncio.to_thread(
                    match_live_pankti, transcript[-10000:], active_id, 0.55
                )
                if match:
                    _pending_shabad_id = None
                    _pending_shabad_hits = 0
                else:
                    # Search outside the active shabad only for a high
                    # confidence candidate, then require it twice in a row.
                    candidate = await asyncio.to_thread(
                        match_live_pankti, transcript[-10000:], None, 0.82
                    )
                    if candidate and candidate["shabad_id"] == active_id:
                        match = candidate
                        _pending_shabad_id = None
                        _pending_shabad_hits = 0
                    elif candidate:
                        candidate_id = str(candidate["shabad_id"])
                        if candidate_id == _pending_shabad_id:
                            _pending_shabad_hits += 1
                        else:
                            _pending_shabad_id = candidate_id
                            _pending_shabad_hits = 1
                        if _pending_shabad_hits >= 2:
                            match = candidate
                            _active_shabad_id = candidate_id
                            _pending_shabad_id = None
                            _pending_shabad_hits = 0
                    else:
                        _pending_shabad_id = None
                        _pending_shabad_hits = 0
            else:
                match = await asyncio.to_thread(
                    match_live_pankti, transcript[-10000:], None, 0.66
                )
                if match:
                    _active_shabad_id = str(match["shabad_id"])
                    _pending_shabad_id = None
                    _pending_shabad_hits = 0
        except Exception:
            logger.exception("Live pankti matching failed")
            continue
        if match is None:
            _pending_pankti_id = None
            _pending_pankti_hits = 0
            _pending_pankti_audio_time = None
        if match:
            # Require sustained evidence before changing the displayed line.
            # A backward move needs especially strong evidence because short,
            # noisy ASR tails often resemble a previously sung refrain.
            current = _live_match
            try:
                window_end = float(message.get("window_end"))
            except (TypeError, ValueError):
                window_end = None
            matched_offset = _matched_word_offset(
                message, transcript, int(match.get("_query_end") or 0),
                str(match.get("punjabi") or "") if PLAYBACK_DELAY_SECONDS else None,
            )
            match_audio_time = (
                _stream_wall_started_at + max(0.0, audio_offset_seconds + matched_offset)
                if _stream_wall_started_at is not None and matched_offset is not None
                else (_stream_wall_started_at + max(0.0, audio_offset_seconds + window_end)
                      if _stream_wall_started_at is not None and window_end is not None
                      else time.time())
            )
            if (not match.get("_bani_direct") and current
                    and current.get("shabad_id") == match.get("shabad_id")):
                current_id = str(current.get("pankti_id"))
                candidate_id = str(match.get("pankti_id"))
                if candidate_id == current_id:
                    _pending_pankti_id = None
                    _pending_pankti_hits = 0
                else:
                    try:
                        is_backward = int(match.get("order_id") or 0) < int(current.get("order_id") or 0)
                    except (TypeError, ValueError):
                        is_backward = False
                    required_score = 0.82 if is_backward else 0.66
                    required_hits = 3 if is_backward else 2
                    if float(match.get("score") or 0.0) < required_score:
                        _pending_pankti_id = None
                        _pending_pankti_hits = 0
                        match = None
                    else:
                        if candidate_id == _pending_pankti_id:
                            _pending_pankti_hits += 1
                        else:
                            _pending_pankti_id = candidate_id
                            _pending_pankti_hits = 1
                            _pending_pankti_audio_time = match_audio_time
                        if _pending_pankti_hits < required_hits:
                            match = None
                        else:
                            _pending_pankti_id = None
                            _pending_pankti_hits = 0
                            match_audio_time = _pending_pankti_audio_time or match_audio_time
                            _pending_pankti_audio_time = None
            else:
                _pending_pankti_id = None
                _pending_pankti_hits = 0
                _pending_pankti_audio_time = None

        if match:
            if _bani_matcher is not None and not match.get("_bani_direct"):
                match = _bani_matcher.adopt(match, transcript, message)
            _live_match = match
            _live_match_at = time.time()
            # Do not extend the old line to the end of every rolling ASR
            # context window. Record transitions at the matched word time;
            # renew an unchanged line only occasionally for long refrains.
            public_match = {key: value for key, value in match.items() if not key.startswith("_")}
            last_at, last_match = _pankti_timeline[-1] if _pankti_timeline else (0.0, None)
            if (
                not last_match
                or last_match.get("pankti_id") != public_match.get("pankti_id")
                or last_match.get("bani_position") != public_match.get("bani_position")
                or last_match.get("bani_id") != public_match.get("bani_id")
                or last_match.get("match_mode") != public_match.get("match_mode")
                or match_audio_time - last_at >= 8.0
            ):
                position = bisect_right([event[0] for event in _pankti_timeline], match_audio_time)
                _pankti_timeline.insert(position, (match_audio_time, public_match))
                if len(_pankti_timeline) > 20000:
                    _pankti_timeline.pop(0)


async def _broadcast_debug_words() -> None:
    if not _debug_clients:
        return
    final_words = _debug_final_words[-30:]
    partial_words = _debug_partial_words
    message = {
        "type": "transcript",
        "final_text": " ".join(final_words),
        "partial_text": " ".join(partial_words),
        "final_words": final_words,
        "partial_words": partial_words,
        "text": " ".join(final_words + partial_words),
    }
    disconnected: list[WebSocket] = []
    for client in tuple(_debug_clients):
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        _debug_clients.discard(client)


async def _publish_audio_chunk(chunk: bytes, source_time: float | None = None) -> None:
    """Fan out frame-aligned original bytes at their delayed media time."""
    if source_time is None:
        source_time = (_stream_wall_started_at or time.time()) + _stream_audio_samples_total / ASR_SAMPLE_RATE
    for queue in tuple(_audio_subscribers):
        if queue.full():
            # Never silently discard audio: doing so would desynchronize the
            # listener's audio clock from the timestamped pankti timeline.
            _audio_subscribers.discard(queue)
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            queue.put_nowait(None)
            continue
        queue.put_nowait((source_time, chunk))


async def _release_delayed_audio() -> None:
    while True:
        for frame in _delayed_audio.release(time.monotonic()):
            await _publish_audio_chunk(frame.data, (_stream_wall_started_at or time.time()) + frame.start)
        await asyncio.sleep(0.02)


async def _feed_source_to_asr(response: Any, process: asyncio.subprocess.Process) -> None:
    """Send one upstream response to both public audio and ffmpeg/ASR."""
    global _stream_started_at, _stream_wall_started_at
    global _source_connected, _source_last_chunk_at, _source_last_chunk_monotonic
    global _source_audio_chunk_count
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    bytes_since_log = 0
    last_source_log = time.monotonic()
    parser = FrameParser(_delayed_audio.head)
    _delayed_audio.new_segment()
    try:
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            _source_connected = True
            _source_last_chunk_at = time.time()
            _source_last_chunk_monotonic = time.monotonic()
            _source_audio_chunk_count += 1
            bytes_since_log += len(chunk)
            if _source_last_chunk_monotonic - last_source_log >= 30.0:
                logger.info(
                    "Live source delivered %d bytes in 30s; public listeners=%d",
                    bytes_since_log,
                    len(_audio_subscribers),
                )
                bytes_since_log = 0
                last_source_log = _source_last_chunk_monotonic
            if _stream_started_at is None:
                _stream_started_at = time.monotonic()
            if _stream_wall_started_at is None:
                _stream_wall_started_at = time.time()
            frames = parser.feed(chunk)
            _delayed_audio.append(frames, time.monotonic())
            if frames:
                process.stdin.write(b"".join(frame.data for frame in frames))
                await process.stdin.drain()
    finally:
        _source_connected = False
        process.stdin.close()
        try:
            await process.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def _read_ffmpeg_pcm(
    process: asyncio.subprocess.Process,
    pcm_queue: asyncio.Queue[tuple[int, bytes] | None],
    history: deque[tuple[int, bytes]],
    source_done: asyncio.Event,
) -> None:
    """Drain FFmpeg continuously, retaining bounded replay and send queues."""
    global _stream_audio_samples_total
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout is unavailable")
    history_limit = int(ASR_REPLAY_SECONDS * ASR_SAMPLE_RATE)
    try:
        while True:
            pcm = await process.stdout.read(PCM_READ_BYTES)
            if not pcm:
                break
            pcm = pcm[:len(pcm) - (len(pcm) % 4)]
            if not pcm:
                continue
            start_sample = _stream_audio_samples_total
            sample_count = len(pcm) // 4
            _stream_audio_samples_total += sample_count
            history.append((start_sample, pcm))
            while (
                len(history) > 1
                and _stream_audio_samples_total - history[0][0] > history_limit
            ):
                history.popleft()
            item = (start_sample, pcm)
            if pcm_queue.full():
                # Never block the source/public-audio path on a stale ASR
                # backlog. A gap forces the ASR forwarder to re-prime from
                # the latest bounded replay window.
                try:
                    pcm_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                pcm_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass
    finally:
        source_done.set()
        if pcm_queue.full():
            try:
                pcm_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            pcm_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass


async def _forward_pcm_to_asr(
    pcm_queue: asyncio.Queue[tuple[int, bytes] | None],
    history: deque[tuple[int, bytes]],
    source_done: asyncio.Event,
) -> None:
    """Reconnect ASR independently and re-prime it from recent stream PCM."""
    global _stream_connected
    retry_delay = 1.0
    has_connected = False
    while True:
        receiver: asyncio.Task | None = None
        connected_at = time.monotonic()
        try:
            async with websockets.connect(
                ASR_INTERNAL_WS_URL,
                max_size=None,
                open_timeout=15,
                ping_interval=20,
                ping_timeout=30,
            ) as socket:
                ready_raw = await asyncio.wait_for(socket.recv(), timeout=20)
                ready = json.loads(ready_raw)
                if ready.get("type") != "ready":
                    raise RuntimeError(f"ASR service did not become ready: {ready_raw!r}")
                logger.info(
                    "ASR runtime build=%s mode=%s block=%s update=%ss KenLM=%s",
                    ready.get("build", "legacy"), ready.get("recognition_mode", "legacy"),
                    ready.get("block_seconds", "unknown"), ready.get("step_seconds", "unknown"),
                    ready.get("enable_language_model", "unknown"),
                )
                if ready.get("recognition_mode") != "blocks" and PLAYBACK_DELAY_SECONDS >= 30:
                    logger.warning(
                        "Delayed playback is connected to a rolling/legacy ASR service. "
                        "Verify /asr/health and deploy the no-audio-fix branch with ASR_RECOGNITION_MODE=blocks."
                    )

                snapshot = tuple(history)
                if snapshot and not has_connected:
                    # At initial startup, joining at the edge avoids making
                    # first captions several seconds older than the live audio.
                    # Full context replay is reserved for reconnect recovery.
                    initial_start = max(
                        snapshot[0][0],
                        _stream_audio_samples_total - ASR_SAMPLE_RATE,
                    )
                    snapshot = tuple(
                        (max(start, initial_start), chunk[(max(start, initial_start) - start) * 4:])
                        for start, chunk in snapshot
                        if start + len(chunk) // 4 > initial_start
                    )
                if snapshot:
                    replay_start = snapshot[0][0]
                    sent_until = snapshot[-1][0] + len(snapshot[-1][1]) // 4
                else:
                    replay_start = _stream_audio_samples_total
                    sent_until = replay_start
                receiver = asyncio.create_task(
                    _handle_asr_messages(socket, replay_start / ASR_SAMPLE_RATE)
                )
                _stream_connected = True
                has_connected = True
                logger.info(
                    "ASR connected at audio offset %.2fs; replaying %.2fs of recent PCM",
                    replay_start / ASR_SAMPLE_RATE,
                    max(0, sent_until - replay_start) / ASR_SAMPLE_RATE,
                )
                if snapshot:
                    await socket.send(b"".join(chunk for _, chunk in snapshot))

                while True:
                    if receiver.done():
                        error = receiver.exception()
                        if error:
                            raise error
                        raise RuntimeError("ASR WebSocket closed while live audio was active")
                    try:
                        item = await asyncio.wait_for(pcm_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if source_done.is_set():
                            return
                        continue
                    if item is None:
                        return
                    start_sample, pcm = item
                    end_sample = start_sample + len(pcm) // 4
                    if end_sample <= sent_until:
                        continue
                    if start_sample > sent_until:
                        raise RuntimeError(
                            "ASR PCM backlog overflowed; reconnecting with fresh replay context"
                        )
                    if start_sample < sent_until:
                        pcm = pcm[(sent_until - start_sample) * 4:]
                        start_sample = sent_until
                    if pcm:
                        await socket.send(pcm)
                        sent_until = end_sample
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _stream_connected = False
            if receiver is not None and not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            if source_done.is_set():
                raise RuntimeError("Audio source ended while ASR was disconnected") from exc
            logger.warning(
                "ASR WebSocket disconnected; keeping live audio running and retrying in %.1fs: %s",
                retry_delay,
                exc,
            )
            await asyncio.sleep(retry_delay)
            if time.monotonic() - connected_at >= 60.0:
                retry_delay = 1.0
            else:
                retry_delay = min(retry_delay * 2.0, 10.0)
        finally:
            _stream_connected = False
            if receiver is not None:
                if not receiver.done():
                    receiver.cancel()
                # Always consume the receiver result. During server shutdown
                # the peer may close at the same moment as cancellation.
                await asyncio.gather(receiver, return_exceptions=True)


async def _run_live_session() -> None:
    global _stream_connected, _source_connected, _live_audio_media_type
    timeout = httpx.Timeout(
        connect=15.0,
        read=SOURCE_READ_TIMEOUT_SECONDS,
        write=15.0,
        pool=15.0,
    )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream(
            "GET",
            LIVE_STREAM_URL,
            headers={"User-Agent": "Mozilla/5.0 BaniGuruLive/1.0", "Icy-MetaData": "0"},
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "audio/aac").split(";", 1)[0].strip()
            if content_type:
                _live_audio_media_type = content_type
            process = await asyncio.create_subprocess_exec(
                FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", "pipe:0",
                "-vn", "-ac", "1", "-ar", str(ASR_SAMPLE_RATE),
                "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            pcm_queue: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue(
                maxsize=ASR_PCM_QUEUE_CHUNKS
            )
            history: deque[tuple[int, bytes]] = deque()
            source_done = asyncio.Event()
            feeder = asyncio.create_task(_feed_source_to_asr(response, process))
            pcm_reader = asyncio.create_task(
                _read_ffmpeg_pcm(process, pcm_queue, history, source_done)
            )
            asr_forwarder = asyncio.create_task(
                _forward_pcm_to_asr(pcm_queue, history, source_done)
            )
            logger.info("Connected to the live source; ASR reconnects independently")
            try:
                await asyncio.gather(feeder, pcm_reader, asr_forwarder)
                code = await process.wait()
                raise RuntimeError(f"ffmpeg ended the live stream (status {code})")
            finally:
                _stream_connected = False
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=3)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                for task in (feeder, pcm_reader, asr_forwarder):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(feeder, pcm_reader, asr_forwarder, return_exceptions=True)


async def _live_worker() -> None:
    global _source_audio_chunk_count, _source_last_chunk_monotonic
    delay = 2.0
    while True:
        started_at = time.monotonic()
        chunks_before = _source_audio_chunk_count
        try:
            await _run_live_session()
        except asyncio.CancelledError:
            raise
        except Exception:
            source_was_stable = (
                _source_audio_chunk_count > chunks_before
                and _source_last_chunk_monotonic is not None
                and _source_last_chunk_monotonic - started_at >= SOURCE_STABLE_SECONDS
            )
            if source_was_stable:
                delay = 2.0
            retry_delay = delay
            logger.exception("Live source disconnected; retrying in %s seconds", retry_delay)
            await asyncio.sleep(retry_delay)
            delay = min(retry_delay * 2, SOURCE_RETRY_MAX_SECONDS)
        else:
            delay = min(delay * 2, SOURCE_RETRY_MAX_SECONDS)


@app.on_event("startup")
async def startup() -> None:
    global _speech_column, _asr_module, _live_task, _audio_delay_task
    _speech_column = await asyncio.to_thread(_ensure_database)
    await asyncio.to_thread(_build_pankti_index)
    # Loads the cache-aware NeMo streaming ONNX encoder/decoder.
    import server_asr_stream as asr_module
    _asr_module = asr_module
    app.mount("/asr", asr_module.app, name="onnx-asr")
    _live_task = asyncio.create_task(_live_worker())
    _audio_delay_task = asyncio.create_task(_release_delayed_audio())


@app.on_event("shutdown")
async def shutdown() -> None:
    global _live_task, _audio_delay_task
    if _live_task and not _live_task.done():
        _live_task.cancel()
        await asyncio.gather(_live_task, return_exceptions=True)
    _live_task = None
    if _audio_delay_task:
        _audio_delay_task.cancel()
        await asyncio.gather(_audio_delay_task, return_exceptions=True)
    _audio_delay_task = None


@app.get("/")
def public_page() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "live.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "stream_connected": _source_connected,
        "asr_connected": _stream_connected,
        "database": BANI_DB_PATH.name,
        "speech_column": _speech_column,
        "asr_build": getattr(_asr_module, "SERVER_BUILD", None),
        "recognition_mode": getattr(_asr_module, "RECOGNITION_MODE", None),
        "block_seconds": getattr(_asr_module, "BLOCK_SECONDS", None),
        "playback_delay_seconds": PLAYBACK_DELAY_SECONDS,
        "buffered_audio_seconds": _delayed_audio.head-(_delayed_audio.released[-1].end if _delayed_audio.released else 0.0),
    }


@app.get("/api/live/state", response_model=None)
def live_state(
    session_id: str | None = None,
    played_seconds: float = 0.0,
    client_started_at: float | None = None,
    known_revision: str | None = None,
) -> Any:
    """Return the pankti at this listener's playback position, or live edge."""
    now = time.time()
    bootstrap = _delayed_audio.bootstrap()
    delayed_position = bootstrap[0].start if bootstrap else max(0.0, _delayed_audio.head-PLAYBACK_DELAY_SECONDS)
    playhead_time = (_stream_wall_started_at or now) + delayed_position
    audio_started_at = None
    if session_id and re.fullmatch(r"[0-9a-fA-F-]{36}", session_id):
        session = _audio_sessions.get(session_id)
        if session:
            session["last_seen"] = time.monotonic()
            if client_started_at is not None and 0 < client_started_at <= now:
                session["client_started_at"] = client_started_at
            audio_started_at = session.get("first_audio_at")
            if audio_started_at is not None:
                playhead_time = audio_started_at + max(0.0, min(played_seconds, 86400.0))
    # Expire abandoned listener anchors to keep memory bounded.
    stale_before = time.monotonic() - 21600
    for key, value in tuple(_audio_sessions.items()):
        if value["last_seen"] < stale_before:
            _audio_sessions.pop(key, None)
    fresh_entry = next(
        ((at, match) for at, match in reversed(_pankti_timeline)
         if at <= playhead_time and playhead_time - at <= MATCH_HOLD_SECONDS),
        None,
    )
    if session_id and audio_started_at is not None:
        session = _audio_sessions.get(session_id)
        if fresh_entry is not None and session is not None:
            session["last_match_timestamp"] = fresh_entry[0]
            session["last_match"] = dict(fresh_entry[1])
        elif session is not None:
            last_at = session.get("last_match_timestamp")
            last_match = session.get("last_match")
            if (
                isinstance(last_at, (int, float))
                and isinstance(last_match, dict)
                and 0 <= playhead_time - last_at <= MATCH_HOLD_SECONDS
            ):
                fresh_entry = (float(last_at), last_match)
    if (
        fresh_entry is None
        and PLAYBACK_DELAY_SECONDS == 0
        and audio_started_at is None
        and _live_match
        and 0 <= now - _live_match_at <= MATCH_HOLD_SECONDS
    ):
        # A new listener receives the current match immediately while the
        # audio element establishes its own timestamp anchor.
        fresh_entry = (_live_match_at, _live_match)
    fresh_match = fresh_entry[1] if fresh_entry else None
    source_active = bool(
        _source_connected
        and _source_last_chunk_monotonic is not None
        and time.monotonic() - _source_last_chunk_monotonic <= SOURCE_READ_TIMEOUT_SECONDS
    )
    state = {
        "enabled": True,
        "playback_delay_seconds": PLAYBACK_DELAY_SECONDS,
        "playback_ready": bool(_delayed_audio.released),
        "audio_generation": _audio_sessions.get(session_id, {}).get("generation", 0),
        "playback_text": " ".join(w["text"] for w in (_debug_timed_final_words + _debug_timed_partial_words)
                                   if (_stream_wall_started_at or now)+w["start"] <= playhead_time)[-500:],
        "connected": source_active,
        "asr_connected": _stream_connected,
        "audio_url": f"/api/live/audio?session_id={session_id}" if session_id else "/api/live/audio",
        "audio_start_timestamp": audio_started_at,
        "client_started_at": (
            _audio_sessions.get(session_id, {}).get("client_started_at")
            if session_id else None
        ),
        "playhead_timestamp": playhead_time if audio_started_at is not None else None,
        "pankti_timestamp": fresh_entry[0] if fresh_entry else None,
        "pankti": fresh_match["punjabi"] if fresh_match else "",
        "punjabi": fresh_match["punjabi"] if fresh_match else "",
        "legacy_gurmukhi": fresh_match["legacy_gurmukhi"] if fresh_match else "",
        "english": fresh_match["english"] if fresh_match else "",
        "shabad_id": fresh_match["shabad_id"] if fresh_match else None,
        "pankti_id": fresh_match["pankti_id"] if fresh_match else None,
        "bani_id": fresh_match.get("bani_id") if fresh_match else None,
        "bani_name": fresh_match.get("bani_name") if fresh_match else None,
        "line_group": fresh_match.get("line_group") if fresh_match else None,
        "bani_position": fresh_match.get("bani_position") if fresh_match else None,
        "match_mode": fresh_match.get("match_mode") if fresh_match else None,
    }
    revision_payload = {
        key: state[key]
        for key in (
            "enabled", "connected", "asr_connected", "audio_url",
            "audio_start_timestamp", "client_started_at", "pankti",
            "legacy_gurmukhi", "english", "shabad_id", "pankti_id",
            "bani_id", "bani_name", "line_group", "bani_position", "match_mode",
            "playback_delay_seconds", "playback_ready", "audio_generation", "playback_text",
        )
    }
    revision = hashlib.sha256(
        json.dumps(revision_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    state["revision"] = revision
    if known_revision == revision:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return state


@app.websocket("/api/live/debug")
async def live_debug(websocket: WebSocket) -> None:
    await websocket.accept()
    _debug_clients.add(websocket)
    try:
        final_words = _debug_final_words[-30:]
        partial_words = _debug_partial_words
        await websocket.send_json({
            "type": "transcript",
            "final_text": " ".join(final_words),
            "partial_text": " ".join(partial_words),
            "final_words": final_words,
            "partial_words": partial_words,
            "text": " ".join(final_words + partial_words),
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _debug_clients.discard(websocket)


@app.get("/api/live/audio")
async def live_audio(session_id: str | None = None) -> StreamingResponse:
    if not session_id or not re.fullmatch(r"[0-9a-fA-F-]{36}", session_id):
        raise HTTPException(status_code=400, detail="A UUID session_id is required.")
    queue: asyncio.Queue[tuple[float, bytes] | None] = asyncio.Queue(maxsize=512)
    _audio_subscribers.add(queue)
    session = _audio_sessions.setdefault(session_id, {"last_seen": time.monotonic()})
    previous_queue = session.get("queue")
    if previous_queue is not None:
        _audio_subscribers.discard(previous_queue)
        while not previous_queue.empty():
            previous_queue.get_nowait()
        previous_queue.put_nowait(None)
    session["queue"] = queue
    session["last_seen"] = time.monotonic()
    session["generation"] = session.get("generation", 0) + 1
    generation = session["generation"]
    session["first_audio_at"] = None
    session.pop("last_match", None)
    session.pop("last_match_timestamp", None)
    for frame in _delayed_audio.bootstrap():
        queue.put_nowait(((_stream_wall_started_at or time.time()) + frame.start, frame.data))

    async def chunks():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                source_time, chunk = item
                if session["generation"] != generation:
                    break
                if session.get("first_audio_at") is None:
                    session["first_audio_at"] = source_time
                    logger.info(
                        "Public audio session %s received its first %d-byte chunk",
                        session_id,
                        len(chunk),
                    )
                yield chunk
        finally:
            _audio_subscribers.discard(queue)

    return StreamingResponse(
        chunks(),
        media_type=_live_audio_media_type,
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
