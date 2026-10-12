"""Soft word-context evidence from the application's independent Gurbani corpus.

This is a likelihood ratio, not a vocabulary constraint: unseen speech is
allowed, and no benchmark reference is loaded by the recognizer.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math
from pathlib import Path
import sqlite3
import unicodedata


def words(text):
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return [word for word in text.split() if not word.isnumeric()]


@lru_cache(maxsize=32768)
def token_prefix_score(prior, tokenizer, token_ids, previous):
    """Reuse exact prefix evidence across previews without re-decoding pieces."""
    flags = _tokenizer_flags(tokenizer)
    if not token_ids or flags is None:
        return prior.score(tokenizer.decode(token_ids), previous)
    starts, unsafe = flags
    if any(unsafe[token] for token in token_ids):
        return prior.score(tokenizer.decode(token_ids), previous)
    boundary = next((index for index in range(len(token_ids) - 1, 0, -1)
                     if starts[token_ids[index]]), 0)
    if not boundary:
        return prior.score(tokenizer.decode(token_ids), previous)
    tail = tokenizer.decode(token_ids[boundary:])
    # Removing a final verse number changes which word is considered partial
    # by the original scorer. Preserve its exact behavior via the full path.
    if tail.strip().isnumeric():
        return prior.score(tokenizer.decode(token_ids), previous)
    prefix, context = _completed_token_prefix(prior, tokenizer, token_ids[:boundary], previous)
    return prefix + prior.score(tail, context)


@lru_cache(maxsize=8)
def _tokenizer_flags(tokenizer):
    if not hasattr(tokenizer, "get_piece_size"):
        return None
    pieces = [tokenizer.id_to_piece(index) for index in range(tokenizer.get_piece_size())]
    starts = tuple(piece.startswith("▁") for piece in pieces)
    unsafe = tuple(any((unicodedata.category(char)[0] in "PS" and char != "▁") or char.isspace()
                       for char in piece) or "▁" in piece[1:] for piece in pieces)
    return starts, unsafe


@lru_cache(maxsize=32768)
def _completed_token_prefix(prior, tokenizer, token_ids, previous):
    text = tokenizer.decode(token_ids)
    tokens = words(text)
    return prior.score(text + " ", previous), tokens[-1] if tokens else previous


class GurbaniLanguagePrior:
    def __init__(self, lines):
        unigram, pairs = Counter(), defaultdict(Counter)
        # Identical corpus lines should not gain weight from database copies.
        for line in set(lines):
            tokens = words(line)
            unigram.update(tokens)
            for left, right in zip(tokens, tokens[1:]):
                pairs[left][right] += 1
        self.total = sum(unigram.values())
        self.prefixes = Counter()
        self.context_prefixes = {}
        self.context_totals = {}
        for word, count in unigram.items():
            for length in range(1, len(word) + 1):
                self.prefixes[word[:length]] += count
            self.prefixes[word + " "] += count
        for left, right_counts in pairs.items():
            counts = Counter()
            for word, count in right_counts.items():
                for length in range(1, len(word) + 1):
                    counts[word[:length]] += count
                counts[word + " "] += count
            self.context_prefixes[left] = counts
            self.context_totals[left] = sum(right_counts.values())

    @classmethod
    def from_database(cls, path: Path):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
            lines = [row[0] for row in database.execute(
                "SELECT DISTINCT unicode_speech FROM panktis WHERE unicode_speech IS NOT NULL")]
        return cls(lines)

    def evidence(self, previous, prefix, complete=False):
        if not previous or not prefix or not self.total:
            return 0.0
        key = prefix + " " if complete else prefix
        global_count = self.prefixes.get(key, 0)
        context_total = self.context_totals.get(previous, 0)
        if not global_count or not context_total:
            return 0.0
        base = global_count / self.total
        conditional = self.context_prefixes[previous].get(key, 0) / context_total
        # Back off for sparse contexts; even a well-known context leaves
        # substantial mass for ordinary speech and novel word combinations.
        strength = 0.75 * context_total / (context_total + 10)
        ratio = ((1 - strength) * base + strength * conditional) / base
        return min(3.0, max(-3.0, math.log(ratio)))

    @lru_cache(maxsize=16384)
    def score(self, text, previous=""):
        tokens = words(text)
        score = 0.0
        for index, token in enumerate(tokens):
            complete = index < len(tokens) - 1 or text.endswith(" ")
            score += self.evidence(previous, token, complete)
            previous = token
        return score
