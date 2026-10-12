"""Exact word-prior updates without decoding every candidate's full prefix."""
from dataclasses import dataclass
from collections import OrderedDict
import unicodedata
import numpy as np

from gurbani_language_prior import _tokenizer_flags, token_prefix_score


@dataclass(frozen=True, slots=True, eq=False)
class WordState:
    word: str
    previous: str
    corpus_completed: float
    lm_completed: float
    lm_context: tuple[str, ...]
    corpus_partial: float
    score: float


class IncrementalLanguageScorer:
    def __init__(self, tokenizer, corpus, kenlm, corpus_weight, kenlm_weight,
                 previous_word="", previous_words=()):
        self.tokenizer = tokenizer
        self.corpus, self.kenlm = corpus, kenlm
        self.corpus_weight, self.kenlm_weight = corpus_weight, kenlm_weight
        self.previous_word, self.previous_words = previous_word, previous_words
        flags = _tokenizer_flags(tokenizer)
        self.flags = flags
        self.pieces = ([tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())]
                       if flags is not None else None)
        if flags is not None:
            self.starts = np.asarray(flags[0])
            self.bound_safe = np.asarray([not unsafe and not piece.lstrip('▁').isnumeric()
                                          for unsafe, piece in zip(flags[1], self.pieces)])
        self.states = {(): WordState("", previous_word, 0.0, 0.0, previous_words, 0.0, 0.0)}
        self.fallback_scores = {}
        self.transitions = OrderedDict()

    def fallback(self, ids):
        if ids not in self.fallback_scores:
            value = (self.corpus_weight * token_prefix_score(
                self.corpus, self.tokenizer, ids, self.previous_word) if self.corpus else 0.0)
            if self.kenlm:
                value += self.kenlm_weight * self.kenlm.token_score(
                    self.tokenizer, ids, self.previous_words)
            self.fallback_scores[ids] = value
        return self.fallback_scores[ids]

    def extend(self, state, label):
        """None requests the unchanged full-prefix path for unsafe pieces."""
        if state is None or self.flags is None or self.flags[1][label]:
            return None
        key = (state, label)
        cached = self.transitions.get(key)
        if cached is not None:
            self.transitions.move_to_end(key)
            return cached
        piece = self.pieces[label]
        completed, lm_completed = state.corpus_completed, state.lm_completed
        previous, context = state.previous, state.lm_context
        if self.flags[0][label]:
            if state.word:
                if self.corpus:
                    completed += self.corpus.evidence(previous, state.word, complete=True)
                previous = state.word
                if self.kenlm:
                    lm_completed += self.kenlm.evidence(context, state.word)
                    context = self.kenlm.advance(context, (state.word,))
            word = piece[1:]
        else:
            word = state.word + piece
        word = unicodedata.normalize("NFC", word)
        # Dropping a trailing verse number can make the preceding word
        # partial again in the legacy scorer. Preserve that exact fallback.
        if word.isnumeric():
            return None
        partial = self.corpus.evidence(previous, word) if self.corpus and word else 0.0
        score = self.corpus_weight * (completed + partial) + self.kenlm_weight * lm_completed
        result = WordState(word, previous, completed, lm_completed, context, partial, score)
        self.transitions[key] = result
        if len(self.transitions) > 16384:
            self.transitions.popitem(last=False)
        return result

    def upper_bounds(self, state, labels, generic):
        """Conservative rescue bounds; continuation pieces cannot finish an LM word."""
        if state is None or self.flags is None:
            return np.full(len(labels), generic)
        within, boundary = 0.0, 0.0
        if self.corpus:
            maximum = 3.0 if self.corpus.context_totals.get(state.previous) else 0.0
            within = self.corpus_weight * (maximum - state.corpus_partial)
            completed = self.corpus.evidence(state.previous, state.word, complete=True) if state.word else 0.0
            next_context = state.word or state.previous
            maximum = 3.0 if self.corpus.context_totals.get(next_context) else 0.0
            boundary = self.corpus_weight * (completed - state.corpus_partial + maximum)
        if self.kenlm and state.word:
            boundary += self.kenlm_weight * self.kenlm.evidence(state.lm_context, state.word)
        bounds = np.where(self.starts[labels], boundary, within)
        bounds[~self.bound_safe[labels]] = generic
        # Keep candidates at a floating-point comparison boundary.
        return bounds + 1e-8

    def state(self, ids):
        if ids not in self.states:
            end = len(ids)
            while ids[:end] not in self.states:
                end -= 1
            current = self.states[ids[:end]]
            for index in range(end, len(ids)):
                current = self.extend(current, ids[index])
                self.states[ids[:index + 1]] = current
        return self.states[ids]

    def score(self, ids):
        state = self.state(ids)
        return state.score if state is not None else self.fallback(ids)
