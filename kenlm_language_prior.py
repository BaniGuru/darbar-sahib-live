"""Cached word-level KenLM evidence for an open-vocabulary RNNT beam.

Only completed words advance the LM. An unfinished word remains acoustic and
mutable, including across encoder cache promotions. KenLM states are rebuilt
from at most order-1 words, never from the accumulated transcript.
"""
from functools import lru_cache
import math
from pathlib import Path

import kenlm

from gurbani_language_prior import words, _tokenizer_flags, token_prefix_parts


class KenLMWordPrior:
    def __init__(self, path: Path, *, mode="ratio", word_bonus=0.0):
        if mode not in {"ratio", "ngram", "trigram", "full_order", "probability"}:
            raise ValueError("KenLM mode must be ratio, ngram, trigram, full_order or probability")
        self.model = kenlm.Model(str(path))
        self.order = self.model.order
        if not 3 <= self.order <= 5:
            raise ValueError(f"Expected a 3-, 4- or 5-gram KenLM model, got order {self.order}")
        self.path = Path(path)
        self.mode = mode
        self.word_bonus = float(word_bonus)
        self.minimum_ngram_order = (self.order if mode == "full_order" else
                                    3 if mode in {"ngram", "trigram"} else 1)
        # A SentencePiece boundary can complete a word; the unknown-piece
        # fallback may complete two. This bounds its positive rescue score.
        self.max_token_reward = 2 * max(0.0, self.word_bonus + (0.0 if mode == "probability" else 3.0))

    def advance(self, context, tokens):
        return (tuple(context) + tuple(tokens))[-(self.order - 1):]

    @lru_cache(maxsize=4096)
    def state(self, context):
        state = kenlm.State()
        self.model.NullContextWrite(state)
        for token in context[-(self.order - 1):]:
            following = kenlm.State()
            self.model.BaseScore(state, token, following)
            state = following
        return state

    @lru_cache(maxsize=32768)
    def evidence(self, context, word):
        if self.mode != "probability" and (not context or word not in self.model):
            return self.word_bonus
        if self.mode in {"ngram", "trigram", "full_order"}:
            # Add genuinely higher-order evidence to the existing bigram
            # prior, rather than reinforcing its fallback a second time.
            minimum_order = self.minimum_ngram_order
            if len(context) < minimum_order - 1:
                return self.word_bonus
            result = self.model.BaseFullScore(self.state(context), word, kenlm.State())
            if result.ngram_length < minimum_order:
                return self.word_bonus
            conditional = result.log_prob * math.log(10)
        else:
            conditional = self.model.BaseScore(self.state(context), word, kenlm.State()) * math.log(10)
        if self.mode == "probability":
            # Bound unknown-word cost so commentary remains possible.
            return max(-12.0, conditional) + self.word_bonus
        unigram = self.model.BaseScore(self.state(()), word, kenlm.State()) * math.log(10)
        # Interpolate with the unigram and cap influence on weak acoustics.
        ratio = min(30.0, conditional - unigram)
        return min(3.0, math.log(0.2 + 0.8 * math.exp(ratio))) + self.word_bonus

    def full_score(self, text, context=()):
        """Reference path for punctuation, unknown pieces and verification."""
        tokens = words(text)
        complete = tokens if text.endswith(" ") else tokens[:-1]
        return self.complete_words(tuple(complete), tuple(context))[0]

    @lru_cache(maxsize=32768)
    def complete_words(self, tokens, context):
        total = 0.0
        for token in tokens:
            total += self.evidence(context, token)
            context = self.advance(context, (token,))
        return total, context

    @lru_cache(maxsize=32768)
    def complete_ids(self, tokenizer, token_ids, context):
        # Recursion shares the completed prefix and native state across beams.
        boundary = self._boundary(tokenizer, token_ids)
        if boundary:
            total, context = self.complete_ids(tokenizer, token_ids[:boundary], context)
            extra, context = self.complete_words(tuple(words(tokenizer.decode(token_ids[boundary:]))), context)
            return total + extra, context
        return self.complete_words(tuple(words(tokenizer.decode(token_ids))), context)

    @staticmethod
    def _boundary(tokenizer, token_ids):
        starts, _ = _tokenizer_flags(tokenizer)
        return next((index for index in range(len(token_ids) - 1, 0, -1)
                     if starts[token_ids[index]]), 0)

    @lru_cache(maxsize=32768)
    def token_score(self, tokenizer, token_ids, context=()):
        parts = token_prefix_parts(tokenizer, token_ids)
        if parts is None or not parts[0]:
            return self.full_score(tokenizer.decode(token_ids), context)
        boundary, _ = parts
        return self.complete_ids(tokenizer, token_ids[:boundary], context)[0]
