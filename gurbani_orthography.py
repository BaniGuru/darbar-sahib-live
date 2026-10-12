"""Restore only silent final ੁ/ਿ with exact independent corpus context.

No transcript reference is consulted, no words are inserted or removed, and
pronounced vowels/consonants are never rewritten. Raw ASR remains available.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import sqlite3

from gurbani_language_prior import words


def stem(word):
    return word.rstrip("ਿੁ")


class GurbaniOrthography:
    def __init__(self, lines):
        self.contexts = defaultdict(Counter)
        for line in set(lines):
            tokens = words(line)
            for index in range(len(tokens) - 2):
                phrase = tuple(tokens[index:index + 3])
                key = tuple(stem(word) for word in phrase)
                self.contexts[key][phrase] += 1

    @classmethod
    def from_database(cls, path: Path):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            lines = [row[0] for row in connection.execute(
                "SELECT DISTINCT unicode_speech FROM panktis WHERE unicode_speech IS NOT NULL")]
        return cls(lines)

    def restore(self, timed_words, tokenizer):
        raw = [word.text for word in timed_words]
        votes = defaultdict(Counter)
        for index in range(len(raw) - 2):
            key = tuple(stem(word) for word in raw[index:index + 3])
            candidates = self.contexts.get(key)
            if not candidates:
                continue
            for phrase, count in candidates.items():
                for position, word in enumerate(phrase):
                    votes[index + position][word] += count
        restored = list(timed_words)
        changes = []
        for index, candidates in votes.items():
            target, count = candidates.most_common(1)[0]
            confidence = count / sum(candidates.values())
            if confidence < .8 or target == raw[index]:
                continue
            assert stem(target) == stem(raw[index])
            restored[index] = replace(timed_words[index], text=target,
                                      token_ids=tokenizer.encode(target, out_type=int))
            changes.append({"word_index": index, "raw": raw[index], "text": target,
                            "confidence": confidence, "evidence": "exact three-word corpus stem context"})
        return restored, changes
