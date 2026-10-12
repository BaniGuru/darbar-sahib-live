from dataclasses import dataclass
import unittest

from gurbani_orthography import GurbaniOrthography


@dataclass
class Word:
    text: str
    start: float
    end: float
    token_ids: list[int]


class Tokenizer:
    def encode(self, word, out_type=int):
        return list(range(len(word)))


class OrthographyTests(unittest.TestCase):
    def convert(self, text):
        return [Word(w, i, i + 1, [i]) for i, w in enumerate(text.split())]

    def test_exact_independent_context_restores_only_final_short_vowels(self):
        corpus = GurbaniOrthography(["ਜੇਤਾ ਸਮੁੰਦੁ ਸਾਗਰੁ ਨੀਰਿ ਭਰਿਆ"])
        original = self.convert("ਜੇਤਾ ਸਮੁੰਦ ਸਾਗਰ ਨੀਰ ਭਰਿਆ")
        corrected, changes = corpus.restore(original, Tokenizer())
        self.assertEqual(" ".join(w.text for w in corrected), "ਜੇਤਾ ਸਮੁੰਦੁ ਸਾਗਰੁ ਨੀਰਿ ਭਰਿਆ")
        self.assertEqual(len(changes), 3)
        self.assertEqual(len(corrected), len(original))
        self.assertEqual([(w.start, w.end) for w in corrected], [(w.start, w.end) for w in original])
        self.assertEqual(original[1].text, "ਸਮੁੰਦ")

    def test_consonants_pronounced_vowels_and_word_splits_are_untouched(self):
        corpus = GurbaniOrthography(["ਸ੍ਰੀ ਵਾਹਿਗੁਰੂ ਨਾਮੁ", "ਨੀਰਿ ਸਾਗਰੁ ਭਰਿਆ"])
        original = self.convert("ਸ੍ਰੀ ਵਾਰ ਗੁਰੂ ਨੀਰੁ ਸਾਗਰਾ ਭਰਿਆ")
        corrected, changes = corpus.restore(original, Tokenizer())
        self.assertEqual(corrected, original)
        self.assertEqual(changes, [])

    def test_ambiguous_and_unseen_contexts_are_not_forced(self):
        corpus = GurbaniOrthography(["ਹਰਿ ਨਾਮੁ ਜਪੋ", "ਹਰਿ ਨਾਮਿ ਜਪੋ"])
        original = self.convert("ਹਰਿ ਨਾਮ ਜਪੋ ਅਣਜਾਣ ਸ਼ਬਦ")
        corrected, changes = corpus.restore(original, Tokenizer())
        self.assertEqual(corrected, original)
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
