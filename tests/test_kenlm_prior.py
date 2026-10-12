import math
from pathlib import Path
import unittest

import sentencepiece as spm

from kenlm_language_prior import KenLMWordPrior


class KenLMPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "kirtan_stream_model"
        cls.prior = KenLMWordPrior(root / "gurbani_3gram.bin")
        cls.tokenizer = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))

    def test_three_gram_context_is_preserved(self):
        prior = self.prior
        self.assertEqual(prior.order, 3)
        self.assertGreater(prior.evidence(("ਸਾਗਰੁ", "ਨੀਰਿ"), "ਭਰਿਆ"),
                           prior.evidence(("ਸਾਗਰੁ", "ਨੀਰੁ"), "ਭਰਿਆ"))
        self.assertEqual(prior.advance(("ਹਰਿ", "ਸਾਗਰੁ"), ("ਨੀਰਿ",)), ("ਸਾਗਰੁ", "ਨੀਰਿ"))

    def test_open_vocabulary_and_partial_words(self):
        self.assertEqual(self.prior.evidence(("ਸਾਗਰੁ",), "ਅਣਜਾਣਸ਼ਬਦਜ਼ਜ਼ਜ਼"), 0)
        self.assertEqual(self.prior.full_score("ਵਾਹਿ", ("ਸਾਗਰੁ",)), 0)
        self.assertEqual(self.prior.full_score("ਵਾਹਿਗੁਰੂ", ("ਸਾਗਰੁ",)), 0)
        self.assertNotEqual(self.prior.full_score("ਵਾਹਿਗੁਰੂ ", ("ਸਾਗਰੁ",)), 0)

    def test_whole_vahiguru_is_preferred_to_the_split_phrase(self):
        for context in [(), ("ਸ੍ਰੀ",), ("ਬੋਲੋ", "ਜੀ")]:
            self.assertGreater(self.prior.full_score("ਵਾਹਿਗੁਰੂ ", context),
                               self.prior.full_score("ਵਾਰ ਗੁਰੂ ", context))

    def test_probability_path_matches_native_word_scores(self):
        p = KenLMWordPrior(self.prior.path, mode="probability", word_bonus=4)
        tokens = ("ਸਾਗਰੁ", "ਨੀਰਿ", "ਭਰਿਆ")
        actual = p.complete_words(tokens, ())[0]
        expected = sum(max(-12, logp * math.log(10)) + 4
                       for logp, _, _ in p.model.full_scores(" ".join(tokens), bos=False, eos=False))
        self.assertAlmostEqual(actual, expected)
        # Restoring a cached context must equal advancing the native state.
        self.assertAlmostEqual(p.evidence(tokens[:2], tokens[2]),
                               list(p.model.full_scores(" ".join(tokens), bos=False, eos=False))[-1][0] * math.log(10) + 4)

    def test_trigram_mode_does_not_duplicate_bigram_fallback(self):
        p = KenLMWordPrior(self.prior.path, mode="trigram")
        self.assertEqual(p.evidence(("ਸਾਗਰੁ",), "ਨੀਰਿ"), 0)
        self.assertGreater(p.evidence(("ਸਾਗਰੁ", "ਨੀਰਿ"), "ਭਰਿਆ"), 0)

    def test_incremental_scores_match_full_text_at_every_piece(self):
        t = self.tokenizer
        for mode, bonus in [("ratio", 0), ("trigram", 0.25), ("probability", 6)]:
            prior = KenLMWordPrior(self.prior.path, mode=mode, word_bonus=bonus)
            for text in ["ਸਾਗਰੁ ਨੀਰਿ ਭਰਿਆ", "ਵਾਹਿਗੁਰੂ ਵਾਰ ਗੁਰੂ", "ਬੋਲੋ ਜੀ ਸਾਗਰੁ ਨੀਰੁ",
                         "ਸਾਗਰੁ ੧", "ਸਾਗਰੁ ੧ ਨਾਮ", "ਸਾਗਰੁ ਨੀਰਿ ੨", "ਹਰਿ ਹਰਿ ਹਰਿ"]:
                ids = t.encode(text) + [t.piece_to_id("▁")]
                for end in range(len(ids) + 1):
                    for context in [(), ("ਸਾਗਰੁ",), ("ਸਾਗਰੁ", "ਨੀਰਿ")]:
                        prefix = tuple(ids[:end])
                        self.assertAlmostEqual(prior.token_score(t, prefix, context),
                                               prior.full_score(t.decode(prefix), context))
                unknown = tuple(ids[:3] + [0] + ids[3:])
                self.assertEqual(prior.token_score(t, unknown, ()), prior.full_score(t.decode(unknown)))

    def test_promotion_preserves_suffix_language_evidence(self):
        p = self.prior
        before = p.complete_words(("ਸਾਗਰੁ", "ਨੀਰਿ", "ਭਰਿਆ"), ())[0]
        prefix, context = p.complete_words(("ਸਾਗਰੁ", "ਨੀਰਿ"), ())
        after = p.complete_words(("ਭਰਿਆ",), context)[0]
        self.assertAlmostEqual(before, prefix + after)

    def test_four_and_five_gram_states_match_native_full_history(self):
        tokens = ("ਹਰਿ", "ਕਾ", "ਨਾਮੁ", "ਧਿਆਇ", "ਕੈ", "ਹੋਹੁ", "ਹਰਿਆ", "ਭਾਈ")
        for order in (4, 5):
            path = self.prior.path.with_name(f"gurbani_{order}gram.bin")
            p = KenLMWordPrior(path, mode="probability", word_bonus=4)
            native = list(p.model.full_scores(" ".join(tokens), bos=False, eos=False))
            for index, word in enumerate(tokens):
                context = p.advance((), tokens[:index])
                self.assertEqual(context, tokens[max(0, index - order + 1):index])
                self.assertAlmostEqual(p.evidence(context, word), max(-12, native[index][0] * math.log(10)) + 4)
            p = KenLMWordPrior(path, mode="ngram")
            ids = self.tokenizer.encode(" ".join(tokens)) + [self.tokenizer.piece_to_id("▁")]
            context = tokens[-(order - 1):]
            for end in range(len(ids) + 1):
                prefix = tuple(ids[:end])
                self.assertAlmostEqual(p.token_score(self.tokenizer, prefix, context),
                                       p.full_score(self.tokenizer.decode(prefix), context))


if __name__ == "__main__":
    unittest.main()
