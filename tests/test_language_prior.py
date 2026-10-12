import unittest
from pathlib import Path
import sentencepiece as spm

from gurbani_language_prior import GurbaniLanguagePrior, words, token_prefix_score


class LanguagePriorTests(unittest.TestCase):
    def setUp(self):
        self.prior = GurbaniLanguagePrior(["ਸਾਗਰੁ ਨੀਰਿ ਭਰਿਆ"] * 10
                                         + ["ਸਾਗਰੁ ਨੀਰਿ", "ਨੀਰੁ ਨਾਮ", "ਪਥਰ ਤਾਰੇ"])

    def test_context_prefers_attested_spelling_without_replacing_text(self):
        self.assertGreater(self.prior.score("ਸਾਗਰੁ ਨੀਰਿ"), self.prior.score("ਸਾਗਰੁ ਨੀਰੁ"))
        self.assertEqual(words("ਸਾਗਰੁ ਨੀਰੁ"), ["ਸਾਗਰੁ", "ਨੀਰੁ"])

    def test_unknown_speech_and_empty_context_are_allowed(self):
        self.assertEqual(self.prior.evidence("ਸਾਗਰੁ", "ਅਣਜਾਣਸ਼ਬਦ"), 0)
        self.assertEqual(self.prior.evidence("ਅਣਜਾਣਸ਼ਬਦ", "ਨੀਰਿ"), 0)
        self.assertEqual(self.prior.score(""), 0)
        self.assertTrue(-3 <= self.prior.evidence("ਸਾਗਰੁ", "ਨੀਰੁ") <= 3)

    def test_partial_prefix_keeps_the_whole_word_mutable(self):
        self.assertGreater(self.prior.evidence("ਪਥਰ", "ਤਾ"), 0)
        self.assertEqual(words("ਵਾਹਿਗੁਰੂ ਵਾਰ ਗੁਰੂ"), ["ਵਾਹਿਗੁਰੂ", "ਵਾਰ", "ਗੁਰੂ"])

    def test_duplicate_database_lines_do_not_change_evidence(self):
        unique = GurbaniLanguagePrior(["ਸਾਗਰੁ ਨੀਰਿ ਭਰਿਆ", "ਸਾਗਰੁ ਨੀਰਿ", "ਨੀਰੁ ਨਾਮ", "ਪਥਰ ਤਾਰੇ"])
        self.assertEqual(self.prior.score("ਸਾਗਰੁ ਨੀਰਿ"), unique.score("ਸਾਗਰੁ ਨੀਰਿ"))

    def test_incremental_prefix_scores_exactly_match_full_text_and_context(self):
        tokenizer = spm.SentencePieceProcessor(model_file=str(
            Path(__file__).resolve().parents[1] / "kirtan_stream_model" / "tokenizer.model"))
        marker = tokenizer.piece_to_id("▁")
        for text in ["ਸਾਗਰੁ ਨੀਰਿ ਭਰਿਆ", "ਸਾਗਰੁ ਨੀਰੁ ਨਾਮ", "ਪਥਰ ਤਾਰੇ",
                     "ਸਾਗਰੁ ੧", "ਸਾਗਰੁ ੧ ਨਾਮ", "ਸਾਗਰੁ ਨੀਰਿ ਪਥਰ ਤਾਰੇ"]:
            ids = tokenizer.encode(text) + [marker]
            for end in range(len(ids) + 1):
                for previous in ["", "ਸਾਗਰੁ", "ਨੀਰੁ"]:
                    prefix = tuple(ids[:end])
                    self.assertEqual(token_prefix_score(self.prior, tokenizer, prefix, previous),
                                     self.prior.score(tokenizer.decode(prefix), previous))
            unknown = tuple(ids[:3] + [0] + ids[3:])
            self.assertEqual(token_prefix_score(self.prior, tokenizer, unknown, "ਸਾਗਰੁ"),
                             self.prior.score(tokenizer.decode(unknown), "ਸਾਗਰੁ"))


if __name__ == "__main__":
    unittest.main()
