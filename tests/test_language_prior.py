import unittest

from gurbani_language_prior import GurbaniLanguagePrior, words


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


if __name__ == "__main__":
    unittest.main()
