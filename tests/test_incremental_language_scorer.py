from pathlib import Path
import unittest
import numpy as np

import sentencepiece as spm

from gurbani_language_prior import GurbaniLanguagePrior, token_prefix_score
from incremental_language_scorer import IncrementalLanguageScorer, WordState
from kenlm_language_prior import KenLMWordPrior


class IncrementalScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.tokenizer = spm.SentencePieceProcessor(model_file=str(root / 'kirtan_stream_model/tokenizer.model'))
        cls.corpus = GurbaniLanguagePrior.from_database(root / 'data/bani.db')
        cls.lm = KenLMWordPrior(root / 'kirtan_stream_model/gurbani_3gram.bin', mode='ngram')

    def test_every_partial_and_next_piece_matches_original_full_prefix_scores(self):
        t = self.tokenizer
        samples = ['ਕਰਿ ਕਿਰਪਾ ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ ਤੇਰੀ ਓਟ ਪੂਰਨ ਗੋਪਾਲਾ',
                   'ਹਰਿ ਕਾ ਨਾਮੁ ਧਿਆਇ ਕੈ ਹੋਹੁ ਹਰਿਆ ਭਾਈ ਕਰਮਿ ਲਿਖੰਤੈ ਪਾਈਐ ਇਹ ਰੁਤਿ ਸੁਹਾਈ',
                   'ਪੰਜੇ ਬਧੇ ਮਹਾਬਲੀ ਕਰਿ ਸਚਾ ਢੋਆ ਆਪਣੇ ਚਰਣ ਜਪਾਇਅਨੁ ਵਿਚਿ ਦਯੁ ਖੜੋਆ',
                   'ਸਾਗਰੁ ੧ ਨਾਮ', 'ਵਾਹਿਗੁਰੂ ਵਾਰ ਗੁਰੂ', 'ਵਾਹਿਗੁਰੂ ੨',
                   'ਜੋਗੁ ਬਨਿਆ ਤੇਰਾ ਕੀਰਤਨੁ ਗਾਇਆ ਰਾਮ', 'ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ']
        for corpus, lm in [(self.corpus, self.lm), (self.corpus, None), (None, self.lm)]:
            for text in samples:
                for previous, context in [('', ()), ('ਸਾਗਰੁ', ('ਸਾਗਰੁ', 'ਨੀਰਿ'))]:
                    scorer = IncrementalLanguageScorer(t, corpus, lm, .6, .3, previous, context)
                    ids = tuple(t.encode(text) + [t.piece_to_id('▁')])
                    for end in range(len(ids) + 1):
                        parent = ids[:end]
                        parent_score = scorer.score(parent)
                        generic = (12 * .6 if corpus else 0) + (.3 * lm.max_token_reward if lm else 0)
                        bounds = scorer.upper_bounds(scorer.state(parent), np.arange(t.get_piece_size()), generic)
                        for label in range(t.get_piece_size()):
                            prefix = parent + (label,)
                            expected = .6 * token_prefix_score(corpus, t, prefix, previous) if corpus else 0
                            if lm:
                                expected += .3 * lm.token_score(t, prefix, context)
                            self.assertAlmostEqual(scorer.score(prefix), expected, places=12,
                                                   msg=(text, previous, prefix, t.decode(prefix)))
                            self.assertLessEqual(expected - parent_score, bounds[label],
                                                 msg=('unsafe pruning bound', text, prefix, t.decode(prefix)))

    def test_unknown_and_verse_number_fallback_matches_original(self):
        t = self.tokenizer
        for text in ['ਸਾਗਰੁ ੧ ਨਾਮ', 'ਵਾਹਿਗੁਰੂ ਵਾਰ ਗੁਰੂ', 'ਬੋਲੋ ਜੀ ਨਾਮੁ ਜਪੋ']:
            ids = t.encode(text)
            ids = tuple(ids[:3] + [0] + ids[3:])
            scorer = IncrementalLanguageScorer(t, self.corpus, self.lm, .6, .3, 'ਸਾਗਰੁ', ('ਸਾਗਰੁ',))
            expected = .6 * token_prefix_score(self.corpus, t, ids, 'ਸਾਗਰੁ') + .3 * self.lm.token_score(t, ids, ('ਸਾਗਰੁ',))
            self.assertEqual(scorer.score(ids), expected)

    def test_transition_cache_is_bounded_and_eviction_preserves_scores(self):
        scorer = IncrementalLanguageScorer(self.tokenizer, self.corpus, None, .6, 0)
        marker = self.tokenizer.piece_to_id('▁')
        first = WordState('ਨੀਰਿ', 'ਸਾਗਰੁ', 0, 0, (),
                          self.corpus.evidence('ਸਾਗਰੁ', 'ਨੀਰਿ'),
                          .6 * self.corpus.evidence('ਸਾਗਰੁ', 'ਨੀਰਿ'))
        expected = scorer.extend(first, marker).score
        for index in range(17000):
            parent = WordState(str(index) + 'ਅ', 'ਸਾਗਰੁ', 0, 0, (), 0, 0)
            scorer.extend(parent, marker)
        self.assertLessEqual(len(scorer.transitions), 16384)
        self.assertEqual(scorer.extend(first, marker).score, expected)

    def test_bounds_in_probability_and_ratio_modes(self):
        t = self.tokenizer
        for mode, bonus in [('probability', 4), ('ratio', 0), ('full_order', .25)]:
            lm = KenLMWordPrior(self.lm.path, mode=mode, word_bonus=bonus)
            scorer = IncrementalLanguageScorer(t, self.corpus, lm, .6, .3, 'ਹਰਿ', ('ਹਰਿ',))
            ids = tuple(t.encode('ਕਾ ਨਾਮੁ ਧਿਆਇ ਕੈ ਹੋਹੁ ਹਰਿਆ ਭਾਈ'))
            for end in range(len(ids) + 1):
                parent = ids[:end]
                value = scorer.score(parent)
                bounds = scorer.upper_bounds(scorer.state(parent), np.arange(t.get_piece_size()),
                                             .6 * 12 + .3 * lm.max_token_reward)
                for label in range(t.get_piece_size()):
                    candidate = parent + (label,)
                    expected = .6 * token_prefix_score(self.corpus, t, candidate, 'ਹਰਿ') + .3 * lm.token_score(t, candidate, ('ਹਰਿ',))
                    self.assertAlmostEqual(scorer.score(candidate), expected, places=12)
                    self.assertLessEqual(expected - value, bounds[label])
