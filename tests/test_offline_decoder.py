import unittest

import numpy as np

from offline_rnnt_decoder import decode_beam
from gurbani_language_prior import GurbaniLanguagePrior


class Tokenizer:
    def decode(self, ids):
        return " ".join({1: "ਨੀਰੁ", 2: "ਨੀਰਿ", 3: "ਨਾਮ", 4: "ਗੁਰੂ"}.get(i, "") for i in ids)


class Session:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = []

    def run(self, _, inputs):
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        labels = inputs["labels"][:, 0]
        logits = np.full((len(labels), 1, 1, 6), -30, np.float32)
        for index, label in enumerate(labels):
            for target, probability in self.probabilities(int(label), float(inputs["encoded"][index, 0, 0])).items():
                logits[index, 0, 0, target] = np.log(probability)
        return [logits, np.ones(len(labels), np.int32),
                inputs["h"] + 1, inputs["c"] + 1]


class DecoderTests(unittest.TestCase):
    def run_decoder(self, encoded, session, prior=None):
        return decode_beam(encoded, encoded.shape[-1], session,
                           ["encoded", "labels", "length", "h", "c"], Tokenizer(),
                           5, 5, 5632, 5, 4, prior, .6)

    def test_blank_keeps_predictor_state_and_cannot_create_words(self):
        session = Session(lambda label, frame: {5: 1})
        self.assertEqual(self.run_decoder(np.zeros((1, 512, 2), np.float32), session), [])
        self.assertEqual(len(session.calls), 2)
        self.assertTrue(all(np.all(call["h"] == 0) for call in session.calls))

    def test_later_audio_can_promote_an_alternative_and_feedback_stays_local(self):
        def probabilities(label, frame):
            if frame == 0 and label == 5632: return {1: .55, 2: .45}
            if frame == 1 and label == 1: return {5: .7, 3: .3}
            if frame == 1 and label == 2: return {4: .99, 5: .01}
            return {5: 1}
        session = Session(probabilities)
        encoded = np.zeros((1, 512, 2), np.float32);encoded[:, :, 1] = 1
        self.assertEqual(self.run_decoder(encoded, session), [(2, 0), (4, 1)])
        self.assertTrue(all(label == 5632 or 0 <= label < 5
                            for call in session.calls for label in call["labels"][:, 0]))

    def test_independent_prior_can_rerank_close_spelling_choices(self):
        session = Session(lambda label, frame: {1: .51, 2: .49} if label == 5632 else {5: 1})
        prior = GurbaniLanguagePrior(["ਸਾਗਰੁ ਨੀਰਿ", "ਨੀਰੁ ਨਾਮ"])
        # Supply the preceding word through the decoded prefix, as a real
        # offline window does, rather than supplying reference text.
        class ContextPrior:
            def score(self, text): return prior.score(text, previous="ਸਾਗਰੁ")
        self.assertEqual(self.run_decoder(np.zeros((1, 512, 1), np.float32), session, ContextPrior()), [(2, 0)])


if __name__ == "__main__":
    unittest.main()
