"""Verify factored inference against the unchanged exported ONNX graph."""
import unittest
import numpy as np
import server_asr_stream as asr
from cached_rnnt_decoder import CachedRNNTDecoder


class CachedDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = getattr(asr.decoder_session, 'original', asr.decoder_session)
        cls.decoder = CachedRNNTDecoder(cls.original, str(asr.MODEL_DIR / asr.RUNTIME['decoder_joint']),
                                       asr.session_options_for(1), cache_size=2)

    def inputs(self, batch):
        rng = np.random.default_rng(9)
        names = asr.decoder_inputs
        return {names[0]: np.repeat(rng.normal(size=(1, 512, 1)).astype(np.float32), batch, axis=0),
                names[1]: np.arange(batch, dtype=np.int32)[:, None],
                names[2]: np.ones(batch, np.int32),
                names[3]: rng.normal(size=(1, batch, 640)).astype(np.float32),
                names[4]: rng.normal(size=(1, batch, 640)).astype(np.float32)}

    def assert_equivalent(self, inputs):
        for original, cached in zip(self.original.run(None, inputs), self.decoder.run(None, inputs)):
            np.testing.assert_array_equal(original, cached)

    def test_dynamic_beams_match_original_logits_and_recurrent_states(self):
        for batch in (1, 4, 8, 16, 24):
            inputs = self.inputs(batch)
            self.assert_equivalent(inputs)
            inputs[asr.decoder_inputs[0]][:, 0, 0] += 0.75
            self.assert_equivalent(inputs)
            inputs[asr.decoder_inputs[3]][:, 0, 0] += 0.5
            self.assert_equivalent(inputs)
            self.assertLessEqual(len(self.decoder.predictor_cache), 2)

    def test_acoustic_cache_is_bounded_and_evicted_frames_recompute_correctly(self):
        inputs = self.inputs(1)
        for frame in range(140):
            inputs[asr.decoder_inputs[0]][:, 0, 0] = frame / 100
            self.decoder.run(None, inputs)
        self.assertEqual(len(self.decoder.encoder_cache), 128)
        inputs[asr.decoder_inputs[0]][:, 0, 0] = 0
        self.assert_equivalent(inputs)

    def test_start_blank_independent_encoder_frames_and_selected_outputs(self):
        inputs = self.inputs(4)
        inputs[asr.decoder_inputs[1]][0, 0] = asr.PREDICTOR_START_ID
        inputs[asr.decoder_inputs[0]][1, 0, 0] += 0.2
        self.assert_equivalent(inputs)
        name = self.original.get_outputs()[2].name
        np.testing.assert_array_equal(self.original.run([name], inputs)[0], self.decoder.run([name], inputs)[0])

    def test_mixed_predictor_hits_and_misses_match_original_batch(self):
        inputs = self.inputs(8)
        names = asr.decoder_inputs
        warm = {name: value[:, :2] if name in names[3:] else value[:2]
                for name, value in inputs.items()}
        self.decoder.run(None, warm)
        self.assert_equivalent(inputs)


if __name__ == '__main__':
    unittest.main()
