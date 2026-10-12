import math
import unittest
from unittest.mock import patch

import numpy as np

from audio_activity import last_audio_sample
import server_asr_stream as asr
from test_streaming_asr import Tokenizer, encoder, frontend, hypothesis, scheduled_decoder


class ActivityTests(unittest.TestCase):
    def test_digital_silence_dc_and_quiet_broadband_noise(self):
        for audio in (np.zeros(12801, np.float32),
                      *[np.full(12801, value, np.float32) for value in (.001, .02, .1, .5, .8)],
                      np.random.default_rng(3).normal(0, .00047, 12801).astype(np.float32)):
            self.assertIsNone(last_audio_sample(audio))

    def test_very_quiet_tone_and_short_tail_are_preserved(self):
        tone = (1e-6 * np.sin(2 * np.pi * 300 * np.arange(320) / 16000)).astype(np.float32)
        self.assertEqual(last_audio_sample(tone), len(tone))
        self.assertEqual(last_audio_sample(np.concatenate((np.zeros(320), tone[:160]))), 480)

    def test_louder_broadband_audio_is_allowed(self):
        audio = np.random.default_rng(1).normal(0, .01, 640).astype(np.float32)
        self.assertEqual(last_audio_sample(audio), len(audio))


class SilenceStreamingTests(unittest.TestCase):
    def setUp(self):
        for name, value in [("tokenizer", Tokenizer()), ("preprocess_audio", frontend),
                            ("run_encoder", encoder), ("language_prior", None),
                            ("kenlm_prior", None), ("SILENCE_GUARD", True)]:
            patched = patch.object(asr, name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def test_idle_does_not_emit_tokens_but_keeps_samples_and_cache_clock(self):
        audio = np.random.default_rng(3).normal(0, .00047, 16000 * 65 + 13).astype(np.float32)
        with patch.object(asr, "decode_encoder_beams") as decoder:
            outputs = []
            for packet_size in (len(audio), 917):
                r = asr.StreamingRecognizer()
                for i in range(0, len(audio), packet_size):
                    state = r.update(audio[i:i + packet_size])
                    self.assertEqual(state['text'], '')
                outputs.append(r.finish())
            decoder.assert_not_called()
        self.assertEqual(outputs[0], outputs[1])
        self.assertFalse(outputs[0]['audio_active'])
        self.assertEqual(outputs[0]['samples'], len(audio))
        self.assertEqual(outputs[0]['encoded_frames'], math.ceil(len(audio) / 640))
        self.assertLessEqual(outputs[0]['peak_pending_window_seconds'], 8)

    def test_audio_after_idle_starts_on_original_clock_and_last_word_survives(self):
        tone = (.00001 * np.sin(2 * np.pi * 300 * np.arange(32000) / 16000)).astype(np.float32)
        token = asr.TimedToken(4, 205)  # after eight seconds of silence
        with patch.object(asr, "decode_encoder_beams", side_effect=scheduled_decoder([token])):
            r = asr.StreamingRecognizer()
            self.assertEqual(r.update(np.zeros(128000, np.float32))['text'], '')
            self.assertTrue(r.update(tone)['audio_active'])
            final = r.finish()
        self.assertEqual(final['text'], 'ਸੇਵਾ')
        self.assertEqual(final['words'][0]['start'], 8.2)
        self.assertEqual(final['samples'], 160000)

    def test_four_second_hangover_protects_pauses(self):
        r = asr.StreamingRecognizer()
        r.last_signal_sample = 16000
        r.pending_audio = np.zeros(12800, np.float32)
        r.total_samples = 16000 + 63999
        self.assertTrue(r._audio_decode_allowed())
        r.total_samples += 1
        self.assertFalse(r._audio_decode_allowed())

    def test_cache_reset_only_at_first_signal_after_initial_idle(self):
        tone = (.01 * np.sin(2 * np.pi * 300 * np.arange(12800) / 16000)).astype(np.float32)
        with patch.object(asr, 'new_encoder_cache', wraps=asr.new_encoder_cache) as caches, \
             patch.object(asr, 'decode_encoder_beams', return_value=[hypothesis()]):
            r = asr.StreamingRecognizer()
            r.update(np.zeros(128000, np.float32))
            self.assertEqual(caches.call_count, 1)
            r.update(tone)
            self.assertEqual(caches.call_count, 2)
            r.update(np.zeros(160000, np.float32))
            self.assertFalse(r._snapshot()['audio_active'])
            r.update(tone)
            self.assertEqual(caches.call_count, 2)

    def test_two_hours_idle_keeps_acoustic_memory_bounded(self):
        with patch.object(asr, "decode_encoder_beams") as decoder:
            r = asr.StreamingRecognizer()
            # The encoder is mocked; this checks stream clocks and bounded PCM,
            # rather than claiming two hours of native model validation.
            for _ in range(1800):
                r.update(np.zeros(64000, np.float32), preview=False)
            final = r.finish()
            decoder.assert_not_called()
        self.assertEqual(final['seconds'], 7200)
        self.assertEqual(final['encoded_frames'], 180000)
        self.assertEqual(final['text'], '')
        self.assertLessEqual(len(r.history_audio), 16 * 16000)
        self.assertLessEqual(final['peak_pending_window_seconds'], 8)


if __name__ == '__main__':
    unittest.main()
