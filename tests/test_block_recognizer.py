"""Single-pass blocks preserve lexical hypotheses and arbitrary PCM tails."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import server_asr_stream as asr
from block_recognizer import BlockRecognizer
import test_streaming_asr as fixtures


class BlockTests(unittest.TestCase):
    def setUp(self):
        fixtures.StreamingTests.setUp(self)

    def test_long_word_crosses_cache_without_being_split(self):
        tokens = [asr.TimedToken(5, 94), asr.TimedToken(7, 105),
                  asr.TimedToken(8, 125), asr.TimedToken(9, 130),
                  asr.TimedToken(10, 140), asr.TimedToken(11, 150),
                  asr.TimedToken(14, 175)]
        for seconds in (2, 4):
            with self.subTest(block=seconds), patch.object(asr, 'decode_encoder_beams',
                                                          side_effect=fixtures.scheduled_decoder(tokens)):
                recognizer = BlockRecognizer(seconds)
                # 8s received lets the 4s acoustic frontier advance; lexical
                # completion has not yet been heard and must remain open.
                first = recognizer.update(np.zeros(128000, np.float32))
                self.assertNotIn('ਵਾਹਿਗੁਰੂ', first['committed_text'])
                recognizer.update(np.zeros(64000, np.float32))
                self.assertEqual(recognizer._snapshot()['committed_text'], 'ਵਾਹਿਗੁਰੂ')
                final = recognizer.finish()
                self.assertEqual(final['text'], 'ਵਾਹਿਗੁਰੂ ਜਪੋ')
                self.assertEqual(final['acoustic_reencodes'], 0)
                self.assertEqual(final['encoder_calls'], 12 // seconds)

    def test_packetization_tail_and_single_pass_encoder_coverage(self):
        audio = np.arange(192013, dtype=np.float32)
        for seconds in (2, 4):
            outputs = []
            for packets in ([audio], [audio[i:i+1237] for i in range(0, len(audio), 1237)]):
                with patch.object(asr, 'decode_encoder_beams', return_value=[fixtures.hypothesis()]), \
                     patch.object(asr, 'run_encoder', wraps=fixtures.encoder) as encode:
                    recognizer = BlockRecognizer(seconds)
                    for packet in packets:
                        recognizer.update(packet)
                    final = recognizer.finish()
                    self.assertEqual(final['samples'], len(audio))
                    self.assertEqual(final['encoded_frames'], math.ceil(len(audio)/640))
                    self.assertEqual(encode.call_count, math.ceil(len(audio)/(16000*seconds)))
                    self.assertLessEqual(final['peak_pending_window_seconds'], seconds+4)
                    np.testing.assert_array_equal(recognizer.history_audio, audio)
                    self.assertEqual(final, recognizer.finish())
                    outputs.append(final)
            self.assertEqual(outputs[0], outputs[1])

    def test_future_voice_cannot_make_idle_current_block_decode(self):
        with patch.object(asr, 'last_audio_sample', side_effect=lambda a: len(a) if np.any(a) else None), \
             patch.object(asr, 'decode_encoder_beams', return_value=[fixtures.hypothesis()]) as decode:
            recognizer = BlockRecognizer(4)
            recognizer.update(np.concatenate((np.zeros(64000), np.ones(64000))))
            decode.assert_not_called()
            recognizer.update(np.ones(64000))
            self.assertEqual(decode.call_count, 1)

