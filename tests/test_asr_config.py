import unittest
from unittest.mock import patch

import asr_config


class ConfigurationTests(unittest.TestCase):
    def test_single_pass_default_and_explicit_diagnostic_modes(self):
        with patch.dict(asr_config.os.environ, {}, clear=True):
            self.assertEqual(asr_config.configured_recognition(), ('blocks', 4))
        for mode, block in (('blocks', 2), ('rolling', 4)):
            with patch.dict(asr_config.os.environ, {'ASR_RECOGNITION_MODE': mode, 'ASR_BLOCK_SECONDS': str(block)}):
                self.assertEqual(asr_config.configured_recognition(), (mode, block))
        for values in ({'ASR_RECOGNITION_MODE': 'typo'}, {'ASR_BLOCK_SECONDS': '8'}):
            with patch.dict(asr_config.os.environ, values):
                with self.assertRaises(ValueError):
                    asr_config.configured_recognition()

    def test_encoder_pool_leaves_capacity_without_slowing_two_core_default(self):
        expected = {1: 1, 2: 2, 3: 2, 4: 2, 6: 3, 8: 4, 32: 4}
        for budget, threads in expected.items():
            self.assertEqual(asr_config.default_encoder_threads(budget), threads)
    def test_optimized_default_and_explicit_previous_decoder(self):
        with patch.dict(asr_config.os.environ, {}, clear=True):
            self.assertEqual(asr_config.configured_decoder_mode(), 'optimized')
        with patch.dict(asr_config.os.environ, {'ASR_DECODER_MODE': 'accuracy'}):
            self.assertEqual(asr_config.configured_decoder_mode(), 'accuracy')
        with patch.dict(asr_config.os.environ, {'ASR_DECODER_MODE': 'optimized'}):
            self.assertEqual(asr_config.configured_decoder_mode(), 'optimized')
        with patch.dict(asr_config.os.environ, {'ASR_DECODER_MODE': 'typo'}):
            with self.assertRaises(ValueError):
                asr_config.configured_decoder_mode()

    def test_editable_flag_and_environment_override(self):
        with patch.dict(asr_config.os.environ, {}, clear=True), \
             patch.object(asr_config, 'enable_language_model', False):
            self.assertFalse(asr_config.language_model_enabled())
            with patch.dict(asr_config.os.environ, {'ENABLE_LANGUAGE_MODEL': 'True'}):
                self.assertTrue(asr_config.language_model_enabled())
        for value in ('False', 'false', '0', 'off'):
            with patch.dict(asr_config.os.environ, {'ENABLE_LANGUAGE_MODEL': value}):
                self.assertFalse(asr_config.language_model_enabled())

    def test_invalid_switch_does_not_silently_change_accuracy(self):
        with patch.dict(asr_config.os.environ, {'ENABLE_LANGUAGE_MODEL': 'typo'}):
            with self.assertRaises(ValueError):
                asr_config.language_model_enabled()

    def budget(self, quota, affinity=8):
        with patch.object(asr_config.os, 'cpu_count', return_value=32), \
             patch.object(asr_config.os, 'sched_getaffinity', return_value=set(range(affinity))), \
             patch.object(asr_config.Path, 'read_text', return_value=quota):
            return asr_config.available_cpu_count()

    def test_quota_and_affinity_bound_thread_default(self):
        self.assertEqual(self.budget('200000 100000'), 2)
        self.assertEqual(self.budget('max 100000', 2), 2)
        self.assertEqual(self.budget('150000 100000'), 1)

    def test_v1_quota_and_unavailable_limits(self):
        with patch.object(asr_config.os, 'cpu_count', return_value=16), \
             patch.object(asr_config.os, 'sched_getaffinity', return_value={0, 1, 2, 3}), \
             patch.object(asr_config.Path, 'read_text', side_effect=[FileNotFoundError(), '200000', '100000']):
            self.assertEqual(asr_config.available_cpu_count(), 2)
        with patch.object(asr_config.os, 'cpu_count', return_value=None), \
             patch.object(asr_config.os, 'sched_getaffinity', side_effect=OSError()), \
             patch.object(asr_config.Path, 'read_text', side_effect=FileNotFoundError()):
            self.assertEqual(asr_config.available_cpu_count(), 1)
