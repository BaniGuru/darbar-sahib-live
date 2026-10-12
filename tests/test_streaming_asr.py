"""Regression checks for audio loss, cache boundaries, and end-of-stream flush."""
import asyncio
import math
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

import server_asr_stream as asr


class Tokenizer:
    pieces = {1: "▁ਪ੍ਰ", 2: "ਭ", 3: "▁ਦੀਨ", 4: "▁ਸੇਵਾ",
              5: "▁ਵ", 6: "ਾਰ", 7: "ਾ", 8: "ਹਿ", 9: "ਗ", 10: "ੁਰ", 11: "ੂ",
              12: "▁ਗ", 13: "▁ਨਾਮ", 14: "▁ਜਪੋ"}

    def id_to_piece(self, token):
        return self.pieces.get(token, "<unk>")

    def decode(self, tokens):
        return "".join(self.pieces[token].replace("▁", " ") for token in tokens).strip()


def frontend(audio):
    return np.zeros((1, 80, len(audio) // asr.HOP_LENGTH + 1), np.float32)


def encoder(features, cache):
    length = math.ceil(features.shape[2] / 4) - 2
    return np.zeros((1, 512, length), np.float32), length, cache


def hypothesis(tokens=(), score=0.0):
    states = tuple((*asr.new_predictor_state()[:2], token.token_id) for token in tokens)
    predictor = states[-1] if states else asr.new_predictor_state()
    return asr.DecoderHypothesis(tuple(tokens), tuple(t.token_id for t in tokens), predictor, score, states)


def scheduled_decoder(tokens):
    def decode(encoded, length, beams, frame_start, **kwargs):
        emitted = [token for token in tokens if frame_start <= token.encoder_frame < frame_start + length]
        return [hypothesis(list(beams[0].tokens) + emitted)]
    return decode


class StreamingTests(unittest.TestCase):
    def setUp(self):
        for name, value in [("tokenizer", Tokenizer()), ("preprocess_audio", frontend),
                            ("run_encoder", encoder), ("language_prior", None), ("kenlm_prior", None),
                            # These fixtures mock the acoustic model; zeros represent audible input.
                            ("last_audio_sample", lambda audio: len(audio))]:
            patched = patch.object(asr, name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def test_partial_is_visible_on_first_decode(self):
        token = asr.TimedToken(4, 10)
        with patch.object(asr, "decode_encoder_beams", return_value=[hypothesis([token])]):
            state = asr.StreamingRecognizer().update(np.zeros(12800, np.float32))
        self.assertEqual(state["partial_text"], "ਸੇਵਾ")

    def test_word_survives_and_completes_across_four_second_cache_commit(self):
        tokens = [asr.TimedToken(1, 99), asr.TimedToken(2, 102), asr.TimedToken(3, 112)]
        with patch.object(asr, "decode_encoder_beams", side_effect=scheduled_decoder(tokens)):
            recognizer = asr.StreamingRecognizer()
            boundary = recognizer.update(np.zeros(64000, np.float32))
            continued = recognizer.update(np.zeros(12800, np.float32))
            confirmed = recognizer.update(np.zeros(12800, np.float32))
            final = recognizer.finish()
        self.assertEqual(boundary["partial_text"], "ਪ੍ਰ")
        self.assertEqual(boundary["committed_until"], 0.0)
        self.assertEqual(continued["text"], "ਪ੍ਰਭ ਦੀਨ")
        self.assertEqual([word["text"] for word in continued["partial_words"]], ["ਪ੍ਰਭ", "ਦੀਨ"])
        self.assertEqual(confirmed["committed_text"], "ਪ੍ਰਭ")
        self.assertEqual(confirmed["cache_committed_until"], 4.0)
        self.assertEqual(final["text"], "ਪ੍ਰਭ ਦੀਨ")

    def test_packets_crossing_cache_boundary_preserve_every_sample(self):
        audio = np.arange(72013, dtype=np.float32)
        with patch.object(asr, "decode_encoder_beams", return_value=[hypothesis()]):
            recognizer = asr.StreamingRecognizer()
            recognizer.update(audio)
            np.testing.assert_array_equal(np.concatenate((recognizer.history_audio, recognizer.pending_audio)), audio)
            result = recognizer.finish()
        self.assertEqual(result["samples"], len(audio))
        self.assertEqual(result["encoded_frames"], math.ceil(len(audio) / 640))

    def test_finish_flushes_tail_is_idempotent_and_rejects_later_audio(self):
        with patch.object(asr, "decode_encoder_beams", return_value=[hypothesis()]):
            recognizer = asr.StreamingRecognizer()
            recognizer.update(np.zeros(64000, np.float32))
            final = recognizer.finish(np.zeros(10, np.float32))
            self.assertEqual(final, recognizer.finish())
            self.assertEqual(final["samples"], 64010)
            self.assertEqual(final["encoded_frames"], 101)
            self.assertEqual(final["partial_text"], "")
            with self.assertRaises(RuntimeError):
                recognizer.update(np.zeros(1, np.float32))

    def test_kenlm_scores_terminal_word_at_eof_without_extra_audio(self):
        class FinalWordPrior:
            def full_score(self, text, context):
                return 2.0 if text.strip() == "ਸੇਵਾ" else 0.0
            def token_score(self, tokenizer, ids, context):
                return 0.0
        wrong = hypothesis([asr.TimedToken(3, 0)])
        correct = hypothesis([asr.TimedToken(4, 0)], -0.1)
        with patch.object(asr, "kenlm_prior", FinalWordPrior()), \
             patch.object(asr, "KENLM_WEIGHT", 0.3), \
             patch.object(asr, "decode_encoder_beams", return_value=[wrong, correct]):
            recognizer = asr.StreamingRecognizer()
            recognizer.update(np.zeros(321, np.float32))
            final = recognizer.finish()
            self.assertEqual(final, recognizer.finish())
        self.assertEqual(final["text"], "ਸੇਵਾ")
        self.assertEqual(final["samples"], 321)

    def test_two_word_lm_context_survives_encoder_cache_promotion(self):
        tokens = [asr.TimedToken(1, 10), asr.TimedToken(2, 30), asr.TimedToken(3, 80),
                  asr.TimedToken(4, 120), asr.TimedToken(13, 170)]
        scheduled = scheduled_decoder(tokens)
        seen = []
        def decode(*args, **kwargs):
            seen.append(kwargs["previous_words"])
            return scheduled(*args, **kwargs)
        with patch.object(asr, "decode_encoder_beams", side_effect=decode):
            recognizer = asr.StreamingRecognizer()
            recognizer.update(np.zeros(128000, np.float32))
            final = recognizer.finish()
        self.assertIn(("ਪ੍ਰਭ", "ਦੀਨ"), seen)
        self.assertEqual(final["text"], "ਪ੍ਰਭ ਦੀਨ ਸੇਵਾ ਨਾਮ")
        self.assertEqual(final["samples"], 128000)

    def test_five_gram_keeps_four_words_across_encoder_cache_promotion(self):
        class FiveGramPrior:
            order = 5
            def full_score(self, text, context):
                return 0.0
            def token_score(self, tokenizer, ids, context):
                return 0.0
        tokens = [asr.TimedToken(1, 10), asr.TimedToken(2, 20), asr.TimedToken(3, 30),
                  asr.TimedToken(4, 50), asr.TimedToken(13, 80), asr.TimedToken(14, 120),
                  asr.TimedToken(4, 170)]
        scheduled = scheduled_decoder(tokens)
        seen = []
        def decode(*args, **kwargs):
            seen.append(kwargs["previous_words"])
            return scheduled(*args, **kwargs)
        with patch.object(asr, "kenlm_prior", FiveGramPrior()), \
             patch.object(asr, "decode_encoder_beams", side_effect=decode):
            recognizer = asr.StreamingRecognizer()
            recognizer.update(np.zeros(128000, np.float32))
            final = recognizer.finish()
        self.assertIn(("ਪ੍ਰਭ", "ਦੀਨ", "ਸੇਵਾ", "ਨਾਮ"), seen)
        self.assertEqual(final["text"], "ਪ੍ਰਭ ਦੀਨ ਸੇਵਾ ਨਾਮ ਜਪੋ ਸੇਵਾ")

    def test_websocket_finish_consumes_less_than_one_update_of_buffered_audio(self):
        async def run():
            socket = AsyncMock()
            socket.receive.side_effect = [
                {"type": "websocket.receive", "bytes": np.zeros(321, "<f4").tobytes()},
                {"type": "websocket.receive", "text": '{"type":"finish"}'},
            ]
            with patch.object(asr, "decode_encoder_beams", return_value=[hypothesis()]):
                await asr.websocket_transcribe(socket)
            final = socket.send_json.call_args_list[-1].args[0]
            self.assertEqual(final["type"], "final")
            self.assertEqual(final["samples"], 321)
            self.assertEqual(final["encoded_frames"], 1)
            socket.close.assert_awaited_once()
        asyncio.run(run())


    def test_wrong_boundary_word_can_be_replaced_after_cache_advance(self):
        wrong = hypothesis([asr.TimedToken(5, 94), asr.TimedToken(6, 98)])
        correct = hypothesis([asr.TimedToken(5, 94), asr.TimedToken(7, 95), asr.TimedToken(8, 99)], -1)
        completed = hypothesis(list(correct.tokens) + [asr.TimedToken(9, 104), asr.TimedToken(10, 108),
                                                     asr.TimedToken(11, 113), asr.TimedToken(13, 118)])
        def decode(encoded, length, beams, frame_start, **kwargs):
            if frame_start >= 100:
                return [completed]
            if frame_start + length >= 100:
                return [wrong, correct]
            return [hypothesis()]
        with patch.object(asr, "decode_encoder_beams", side_effect=decode):
            recognizer = asr.StreamingRecognizer()
            boundary = recognizer.update(np.zeros(64000, np.float32))
            continued = recognizer.update(np.zeros(12800, np.float32))
        self.assertEqual(boundary["partial_text"], "ਵਾਰ")
        self.assertEqual(boundary["committed_text"], "")
        self.assertEqual(boundary["committed_until"], 0)
        self.assertEqual(continued["text"], "ਵਾਹਿਗੁਰੂ ਨਾਮ")
        self.assertNotIn("ਵਾਰ", continued["text"])
        self.assertNotIn("ਵਾਹਿ ਗੁਰੂ", continued["text"])

    def test_disputed_word_boundary_does_not_finalize_a_false_split(self):
        wrong = hypothesis([asr.TimedToken(5, 50), asr.TimedToken(6, 60), asr.TimedToken(12, 70),
                            asr.TimedToken(10, 72), asr.TimedToken(11, 75)])
        correct = hypothesis([asr.TimedToken(5, 50), asr.TimedToken(7, 60), asr.TimedToken(8, 65),
                              asr.TimedToken(9, 70), asr.TimedToken(10, 72), asr.TimedToken(11, 75)], -1)
        self.assertEqual(asr.settled_word_prefix([wrong, correct], 100), 0)

    def test_boundary_word_finishing_at_six_seconds_waits_one_second_then_caches_four(self):
        tokens = [asr.TimedToken(5, 94), asr.TimedToken(7, 95), asr.TimedToken(8, 105),
                  asr.TimedToken(9, 125), asr.TimedToken(10, 140), asr.TimedToken(11, 149),
                  asr.TimedToken(13, 150), asr.TimedToken(14, 180)]
        with patch.object(asr, "decode_encoder_beams", side_effect=scheduled_decoder(tokens)):
            recognizer = asr.StreamingRecognizer()
            recognizer.update(np.zeros(64000, np.float32))
            early = recognizer.update(np.zeros(38400, np.float32))  # 6.4 s: only 0.4 s after word end.
            confirmed = recognizer.update(np.zeros(12800, np.float32))  # 7.2 s.
            self.assertEqual(recognizer.decoder_frame_offset, 150)
            self.assertEqual(recognizer.beams[0].predictor[2], 11)
            self.assertEqual(recognizer.pending_audio.size, 51200)  # Lookahead remains pending.
            recognizer.update(np.zeros(12800, np.float32))
            final = recognizer.finish(np.zeros(10, np.float32))
        self.assertEqual(early["cache_committed_until"], 0)
        self.assertEqual(confirmed["cache_committed_until"], 4)
        self.assertEqual(confirmed["text_committed_until"], 6)
        self.assertEqual(confirmed["committed_text"], "ਵਾਹਿਗੁਰੂ")
        self.assertEqual(final["text"], "ਵਾਹਿਗੁਰੂ ਨਾਮ ਜਪੋ")
        self.assertEqual(final["samples"], 128010)
        self.assertEqual(final["encoded_frames"], 201)
        self.assertEqual(final["cache_events"][0]["at_seconds"], 7.2)

    def test_eight_second_limit_advances_cache_without_finalizing_the_unfinished_word(self):
        tokens = [asr.TimedToken(5, 94), asr.TimedToken(7, 95), asr.TimedToken(8, 105),
                  asr.TimedToken(9, 190), asr.TimedToken(10, 210), asr.TimedToken(11, 220),
                  asr.TimedToken(13, 225)]
        with patch.object(asr, "decode_encoder_beams", side_effect=scheduled_decoder(tokens)):
            recognizer = asr.StreamingRecognizer()
            capped = recognizer.update(np.zeros(128000, np.float32))
            self.assertEqual(recognizer.pending_audio.size, 64000)
            recognizer.update(np.zeros(38400, np.float32))  # 10.4 s: one second beyond word end.
            final = recognizer.finish()
        self.assertEqual(capped["cache_committed_until"], 4)
        self.assertEqual(capped["committed_text"], "")
        self.assertEqual(capped["partial_text"], "ਵਾਹਿਗ")
        self.assertEqual(final["text"], "ਵਾਹਿਗੁਰੂ ਨਾਮ")
        self.assertEqual(final["peak_pending_window_seconds"], 8)
        self.assertEqual(final["cache_events"][0]["reason"], "window-limit")

    def test_word_finishing_at_seven_seconds_can_confirm_at_the_eight_second_limit(self):
        tokens = [asr.TimedToken(5, 94), asr.TimedToken(7, 95), asr.TimedToken(8, 105),
                  asr.TimedToken(9, 140), asr.TimedToken(10, 160), asr.TimedToken(11, 174),
                  asr.TimedToken(13, 175)]
        with patch.object(asr, "decode_encoder_beams", side_effect=scheduled_decoder(tokens)):
            recognizer = asr.StreamingRecognizer()
            result = recognizer.update(np.zeros(128000, np.float32))
        self.assertEqual(result["committed_text"], "ਵਾਹਿਗੁਰੂ")
        self.assertEqual(result["cache_committed_until"], 4)
        self.assertEqual(recognizer.cache_events[0]["at_seconds"], 8)
        self.assertEqual(recognizer.cache_events[0]["word_end_seconds"], 7)
        self.assertEqual(recognizer.cache_events[0]["reason"], "word-confirmed")

    def test_packet_sizes_do_not_change_adaptive_cache_decisions(self):
        tokens = [asr.TimedToken(1, 94), asr.TimedToken(2, 125), asr.TimedToken(3, 150)]
        audio = np.zeros(144013, np.float32)
        with patch.object(asr, "decode_encoder_beams", side_effect=scheduled_decoder(tokens)):
            batch = asr.StreamingRecognizer()
            batch.update(audio, preview=False)
            full = batch.finish()
            live = asr.StreamingRecognizer()
            for start in range(0, len(audio), 317):
                live.update(audio[start:start + 317])
            streamed = live.finish()
        self.assertEqual(full, streamed)
        self.assertLessEqual(full["peak_pending_window_seconds"], 8)

    def test_complete_word_needs_next_boundary_and_acoustic_lookahead(self):
        word = [asr.TimedToken(1, 50), asr.TimedToken(2, 75)]
        self.assertEqual(asr.settled_word_prefix([hypothesis(word)], 200), 0)
        completed = word + [asr.TimedToken(3, 90)]
        self.assertEqual(asr.settled_word_prefix([hypothesis(completed)], 100), 0)
        self.assertEqual(asr.settled_word_prefix([hypothesis(completed)], 120), 2)

    def test_spelling_alternatives_do_not_stall_a_confirmed_word_boundary(self):
        best = hypothesis([asr.TimedToken(1, 50), asr.TimedToken(2, 75), asr.TimedToken(3, 90)])
        alternative = hypothesis([asr.TimedToken(1, 50), asr.TimedToken(6, 75), asr.TimedToken(3, 92)], -1)
        self.assertEqual(asr.settled_word_prefix([best, alternative], 120), 2)

    def test_short_word_prefix_cannot_commit_while_a_longer_word_continues(self):
        short = hypothesis([asr.TimedToken(12, 10), asr.TimedToken(10, 15), asr.TimedToken(13, 60)])
        long = hypothesis([asr.TimedToken(12, 10), asr.TimedToken(10, 15),
                           asr.TimedToken(11, 20), asr.TimedToken(13, 61)], -0.1)
        self.assertEqual(asr.settled_word_prefix([short, long], 100), 0)

    def test_hard_cache_limit_preserves_whole_prefix_word_alternatives(self):
        short = hypothesis([asr.TimedToken(12, 10), asr.TimedToken(10, 15), asr.TimedToken(13, 60)])
        long = hypothesis([asr.TimedToken(12, 10), asr.TimedToken(10, 15),
                           asr.TimedToken(11, 20), asr.TimedToken(13, 61)], -0.1)
        finishing = False
        def decode(encoded, length, beams, frame_start, **kwargs):
            if frame_start < 100:
                return [short, long]
            self.assertEqual(beams[1].token_ids[:3], (12, 10, 11))
            return [long, short] if finishing else [short, long]
        with patch.object(asr, "decode_encoder_beams", side_effect=decode):
            r = asr.StreamingRecognizer()
            state = r.update(np.zeros(128000, np.float32))
            finishing = True
            final = r.finish()
        self.assertEqual(state["committed_text"], "")
        self.assertEqual(state["cache_committed_until"], 4)
        self.assertEqual(final["text"], "ਗੁਰੂ ਨਾਮ")
        self.assertEqual(final["peak_pending_window_seconds"], 8)

    def test_beam_blank_preserves_predictor_state_without_mutating_input(self):
        initial = asr.new_decoder_beams()
        logits = np.full((1, 1, 1, asr.LOCAL_VOCAB_SIZE + 1), -30, np.float32)
        logits[0, 0, 0, asr.JOINT_BLANK_ID] = 0
        output = [logits, np.ones(1, np.int32), np.ones((1, 1, 640), np.float32),
                  np.ones((1, 1, 640), np.float32)]
        with patch.object(asr.decoder_session, "run", return_value=output):
            result = asr.decode_encoder_beams(np.zeros((1, 512, 1), np.float32), 1, initial, 0)
        self.assertEqual(result[0].token_ids, ())
        self.assertIs(result[0].predictor, initial[0].predictor)
        self.assertEqual(initial[0].score, 0)

    def test_context_prior_reranks_close_acoustic_spellings_and_does_not_generate_from_blank(self):
        from gurbani_language_prior import GurbaniLanguagePrior
        class ContextTokenizer(Tokenizer):
            pieces = {0: "<unk>", 1: "▁ਨੀਰੁ", 2: "▁ਨੀਰਿ", 3: "▁ਨਾਮ", 4: "▁ਗੁਰੂ"}
        prior = GurbaniLanguagePrior(["ਸਾਗਰੁ ਨੀਰਿ", "ਨੀਰੁ ਨਾਮ"])
        blank_only = False
        def joint(_, inputs):
            labels = inputs[asr.decoder_inputs[1]][:, 0]
            logits = np.full((len(labels), 1, 1, 6), -30, np.float32)
            for index, label in enumerate(labels):
                if label == asr.PREDICTOR_START_ID and not blank_only:
                    logits[index, 0, 0, 1] = np.log(.51)
                    logits[index, 0, 0, 2] = np.log(.49)
                else:
                    logits[index, 0, 0, 5] = 0
            return [logits, np.ones(len(labels), np.int32),
                    inputs[asr.decoder_inputs[3]].copy(), inputs[asr.decoder_inputs[4]].copy()]
        with patch.object(asr, "tokenizer", ContextTokenizer()), patch.object(asr, "language_prior", prior), \
             patch.object(asr, "LANGUAGE_PRIOR_WEIGHT", 0.6), patch.object(asr, "LOCAL_VOCAB_SIZE", 5), \
             patch.object(asr, "JOINT_BLANK_ID", 5), patch.object(asr.decoder_session, "run", side_effect=joint):
            encoded = np.zeros((1, 512, 1), np.float32)
            chosen = asr.decode_encoder_beams(encoded, 1, asr.new_decoder_beams(), 0, previous_word="ਸਾਗਰੁ")
            self.assertEqual(chosen[0].token_ids, (2,))
            blank_only = True
            chosen = asr.decode_encoder_beams(encoded, 1, asr.new_decoder_beams(), 0, previous_word="ਸਾਗਰੁ")
            self.assertEqual(chosen[0].token_ids, ())

    def test_settled_word_cannot_grow_a_new_suffix_when_lookahead_is_replayed(self):
        def joint(_, inputs):
            labels = inputs[asr.decoder_inputs[1]][:, 0]
            logits = np.full((len(labels), 1, 1, asr.LOCAL_VOCAB_SIZE + 1), -30, np.float32)
            for index, label in enumerate(labels):
                if label == asr.PREDICTOR_START_ID:
                    logits[index, 0, 0, 2] = 0  # Continuation piece would change the settled word.
                    logits[index, 0, 0, 4] = -1  # A valid new word.
                    logits[index, 0, 0, asr.JOINT_BLANK_ID] = -5
                else:
                    logits[index, 0, 0, asr.JOINT_BLANK_ID] = 0
            return [logits, np.ones(len(labels), np.int32),
                    inputs[asr.decoder_inputs[3]].copy(), inputs[asr.decoder_inputs[4]].copy()]
        with patch.object(asr.decoder_session, "run", side_effect=joint):
            result = asr.decode_encoder_beams(np.zeros((1, 512, 1), np.float32), 1,
                                              asr.new_decoder_beams(), 150, require_word_start=True)
        self.assertEqual(result[0].token_ids, (4,))

    def test_later_acoustic_evidence_can_promote_an_alternative_from_prior_block(self):
        def joint(_, inputs):
            labels = inputs[asr.decoder_inputs[1]][:, 0]
            frames = inputs[asr.decoder_inputs[0]][:, 0, 0]
            logits = np.full((len(labels), 1, 1, asr.LOCAL_VOCAB_SIZE + 1), -30, np.float32)
            for index, (label, frame) in enumerate(zip(labels, frames)):
                probs = {asr.JOINT_BLANK_ID: 1.0}
                if frame == 0 and label == asr.PREDICTOR_START_ID:
                    probs = {6: 0.55, 7: 0.45}
                elif frame == 1 and label == 6:
                    probs = {asr.JOINT_BLANK_ID: 0.7, 9: 0.3}
                elif frame == 1 and label == 7:
                    probs = {8: 0.99, asr.JOINT_BLANK_ID: 0.01}
                for token, probability in probs.items():
                    logits[index, 0, 0, token] = np.log(probability)
            return [logits, np.ones(len(labels), np.int32),
                    inputs[asr.decoder_inputs[3]].copy(), inputs[asr.decoder_inputs[4]].copy()]

        with patch.object(asr.decoder_session, "run", side_effect=joint):
            before = asr.decode_encoder_beams(np.zeros((1, 512, 1), np.float32), 1, asr.new_decoder_beams(), 99)
            after = asr.decode_encoder_beams(np.ones((1, 512, 1), np.float32), 1, before, 100)
        self.assertEqual(before[0].token_ids, (6,))
        self.assertEqual(after[0].token_ids, (7, 8))
        self.assertEqual([token.encoder_frame for token in after[0].tokens], [99, 100])


class ExportTests(unittest.TestCase):
    def test_cached_display_matches_full_tokenizer_and_keeps_prior_snapshots_unchanged(self):
        ids = asr.tokenizer.encode("ਸ੍ਰੀ ਵਾਹਿਗੁਰੂ ਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ", out_type=int)
        tokens = [asr.TimedToken(label, index * 5) for index, label in enumerate(ids)]
        starts = [i for i, token in enumerate(tokens)
                  if asr.tokenizer.id_to_piece(token.token_id).startswith("▁") and i]
        recognizer = asr.StreamingRecognizer()
        retained = None
        for split in starts:
            recognizer.committed_tokens = tokens[:split]
            recognizer.preview_tokens = tokens[split:]
            state = recognizer._snapshot()
            self.assertEqual(state["text"], asr.tokenizer.decode(ids).strip())
            self.assertEqual(state["committed_words"] + state["partial_words"], asr.tokens_to_words(tokens))
            if retained:
                previous, frozen = retained
                self.assertEqual(previous, frozen)
            import copy
            retained = (state, copy.deepcopy(state))

    def test_frontend_overlap_recovers_all_hundred_frames_per_cache_block(self):
        _, length, _ = asr.run_encoder(np.zeros((1, 80, 408), np.float32), asr.new_encoder_cache())
        self.assertEqual(length, 100)

    def test_single_sample_frontend_is_finite(self):
        self.assertTrue(np.isfinite(asr.preprocess_audio(np.zeros(1, np.float32))).all())


if __name__ == "__main__":
    unittest.main()
