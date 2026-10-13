"""Single-pass acoustic blocks; unfinished lexical hypotheses cross caches."""
import numpy as np

import server_asr_stream as asr


class BlockRecognizer(asr.StreamingRecognizer):
    def __init__(self, block_seconds=4):
        super().__init__()
        if block_seconds not in (2, 4):
            raise ValueError('Recognition blocks must be 2 or 4 seconds')
        self.block_samples = int(block_seconds * asr.SAMPLE_RATE)
        self.lookahead_samples = asr.FEATURE_CHUNK_SAMPLES
        self.encoder_calls = 0

    def _commit_prefix(self, beams, count):
        if not count:
            self.beams = beams
            return
        prefix = beams[0].token_ids[:count]
        tokens = beams[0].tokens[:count]
        self.committed_tokens.extend(tokens)
        words = asr.prior_words(asr.tokenizer.decode(list(prefix)))
        if words:
            self.committed_last_word = words[-1]
            context = asr.kenlm_prior.order - 1 if asr.kenlm_prior else 2
            self.committed_lm_context = (self.committed_lm_context + tuple(words))[-context:]
        self.beams = [asr.DecoderHypothesis(h.tokens[count:], h.token_ids[count:], h.predictor,
                                         h.score, h.predictor_states[count:])
                      for h in beams if h.token_ids[:count] == prefix]

    def _consume(self, samples):
        block = self.pending_audio[:samples]
        received = self.total_samples
        full_pending = self.pending_audio
        initially_idle = self.last_signal_sample is None and self.silence_suppressed
        # Future audio informs normalization, but must not make silence in
        # the current block count as voice or shorten the signal hangover.
        self.pending_audio = block
        self.total_samples = round(self.encoder_frame_offset * asr.ENCODER_FRAME_SECONDS * asr.SAMPLE_RATE) + samples
        allowed = self._audio_decode_allowed()
        self.pending_audio, self.total_samples = full_pending, received
        if initially_idle and allowed:
            self.cache = asr.new_encoder_cache()

        features = asr.preprocess_audio(np.concatenate((self.history_audio, full_pending)))
        history_frames = self.history_audio.size // asr.HOP_LENGTH
        if history_frames < asr.PREENCODE_CONTEXT_FRAMES:
            pad = asr.PREENCODE_CONTEXT_FRAMES - history_frames
            features = np.pad(features, ((0, 0), (0, 0), (pad, 0)), mode='edge')
            history_frames += pad
        count = int(np.ceil(samples / asr.HOP_LENGTH))
        features = features[:, :, history_frames - asr.PREENCODE_CONTEXT_FRAMES:history_frames + count]
        encoded, length, cache = asr.run_encoder(features, self.cache)
        valid = int(np.ceil(samples / (asr.HOP_LENGTH * 4)))
        if length < valid:
            raise RuntimeError('Encoder did not cover the complete acoustic block')
        self.encoder_calls += 1
        beams = self.beams
        if allowed:
            beams = asr.decode_encoder_beams(encoded[:, :, :valid], valid, beams,
                                            self.encoder_frame_offset,
                                            require_word_start=bool(self.committed_tokens),
                                            previous_word=self.committed_last_word,
                                            previous_words=self.committed_lm_context)
        self.encoder_frame_offset += valid
        self.decoder_frame_offset = self.encoder_frame_offset
        # Cache advancement never completes a word. Settle only an observed
        # shared word boundary with a full second of following acoustic time.
        settled = asr.settled_word_prefix(beams, self.encoder_frame_offset)
        self._commit_prefix(beams, settled)
        self.cache = cache
        self.history_audio = np.concatenate((self.history_audio, block))[-asr.NORMALIZATION_HISTORY_SAMPLES:]
        self.pending_audio = full_pending[samples:]
        self.preview_tokens = list(self.beams[0].tokens)
        self.hold_reason = 'word-pending' if self.preview_tokens else 'collecting'
        self.cache_events.append({'at_seconds': received / asr.SAMPLE_RATE,
                                  'cached_until': self.encoder_frame_offset * asr.ENCODER_FRAME_SECONDS,
                                  'window_seconds': full_pending.size / asr.SAMPLE_RATE,
                                  'word_end_seconds': None, 'reason': 'single-pass-block'})

    def update(self, audio, *, preview=True):
        if self.finished:
            raise RuntimeError('Cannot append audio after finish.')
        audio = np.asarray(audio, np.float32).reshape(-1)
        limit = self.block_samples + self.lookahead_samples
        offset = 0
        while offset < audio.size:
            take = min(limit - self.pending_audio.size, audio.size - offset)
            self.pending_audio = np.concatenate((self.pending_audio, audio[offset:offset + take]))
            self.total_samples += take
            self.peak_pending_samples = max(self.peak_pending_samples, self.pending_audio.size)
            offset += take
            if self.pending_audio.size == limit:
                self._consume(self.block_samples)
        return self._snapshot()

    def _snapshot(self, final=False):
        state = super()._snapshot(final)
        processed = min(self.total_samples / asr.SAMPLE_RATE,
                        self.encoder_frame_offset * asr.ENCODER_FRAME_SECONDS)
        return {**state, 'window_end': processed, 'window_seconds': processed,
                'received_until': self.total_samples / asr.SAMPLE_RATE,
                'recognition_mode': 'blocks', 'block_seconds': self.block_samples / asr.SAMPLE_RATE,
                'lookahead_seconds': self.lookahead_samples / asr.SAMPLE_RATE,
                'encoder_calls': self.encoder_calls, 'acoustic_reencodes': 0}

    def finish(self, audio=None):
        if audio is not None:
            self.update(audio)
        if not self.finished:
            while self.pending_audio.size:
                self._consume(min(self.block_samples, self.pending_audio.size))
            best = self.beams[0]
            if asr.kenlm_prior:
                def complete(h):
                    ids = h.token_ids
                    score = asr.kenlm_prior.full_score(asr.tokenizer.decode(ids) + ' ', self.committed_lm_context)
                    old = asr.kenlm_prior.token_score(asr.tokenizer, ids, self.committed_lm_context)
                    return h.score + asr.KENLM_WEIGHT * (score - old)
                best = max(self.beams, key=complete)
            self.committed_tokens.extend(best.tokens)
            self.beams = asr.new_decoder_beams()
            self.preview_tokens = None
            self.finished = True
            self.hold_reason = 'finished'
        tokens = self.committed_tokens
        text = asr.tokenizer.decode([t.token_id for t in tokens]).strip()
        return {**self._snapshot(True), 'final_text': text, 'tokens': [t.token_id for t in tokens],
                'words': asr.tokens_to_words(tokens), 'samples': self.total_samples,
                'seconds': self.total_samples / asr.SAMPLE_RATE,
                'feature_frames': int(np.ceil(self.total_samples / asr.HOP_LENGTH)),
                'encoded_frames': self.encoder_frame_offset,
                'peak_pending_window_seconds': self.peak_pending_samples / asr.SAMPLE_RATE,
                'cache_events': self.cache_events}
