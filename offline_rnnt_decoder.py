"""Probability-pruned RNNT beam search for the offline Punjabi ONNX head."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Hypothesis:
    tokens: tuple[tuple[int, int], ...]
    ids: tuple[int, ...]
    predictor: tuple[np.ndarray, np.ndarray, int]
    score: float = 0.0


def merge(candidates, hypothesis):
    previous = candidates.get(hypothesis.ids)
    if previous is None:
        candidates[hypothesis.ids] = hypothesis
    else:
        best = hypothesis if hypothesis.score > previous.score else previous
        candidates[hypothesis.ids] = Hypothesis(
            best.tokens, best.ids, best.predictor,
            float(np.logaddexp(previous.score, hypothesis.score)))


def decode_beam(encoded, length, session, inputs, tokenizer, vocabulary, blank,
                start_label, max_symbols, width=4, prior=None, prior_weight=0.6):
    beams = [Hypothesis((), (), (np.zeros((1, 1, 640), np.float32),
                                 np.zeros((1, 1, 640), np.float32), start_label))]
    prior_scores = {}
    def evidence(ids):
        if prior is None:
            return 0.0
        if ids not in prior_scores:
            prior_scores[ids] = prior_weight * prior.score(tokenizer.decode(ids))
        return prior_scores[ids]
    expansion = 2 * width if prior is not None else width
    for frame in range(int(length)):
        active, completed = beams, {}
        for symbols in range(max_symbols + 1):
            if not active:
                break
            batch = len(active)
            result = session.run(None, {
                inputs[0]: np.repeat(encoded[:, :, frame:frame + 1], batch, axis=0),
                inputs[1]: np.asarray([[h.predictor[2]] for h in active], np.int32),
                inputs[2]: np.ones(batch, np.int32),
                inputs[3]: np.concatenate([h.predictor[0] for h in active], axis=1),
                inputs[4]: np.concatenate([h.predictor[1] for h in active], axis=1)})
            logits = result[0].reshape(batch, -1, vocabulary + 1)[:, 0, :].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            for index, h in enumerate(active):
                merge(completed, Hypothesis(h.tokens, h.ids, h.predictor,
                                           h.score + float(probabilities[index, blank])))
            best_completed = max(h.score for h in completed.values())
            if symbols == max_symbols:
                break
            expanded = {}
            for index, h in enumerate(active):
                labels = np.argsort(probabilities[index, :vocabulary])[-expansion:]
                for label in labels:
                    raw = h.score + float(probabilities[index, label])
                    if raw < best_completed - 6.0 - 12 * prior_weight:
                        continue
                    label = int(label)
                    ids = h.ids + (label,)
                    score = raw + evidence(ids) - evidence(h.ids)
                    if score < best_completed - 6.0:
                        continue
                    # This exported decoder's embedding table accepts local
                    # Punjabi feedback IDs; the start sentinel remains 5632.
                    state = (result[2][:, index:index + 1, :].copy(),
                             result[3][:, index:index + 1, :].copy(), label)
                    merge(expanded, Hypothesis(h.tokens + ((label, frame),), ids, state, score))
            active = sorted(expanded.values(), key=lambda h: h.score, reverse=True)[:width]
        beams = sorted(completed.values(), key=lambda h: h.score, reverse=True)[:width]
        top_score = beams[0].score
        beams = [Hypothesis(h.tokens, h.ids, h.predictor, h.score - top_score)
                 for h in beams if h.score >= top_score - 6.0]
    return list(beams[0].tokens)
