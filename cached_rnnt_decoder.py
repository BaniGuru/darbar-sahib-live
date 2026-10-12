"""Factor the existing RNNT graph so blanks reuse identical predictor work.

Only the ONNX computation is separated; weights and token scores are unchanged.
Bounded caches use complete input tensors, never audio or reference text.
"""
from collections import OrderedDict

import numpy as np
import onnx
from onnx.utils import Extractor
import onnxruntime as ort


class CachedRNNTDecoder:
    def __init__(self, original, model_path, session_options, cache_size=16384):
        self.original = original
        self.cache_size = cache_size
        graph = onnx.shape_inference.infer_shapes(onnx.load(model_path))
        extractor = Extractor(graph)
        self.encoder_name = "/joint/enc/Add_output_0"
        self.predictor_name = "/joint/pred/Add_output_0"
        inputs = [value.name for value in original.get_inputs()]
        self.input_names = inputs
        self.output_names = [value.name for value in original.get_outputs()]
        def session(ins, outs):
            model = extractor.extract_model(ins, outs)
            return ort.InferenceSession(model.SerializeToString(), sess_options=session_options,
                                        providers=["CPUExecutionProvider"])
        self.encoder = session([inputs[0]], [self.encoder_name])
        self.predictor = session([inputs[1], inputs[3], inputs[4]],
                                 [self.predictor_name, self.output_names[2], self.output_names[3]])
        self.joint = session([self.encoder_name, self.predictor_name], [self.output_names[0]])
        self.predictor_cache = OrderedDict()
        self.encoder_cache = OrderedDict()

    def get_inputs(self):
        return self.original.get_inputs()

    def get_outputs(self):
        return self.original.get_outputs()

    @staticmethod
    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)
        return value

    def run(self, output_names, inputs):
        names = self.input_names
        acoustic, labels, lengths, h, c = (inputs[name] for name in names)
        # Beam search repeats one acoustic frame across the batch. Project it
        # once; include shape and dtype to distinguish arbitrary callers.
        encoder_key = (acoustic.shape[1:], acoustic.dtype.str, acoustic[0:1].tobytes())
        shared_frame = np.array_equal(acoustic, np.broadcast_to(acoustic[0:1], acoustic.shape))
        if shared_frame:
            projected = self.encoder_cache.get(encoder_key)
            if projected is None:
                projected = self._remember(self.encoder_cache, encoder_key,
                    self.encoder.run(None, {names[0]: acoustic[0:1]})[0], 128)
            enc = np.repeat(projected, len(labels), axis=0)
        else:
            enc = self.encoder.run(None, {names[0]: acoustic})[0]
        predictions = [None] * len(labels)
        missing = {}
        for index in range(len(labels)):
            state_h, state_c = h[:, index:index + 1], c[:, index:index + 1]
            target = labels[index:index + 1]
            key = tuple((value.shape, value.dtype.str, value.tobytes())
                        for value in (target, state_h, state_c))
            prediction = self.predictor_cache.get(key)
            if prediction is None:
                missing.setdefault(key, []).append(index)
            else:
                self.predictor_cache.move_to_end(key)
                predictions[index] = prediction
        if missing:
            positions = [indices[0] for indices in missing.values()]
            result = self.predictor.run(None, {
                names[1]: labels[positions], names[3]: h[:, positions, :], names[4]: c[:, positions, :],
            })
            for offset, (key, indices) in enumerate(missing.items()):
                prediction = self._remember(self.predictor_cache, key,
                    [result[0][offset:offset + 1].copy(),
                     result[1][:, offset:offset + 1].copy(), result[2][:, offset:offset + 1].copy()],
                    self.cache_size)
                for index in indices:
                    predictions[index] = prediction
        pred = np.concatenate([value[0] for value in predictions], axis=0)
        logits = self.joint.run(None, {self.encoder_name: enc, self.predictor_name: pred})[0]
        result = [logits, lengths.copy(),
                  np.concatenate([value[1] for value in predictions], axis=1),
                  np.concatenate([value[2] for value in predictions], axis=1)]
        if output_names is None:
            return result
        return [result[self.output_names.index(name)] for name in output_names]

    @staticmethod
    def _readonly(value):
        value = value.copy()
        value.setflags(write=False)
        return value

    def run_hypotheses(self, acoustic, checkpoints):
        """Reuse immutable owned checkpoints without serializing their tensors.

        Strong references in each cache entry prevent identity reuse. Mutable
        arrays/views use the existing content-keyed path. Both paths share the
        same bounded cache and unchanged extracted ONNX graphs.
        """
        names = self.input_names
        batch = len(checkpoints)
        if any(h.flags.writeable or c.flags.writeable or not h.flags.owndata or not c.flags.owndata
               for h, c, label in checkpoints):
            result = self.run(None, {
                names[0]: np.repeat(acoustic, batch, axis=0),
                names[1]: np.asarray([[p[2]] for p in checkpoints], np.int32),
                names[2]: np.ones(batch, np.int32),
                names[3]: np.concatenate([p[0] for p in checkpoints], axis=1),
                names[4]: np.concatenate([p[1] for p in checkpoints], axis=1),
            })
            return result[0], [(self._readonly(result[2][:, i:i + 1]),
                                self._readonly(result[3][:, i:i + 1])) for i in range(batch)]
        key = (acoustic.shape[1:], acoustic.dtype.str, acoustic.tobytes())
        projected = self.encoder_cache.get(key)
        if projected is None:
            projected = self._remember(self.encoder_cache, key,
                                       self.encoder.run(None, {names[0]: acoustic})[0], 128)
        predictions = [None] * batch
        missing = {}
        for index, (h, c, label) in enumerate(checkpoints):
            key = ('immutable', id(h), id(c), label)
            cached = self.predictor_cache.get(key)
            if cached is None:
                missing.setdefault(key, []).append(index)
            else:
                self.predictor_cache.move_to_end(key)
                predictions[index] = cached[2]
        if missing:
            positions = [indices[0] for indices in missing.values()]
            result = self.predictor.run(None, {
                names[1]: np.asarray([[checkpoints[i][2]] for i in positions], np.int32),
                names[3]: np.concatenate([checkpoints[i][0] for i in positions], axis=1),
                names[4]: np.concatenate([checkpoints[i][1] for i in positions], axis=1),
            })
            for offset, (key, indices) in enumerate(missing.items()):
                prediction = (result[0][offset:offset + 1].copy(),
                              self._readonly(result[1][:, offset:offset + 1]),
                              self._readonly(result[2][:, offset:offset + 1]))
                h, c, label = checkpoints[indices[0]]
                self._remember(self.predictor_cache, key, (h, c, prediction), self.cache_size)
                for index in indices:
                    predictions[index] = prediction
        logits = self.joint.run(None, {
            self.encoder_name: np.repeat(projected, batch, axis=0),
            self.predictor_name: np.concatenate([p[0] for p in predictions], axis=0),
        })[0]
        return logits, [(p[1], p[2]) for p in predictions]
