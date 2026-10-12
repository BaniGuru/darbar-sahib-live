from dataclasses import dataclass
import unittest

import numpy as np

from offline_audio_windows import merge_overlapping_words, transcribe_windows


@dataclass
class Word:
    text: str
    start: float
    end: float
    token_ids: list[int]


class OfflineWindowTests(unittest.TestCase):
    def test_word_moving_across_nominal_cut_is_kept_once_and_whole(self):
        old = [Word("ਸ੍ਰੀ", 10, 20.2, [1]), Word("ਵਾਹਿਗੁਰੂ", 20.2, 24, [2, 3])]
        new = [Word("ਵਾਹਿਗੁਰੂ", 19.8, 24.1, [2, 3, 4]), Word("ਨਾਮ", 24.1, 25, [5])]
        joined, policy = merge_overlapping_words(old, new, 15, 25)
        self.assertEqual([w.text for w in joined], ["ਸ੍ਰੀ", "ਵਾਹਿਗੁਰੂ", "ਨਾਮ"])
        self.assertEqual(joined[1].token_ids, [2, 3, 4])
        self.assertEqual(policy, "shared-word")

    def test_real_repeated_words_are_not_deduplicated(self):
        old = [Word("ਗੁਰੂ", t, t + 1, [1]) for t in [12, 17, 20]]
        new = [Word("ਗੁਰੂ", t, t + 1, [1]) for t in [17.1, 20.1, 24, 30]]
        joined, _ = merge_overlapping_words(old, new, 15, 25)
        self.assertEqual(len(joined), 5)
        self.assertEqual([w.start for w in joined], [12, 17, 20.1, 24, 30])

    def test_long_recording_bounds_attention_and_covers_last_sample(self):
        audio = np.zeros(65 * 16000 + 13, np.float32)
        calls = []
        def transcribe(part, offset):
            calls.append((offset, len(part)))
            return {"words": [], "encoded_frames": int(np.ceil(len(part) / 640))}
        result = transcribe_windows(audio, transcribe)
        self.assertEqual(result["samples"], len(audio))
        self.assertTrue(all(n <= 30 * 16000 for _, n in calls))
        self.assertEqual(result["offline_windows"][-1]["core_end"], len(audio) / 16000)
        windows = result["offline_windows"]
        self.assertTrue(all(a["core_end"] == b["core_start"] for a, b in zip(windows, windows[1:])))
        self.assertEqual(calls[-1][0] + calls[-1][1] / 16000, len(audio) / 16000)


if __name__ == "__main__":
    unittest.main()
