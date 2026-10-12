"""Exact error accounting for verified-reference benchmarks; no ASR ground truth is inferred."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reference_scoring as scoring


class ReferenceScoringTests(unittest.TestCase):
    def test_short_excerpt_cannot_claim_complete_reference_coverage(self):
        report = scoring.score([scoring.Section(20, 30, "ਸ੍ਰੀ ਵਾਹਿਗੁਰੂ")],
                               {"text": "", "seconds": 10})
        self.assertTrue(report["reference_extends_past_audio"])
        self.assertEqual(report["deletions"], 2)

    def test_unannotated_intro_is_reported_separately_without_hiding_raw_error(self):
        report = scoring.score([scoring.Section(20, 30, "ਸ੍ਰੀ ਵਾਹਿਗੁਰੂ")],
                               {"text": "ਬੋਲੋ ਸ੍ਰੀ ਵਾਹਿਗੁਰੂ", "words": [
                                   {"text": "ਬੋਲੋ", "start": 1}, {"text": "ਸ੍ਰੀ", "start": 20},
                                   {"text": "ਵਾਹਿਗੁਰੂ", "start": 22}]})
        self.assertEqual(report["insertions"], 1)
        self.assertEqual(report["verified_window_score"]["wer"], 0)
        self.assertEqual(report["verified_window_score"]["outside_window_words"], 1)

    def test_false_word_split_is_counted_instead_of_joined_away(self):
        report = scoring.score([scoring.Section(3.75, 6, "ਵਾਹਿਗੁਰੂ ਨਾਮ ਜਪੋ")],
                               {"text": "ਵਾਰ ਗੁਰੂ ਨਾਮ ਜਪੋ"})
        self.assertEqual(report["substitutions"], 1)
        self.assertEqual(report["insertions"], 1)
        self.assertEqual(report["deletions"], 0)
        self.assertAlmostEqual(report["wer"], 2 / 3)

    def test_vowel_marks_and_repeated_words_are_preserved(self):
        report = scoring.score([scoring.Section(0, 4, "ਗੁਰੁ ਗੁਰੂ ਗੁਰੂ")], {"text": "ਗੁਰੂ ਗੁਰੂ"})
        self.assertEqual(report["reference_words"], 3)
        self.assertEqual(report["hypothesis_words"], 2)
        self.assertEqual(report["deletions"], 1)
        self.assertAlmostEqual(report["wer"], 1 / 3)
        self.assertEqual(scoring.normalized_words("ਗੁਰੁ, ਗੁਰੂ ॥੧॥"), ["ਗੁਰੁ", "ਗੁਰੂ"])

    def test_missing_phrase_reports_deletions_in_its_reference_section(self):
        sections = [scoring.Section(0, 4, "ਕਰਿ ਕਿਰਪਾ"), scoring.Section(4, 8, "ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ")]
        report = scoring.score(sections, {"text": "ਕਰਿ ਕਿਰਪਾ"})
        self.assertEqual(report["deletions"], 3)
        self.assertEqual(report["sections"][0]["wer"], 0)
        self.assertEqual(report["sections"][1]["wer"], 1)

    def test_linear_memory_alignment_and_character_distance_keep_exact_edit_cost(self):
        reference = ["ਨਾਮ", "ਗੁਰੂ", "ਨਾਮ", "ਜਪੋ", "ਕਰਿ"]
        hypothesis = ["ਨਾਮ", "ਗੁਰੂ", "ਗੁਰੂ", "ਨਾਮ", "ਜਪੋ", "ਕਿਰਪਾ"]
        operations = scoring.alignment(reference, hypothesis, matrix_cell_limit=2)
        self.assertEqual(sum(operation != "equal" for operation, _, _ in operations), 2)
        self.assertEqual(scoring.edit_distance(reference, hypothesis), 2)
        self.assertEqual(scoring.edit_distance("kitten", "sitting"), 3)
        self.assertEqual(scoring.edit_distance("ਵਾਹਿਗੁਰੂ", "ਵਾਹਿਗੁਰੂ"), 0)
        self.assertEqual(scoring.edit_distance("ਗੁਰੂ", ""), 4)

    def test_reference_schema_supports_timed_sections_and_rejects_invalid_times(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps({"segments": [{"start": "00:03.75", "end": 6, "text": "ਵਾਹਿਗੁਰੂ"}]}))
            self.assertEqual(scoring.read_sections(path), [scoring.Section(3.75, 6, "ਵਾਹਿਗੁਰੂ")])
            path.write_text(json.dumps([{"start": 6, "end": 3.75, "text": "ਵਾਹਿਗੁਰੂ"}]))
            with self.assertRaises(ValueError):
                scoring.read_sections(path)


if __name__ == "__main__":
    unittest.main()
