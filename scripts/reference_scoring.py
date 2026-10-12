"""Exact human-reference scoring for offline and streaming validation.

Reference sections are used only to score completed recognition. Word
boundaries and Gurmukhi vowel marks remain significant in error counts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import unicodedata

import numpy as np


@dataclass
class Section:
    start: float
    end: float
    text: str


def normalized_words(text: str) -> list[str]:
    """Keep Gurmukhi letters/vowel marks and word divisions intact.

    Ignore punctuation and standalone verse numbers. No spelling fixes,
    diacritic stripping, approximate matching, or joining of split words.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    cleaned = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return [word for word in cleaned.split() if not word.isnumeric()]


def seconds(value) -> float:
    if isinstance(value, str) and ":" in value:
        parts = value.split(":")
        result = sum(float(part) * 60 ** index for index, part in enumerate(reversed(parts)))
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError("Reference timestamps must be finite.")
    return result


def read_sections(path: Path) -> list[Section]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = next((payload[key] for key in ("segments", "sections", "chunks", "transcript", "transcription")
                        if isinstance(payload.get(key), list)), None)
    if not isinstance(payload, list):
        raise ValueError("Expected a list of timed sections, or an object containing segments/sections/chunks.")
    sections = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Reference section {index} is not an object.")
        text = next((item[key] for key in ("text", "transcript", "sentence")
                     if isinstance(item.get(key), str)), None)
        if text is None:
            raise ValueError(f"Reference section {index} has no text field.")
        if "timestamp" in item:
            start, end = item["timestamp"]
        else:
            start, end = item.get("start", item.get("start_time")), item.get("end", item.get("end_time"))
        if start is None or end is None:
            raise ValueError(f"Reference section {index} is missing its start or end time.")
        start, end = seconds(start), seconds(end)
        if start < 0 or end <= start:
            raise ValueError(f"Reference section {index} has an invalid interval: {start}–{end}.")
        if normalized_words(text):
            sections.append(Section(start, end, text))
    if not sections:
        raise ValueError("The reference contains no nonempty transcript sections.")
    return sorted(sections, key=lambda section: (section.start, section.end))


def edit_distance(reference, hypothesis) -> int:
    """Exact bit-parallel Levenshtein distance, including long character strings."""
    if not reference:
        return len(hypothesis)
    matches = {}
    for index, value in enumerate(reference):
        matches[value] = matches.get(value, 0) | (1 << index)
    mask = (1 << len(reference)) - 1
    last = 1 << (len(reference) - 1)
    positive, negative, distance = mask, 0, len(reference)
    for value in hypothesis:
        equal = matches.get(value, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        up = negative | ~(horizontal | positive)
        down = positive & horizontal
        distance += bool(up & last) - bool(down & last)
        up, down = (up << 1) | 1, down << 1
        positive = (down | ~(vertical | up)) & mask
        negative = (up & vertical) & mask
    return distance


def alignment(reference: list[str], hypothesis: list[str], *, matrix_cell_limit: int = 20_000_000):
    """Exact word alignment; linear-memory divide-and-conquer for large inputs."""
    vocabulary = {word: index for index, word in enumerate(set(reference + hypothesis))}
    left = np.asarray([vocabulary[word] for word in reference], dtype=np.int32)
    right = np.asarray([vocabulary[word] for word in hypothesis], dtype=np.int32)

    def rows(xs, ys, keep):
        positions = np.arange(len(ys) + 1, dtype=np.int32)
        previous = positions.copy()
        matrix = np.empty((len(xs) + 1, len(ys) + 1), np.int32) if keep else None
        if keep:
            matrix[0] = previous
        for index, value in enumerate(xs, 1):
            current = np.empty_like(previous)
            current[0] = index
            current[1:] = np.minimum(previous[1:] + 1, previous[:-1] + (ys != value))
            previous = np.minimum.accumulate(current - positions) + positions
            if keep:
                matrix[index] = previous
        return matrix if keep else previous

    def solve(xs, ys, xoffset, yoffset):
        if not len(xs):
            return [("insert", None, yoffset + index) for index in range(len(ys))]
        if not len(ys):
            return [("delete", xoffset + index, None) for index in range(len(xs))]
        if min(len(xs), len(ys)) > 1 and (len(xs) + 1) * (len(ys) + 1) > matrix_cell_limit:
            middle = len(xs) // 2
            split = int(np.argmin(rows(xs[:middle], ys, False) + rows(xs[middle:][::-1], ys[::-1], False)[::-1]))
            return solve(xs[:middle], ys[:split], xoffset, yoffset) + solve(xs[middle:], ys[split:], xoffset + middle, yoffset + split)
        matrix = rows(xs, ys, True)
        x, y, operations = len(xs), len(ys), []
        while x or y:
            if x and y and matrix[x, y] == matrix[x - 1, y - 1] + (xs[x - 1] != ys[y - 1]):
                operations.append(("equal" if xs[x - 1] == ys[y - 1] else "substitute", xoffset + x - 1, yoffset + y - 1))
                x, y = x - 1, y - 1
            elif x and matrix[x, y] == matrix[x - 1, y] + 1:
                operations.append(("delete", xoffset + x - 1, None))
                x -= 1
            else:
                operations.append(("insert", None, yoffset + y - 1))
                y -= 1
        return operations[::-1]

    return solve(left, right, 0, 0)


def score(sections: list[Section], result: dict, *, include_coverage: bool = True) -> dict:
    reference, owners = [], []
    details = []
    for index, section in enumerate(sections):
        words = normalized_words(section.text)
        reference.extend(words)
        owners.extend([index] * len(words))
        details.append({"start": section.start, "end": section.end, "reference": section.text,
                        "reference_words": len(words), "substitute": 0, "delete": 0, "insert": 0,
                        "hypothesis": [], "errors": []})
    hypothesis = normalized_words(result["text"])
    operations = alignment(reference, hypothesis)
    counts = {"substitute": 0, "delete": 0, "insert": 0}
    previous_owner = 0
    for operation, ref_index, hyp_index in operations:
        owner = owners[ref_index] if ref_index is not None else previous_owner
        previous_owner = owner
        detail = details[owner]
        if hyp_index is not None:
            detail["hypothesis"].append(hypothesis[hyp_index])
        if operation != "equal":
            counts[operation] += 1
            detail[operation] += 1
            detail["errors"].append({"operation": operation,
                                     "reference": reference[ref_index] if ref_index is not None else None,
                                     "hypothesis": hypothesis[hyp_index] if hyp_index is not None else None})
    for detail in details:
        detail["hypothesis"] = " ".join(detail["hypothesis"])
        detail["wer"] = sum(detail[key] for key in counts) / detail["reference_words"]
    ref_text, hyp_text = " ".join(reference), " ".join(hypothesis)
    raw_ref = " ".join(section.text for section in sections).split()
    overlaps = sum(left.end > right.start for left, right in zip(sections, sections[1:]))
    duration = result.get("seconds")
    report = {"reference_words": len(reference), "hypothesis_words": len(hypothesis),
            "substitutions": counts["substitute"], "deletions": counts["delete"], "insertions": counts["insert"],
            "wer": sum(counts.values()) / len(reference),
            "cer_including_spaces": edit_distance(ref_text, hyp_text) / len(ref_text),
            "strict_wer": edit_distance(raw_ref, result["text"].split()) / len(raw_ref),
            "reference_section_count": len(sections), "overlapping_reference_intervals": overlaps,
            "reference_last_end_seconds": max(section.end for section in sections),
            "scored_audio_seconds": duration,
            "reference_extends_past_audio": (max(section.end for section in sections) > duration + 0.5
                                            if duration is not None else None),
            "normalization": "NFC, casefold, punctuation removal and standalone verse-number removal; word spaces and Gurmukhi vowel marks preserved",
            "section_alignment": "Sections use global word alignment, not timestamp cuts; repeated passages can have ambiguous alignment.",
            "sections": details}
    # Keep the all-output score primary. The supplementary score distinguishes
    # unannotated gaps from errors inside the supplied human-verified windows.
    if include_coverage and isinstance(result.get("words"), list):
        tolerance = 0.5
        covered, outside = [], []
        for word in result["words"]:
            if any(section.start - tolerance <= float(word["start"]) <= section.end + tolerance
                   for section in sections):
                covered.append(word["text"])
            else:
                outside.append(word["text"])
        window_score = score(sections, {"text": " ".join(covered)}, include_coverage=False)
        report["verified_window_score"] = {key: window_score[key] for key in
            ("reference_words", "hypothesis_words", "substitutions", "deletions", "insertions", "wer")}
        report["verified_window_score"].update({"timestamp_tolerance_seconds": tolerance,
            "selection": "Recognized word starts inside any reference interval, including the declared tolerance",
            "outside_window_words": len(normalized_words(" ".join(outside))),
            "outside_window_text": " ".join(outside)})
    return report
