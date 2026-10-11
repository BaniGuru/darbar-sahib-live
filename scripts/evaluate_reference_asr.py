"""Score a complete streaming recording against an aligned reference.

Reference text is used only for scoring after recognition. Replay is unpaced
by default; --paced sends each 800 ms of audio on its original time clock.
Use --hypothesis to score a previously saved recognizer/validation JSON.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
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


def replay(audio_path: Path, paced: bool, checkpoint_path: Path | None = None) -> tuple[dict, dict]:
    # The reference is deliberately absent from this function's inputs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import server_asr_stream as asr

    with tempfile.TemporaryDirectory() as directory:
        pcm_path = Path(directory) / "audio.f32"
        subprocess.run([os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg",
                        "-hide_banner", "-loglevel", "error", "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1", "-f", "f32le", str(pcm_path)], check=True)
        audio = np.memmap(pcm_path, dtype="<f4", mode="r")
        recognizer = asr.StreamingRecognizer()
        step, timings = asr.PARTIAL_INTERVAL_SAMPLES, []
        origin, last_log = time.perf_counter(), time.perf_counter()
        checkpoint_at = 60.0
        for start in range(0, len(audio), step):
            end = min(start + step, len(audio))
            if paced:
                time.sleep(max(0, origin + end / 16000 - time.perf_counter()))
            before = time.perf_counter()
            state = recognizer.update(audio[start:end])
            timings.append(time.perf_counter() - before)
            if checkpoint_path and end / 16000 >= checkpoint_at:
                partial = {"completed": False, "text": state["text"],
                           "words": state["committed_words"] + state["partial_words"],
                           "seconds": end / 16000, "samples": end,
                           "cache_events": recognizer.cache_events,
                           "processing_seconds": time.perf_counter() - origin}
                temporary = checkpoint_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(partial, ensure_ascii=False), encoding="utf-8")
                temporary.replace(checkpoint_path)
                checkpoint_at += 60
            if time.perf_counter() - last_log >= 30:
                print(json.dumps({"audio_seconds": end / 16000, "total_seconds": len(audio) / 16000,
                                  "elapsed_seconds": time.perf_counter() - origin,
                                  "cached_until": state["cache_committed_until"],
                                  "pending_window_seconds": state["pending_window_seconds"]}), flush=True)
                last_log = time.perf_counter()
        result = recognizer.finish()
        elapsed = time.perf_counter() - origin
    return result, {"paced": paced, "elapsed_seconds": elapsed,
                    "real_time_factor": elapsed / result["seconds"], "updates": len(timings),
                    "update_processing_seconds": {"median": float(np.median(timings)),
                                                  "p95": float(np.percentile(timings, 95)), "max": max(timings)},
                    "first_quarter_median_seconds": float(np.median(timings[:max(1, len(timings) // 4)])),
                    "last_quarter_median_seconds": float(np.median(timings[-max(1, len(timings) // 4):]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--hypothesis", type=Path)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.audio) == bool(args.hypothesis):
        parser.error("Choose exactly one of --audio or --hypothesis.")
    # Validate the schema first, but never pass its text/timestamps to ASR.
    sections = read_sections(args.reference)
    if args.hypothesis:
        result = json.loads(args.hypothesis.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Hypothesis JSON must contain a recognizer result object.")
        result = result.get("recognition", result.get("streaming_final", result.get("full", result)))
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ValueError("Hypothesis JSON does not contain transcript text.")
        performance = None
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        result, performance = replay(args.audio, args.paced, args.report.with_suffix(".partial.json"))
    audio_digest = None
    if args.audio:
        with args.audio.open("rb") as source:
            audio_digest = hashlib.file_digest(source, "sha256").hexdigest()
    scores = score(sections, result)
    report = {"completed": not scores["reference_extends_past_audio"], "recognition_completed": True,
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "audio_sha256": audio_digest,
              "audio": str(args.audio) if args.audio else None, "performance": performance,
              "scores": scores, "recognition": result}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report["scores"].items() if key != "sections"}, ensure_ascii=False, indent=2))
    print("Report:", args.report.resolve())
    if scores["reference_extends_past_audio"]:
        print("INCOMPLETE BENCHMARK: The reference extends beyond this audio; full-reference accuracy is not validated.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
