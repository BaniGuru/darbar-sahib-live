"""Transcribe every sample with the offline model and score a human reference.

--source can load an archived original main module with the same CPU budget.
Windowing is identical for baseline/candidate and independent of references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import onnxruntime as ort

from reference_scoring import read_sections, score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline_audio_windows import transcribe_windows


def load_model(source_path: Path):
    source = source_path.read_text(encoding="utf-8")
    module = types.ModuleType("offline_benchmark_model")
    # Resolve assets in the checkout even when source is an archived snapshot.
    module.__file__ = str(ROOT / "server_asr_offline.py")
    sys.modules[module.__name__] = module
    def options(threads):
        value = ort.SessionOptions()
        value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        value.intra_op_num_threads = threads
        value.inter_op_num_threads = 1
        value.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return value
    module.__dict__["_benchmark_options"] = options
    # Thread limits do not change main's model or greedy decoding algorithm.
    source = source.replace("sess_options=session_options,", f"sess_options=_benchmark_options({int(os.getenv('ASR_ENCODER_THREADS', '4'))}),", 1)
    source = source.replace("sess_options=session_options,", "sess_options=_benchmark_options(1),", 1)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def json_words(words):
    return [word.to_json() if hasattr(word, "to_json") else word for word in words]


def run(audio_path, source_path, checkpoint, stitch="anchors"):
    model = load_model(source_path)
    with tempfile.TemporaryDirectory() as directory:
        pcm = Path(directory) / "audio.f32"
        subprocess.run([os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg",
                        "-hide_banner", "-loglevel", "error", "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1", "-f", "f32le", str(pcm)], check=True)
        audio = np.memmap(pcm, dtype="<f4", mode="r")
        origin = last_log = time.perf_counter()
        def progress(seconds, words, windows):
            nonlocal last_log
            state = {"completed": False, "seconds": seconds,
                     "text": " ".join(w.text for w in words), "words": json_words(words),
                     "offline_windows": windows, "processing_seconds": time.perf_counter() - origin}
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            temporary.replace(checkpoint)
            if time.perf_counter() - last_log >= 30:
                print(json.dumps({"audio_seconds": seconds, "total_seconds": len(audio) / 16000,
                                  "elapsed_seconds": time.perf_counter() - origin}), flush=True)
                last_log = time.perf_counter()
        if hasattr(model, "_transcribe_window_sync"):
            result = model.transcribe_pcm_sync(audio, progress=progress, stitch=stitch)
        else:
            result = transcribe_windows(audio, model.transcribe_pcm_sync, progress=progress, stitch=stitch)
        elapsed = time.perf_counter() - origin
        result["words"] = json_words(result["words"])
    return result, {"elapsed_seconds": elapsed, "real_time_factor": elapsed / result["seconds"],
                    "encoder_threads": int(os.getenv("ASR_ENCODER_THREADS", "4")), "decoder_threads": 1, "causal": False,
                    "window_seconds": 30, "core_seconds": 20, "left_right_context_seconds": 5,
                    "stitch": stitch, "decoder_beam_width": getattr(model, "DECODER_BEAM_WIDTH", 1),
                    "language_prior_weight": getattr(model, "LANGUAGE_PRIOR_WEIGHT", 0),
                    "orthography_restoration": getattr(model, "RESTORE_GURBANI_ORTHOGRAPHY", False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--hypothesis", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "server_asr_offline.py")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stitch", choices=["anchors", "ownership"], default="anchors")
    args = parser.parse_args()
    if bool(args.audio) == bool(args.hypothesis):
        parser.error("Choose exactly one of --audio or --hypothesis")
    sections = read_sections(args.reference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.audio:
        result, performance = run(args.audio, args.source, args.report.with_suffix(".partial.json"), args.stitch)
    else:
        result = json.loads(args.hypothesis.read_text())
        result = result.get("recognition", result)
        performance = None
    scores = score(sections, result)
    report = {"completed": not scores["reference_extends_past_audio"],
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "audio_sha256": None, "performance": performance, "scores": scores, "recognition": result}
    if "raw_text" in result:
        report["raw_decoder_scores"] = score(sections, {"text": result["raw_text"],
            "words": result["raw_words"], "seconds": result["seconds"]})
    if args.audio:
        with args.audio.open("rb") as f: report["audio_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in scores.items() if k != "sections"}, ensure_ascii=False, indent=2))
    print("Report:", args.report.resolve())
    if not report["completed"]:
        raise SystemExit("Incomplete reference coverage")


if __name__ == "__main__":
    main()
