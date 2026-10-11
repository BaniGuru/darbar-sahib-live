"""Compare full-file HTTP ASR with a real-time paced WebSocket recording.

Example: python scripts/validate_streaming_asr.py test.mp3 --url http://127.0.0.1:8001
The server must already be running. Uses only the application's dependencies.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import wave

import httpx
import numpy as np
import websockets


async def validate(args):
    audio_path = Path(args.audio)
    with tempfile.TemporaryDirectory() as directory:
        wav_path = Path(directory) / "audio.wav"
        subprocess.run([os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg",
                        "-hide_banner", "-loglevel", "error", "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)], check=True)
        with wave.open(str(wav_path)) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        wav_bytes = wav_path.read_bytes()
    async with httpx.AsyncClient(timeout=180) as client:
        before = time.perf_counter()
        response = await client.post(args.url.rstrip("/") + "/transcribe",
                                     files={"file": ("audio.wav", wav_bytes, "audio/wav")})
        response.raise_for_status()
        full = response.json()
        full_seconds = time.perf_counter() - before
    if "error" in full:
        raise RuntimeError(full["error"])
    url = args.url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    events = []
    async with websockets.connect(url + "/ws/transcribe", max_size=16 * 1024 * 1024) as socket:
        ready = json.loads(await socket.recv())
        origin = time.perf_counter()

        async def send():
            packet = max(1, round(args.packet_ms * 16))
            for start in range(0, len(pcm), packet):
                end = min(start + packet, len(pcm))
                await asyncio.sleep(max(0, origin + end / 16000 - time.perf_counter()))
                await socket.send(pcm[start:end].astype("<f4").tobytes())
            await socket.send(json.dumps({"type": "finish"}))

        async def receive():
            async for raw in socket:
                state = json.loads(raw)
                if state.get("type") == "error":
                    raise RuntimeError(state)
                elapsed = time.perf_counter() - origin
                state["received_at_seconds"] = elapsed
                state["lag_seconds"] = elapsed - state["window_end"]
                if state["type"] == "final":
                    return state
                events.append(state)
            raise RuntimeError("Socket closed without flushing a final transcript.")

        _, final = await asyncio.wait_for(asyncio.gather(send(), receive()), len(pcm) / 16000 + 90)
    assert final["samples"] == full["samples"] == len(pcm), "Dropped terminal samples"
    assert final["encoded_frames"] == full["encoded_frames"] == int(np.ceil(len(pcm) / 640)), "Dropped encoder frames"
    assert final["tokens"] == full["tokens"], "Streaming tokens differ from full-file tokens"
    assert final["text"] == full["text"], "Streaming transcript differs from full-file transcript"
    previous_committed = ""
    for state in events:
        joined = " ".join(x for x in (state["committed_text"], state["partial_text"]) if x)
        assert joined == state["text"], "A word was split between text fields"
        words = state["committed_words"] + state["partial_words"]
        assert " ".join(word["text"] for word in words) == state["text"], "Word records omit text"
        committed = state["committed_text"]
        assert not previous_committed or committed == previous_committed or committed.startswith(previous_committed + " "), "Settled words changed"
        assert not committed or final["text"] == committed or final["text"].startswith(committed + " "), "A settled word was lost"
        previous_committed = committed
        assert state["pending_window_seconds"] <= ready["max_pending_window_seconds"], "Window exceeded its limit"
    ends = [state["window_end"] for state in events]
    assert np.allclose(np.diff([0] + ends), ready["step_seconds"]), "Incorrect refresh cadence"
    lags = [state["lag_seconds"] for state in events]
    boundaries = final["cache_events"]
    assert boundaries == full["cache_events"], "Packet sizes changed adaptive cache decisions"
    assert final["peak_pending_window_seconds"] <= ready["max_pending_window_seconds"], "Peak window exceeded its limit"
    for index, commit in enumerate(boundaries, 1):
        assert commit["cached_until"] == index * ready["cache_commit_seconds"], "Incorrect cache step"
        assert ready["cache_commit_seconds"] <= commit["window_seconds"] <= ready["max_pending_window_seconds"], "Incorrect lookahead window"
        if commit["word_end_seconds"] is not None:
            assert commit["at_seconds"] + 1e-6 >= commit["word_end_seconds"] + ready["word_lookahead_seconds"], "Word cached without finishing lookahead"
    first = next((state for state in events if state["text"]), None)
    report = {
        "audio": audio_path.name, "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "audio_seconds": len(pcm) / 16000, "packet_ms": args.packet_ms, "paced_in_real_time": True,
        "full_inference_seconds": full_seconds, "full": full, "streaming_final": final,
        "same_tokens_and_text": True, "updates": len(events), "cache_commits": len(boundaries),
        "peak_pending_window_seconds": final["peak_pending_window_seconds"],
        "lag_seconds": {"median": float(np.median(lags)), "p95": float(np.percentile(lags, 95)), "max": max(lags)},
        "first_text_audio_seconds": first["window_end"] if first else None,
        "first_text_received_seconds": first["received_at_seconds"] if first else None,
        "events": events,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"full", "streaming_final", "events"}}, indent=2))
    print("Transcript:", full["text"])
    print("Report:", output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--packet-ms", type=float, default=100)
    parser.add_argument("--report", default="/tmp/darbar-validation/streaming-report.json")
    asyncio.run(validate(parser.parse_args()))
