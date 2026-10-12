"""Compare guard on/off using an idle capture and audible recording(s).

Example: python scripts/validate_no_audio_guard.py --idle idle.aac \
    --audio kirtan.mp3 --report /tmp/no-audio.json
Recognition uses causal 800 ms updates; this is an unpaced CPU replay.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def pcm(path):
    if path.suffix == '.f32':
        return np.fromfile(path, dtype='<f4')
    decoded = subprocess.run([os.getenv('FFMPEG_PATH') or shutil.which('ffmpeg') or 'ffmpeg',
                              '-v', 'error', '-i', str(path), '-ar', '16000', '-ac', '1',
                              '-f', 'f32le', '-'], check=True, capture_output=True)
    return np.frombuffer(decoded.stdout, dtype='<f4')


def replay(asr, audio, enabled):
    asr.SILENCE_GUARD = enabled
    recognizer = asr.StreamingRecognizer()
    states = []
    started = time.perf_counter()
    for start in range(0, len(audio), 12800):
        state = recognizer.update(audio[start:start + 12800])
        if min(start + 12800, len(audio)) % 12800 == 0:
            states.append({key: state[key] for key in ('window_end', 'text', 'partial_text', 'audio_active')})
    final = recognizer.finish()
    elapsed = time.perf_counter() - started
    return {'guard_enabled': enabled, 'elapsed_seconds': elapsed,
            'rtf': elapsed / (len(audio) / 16000), 'updates': states, 'final': final}


def main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import server_asr_stream as asr
    original = asr.SILENCE_GUARD
    report = {'completed': False, 'paced': False, 'runtime': asyncio.run(asr.health()),
              'cases': [], 'sources_sha256': {
                  name: hashlib.sha256((asr.BASE_DIR / name).read_bytes()).hexdigest()
                  for name in ('audio_activity.py', 'server_asr_stream.py', 'server.py')}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        for path in [args.idle, *args.audio]:
            audio = pcm(path)
            if not len(audio):
                raise ValueError(f'Empty recording: {path}')
            gains = [1.0] if path == args.idle else [1.0, .01]
            for gain in gains:
                before = replay(asr, audio * np.float32(gain), False)
                after = replay(asr, audio * np.float32(gain), True)
                if path == args.idle:
                    assert after['final']['text'] == '', 'Idle capture emitted text'
                    assert not after['final']['audio_active'], 'Idle EOF marked active'
                    assert all(not s['text'] and not s['audio_active'] for s in after['updates'])
                else:
                    for key in ('text', 'tokens', 'words', 'cache_events', 'samples', 'encoded_frames'):
                        assert before['final'][key] == after['final'][key], (path.name, gain, key)
                    assert [(s['text'], s['partial_text']) for s in before['updates']] == [
                        (s['text'], s['partial_text']) for s in after['updates']], 'Partial captions differ'
                report['cases'].append({'audio': path.name, 'gain': gain,
                                        'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                        'pcm_sha256': hashlib.sha256(audio.tobytes()).hexdigest(),
                                        'seconds': len(audio) / 16000, 'before': before, 'after': after})
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                print(path.name, gain, 'passed', flush=True)
        report['completed'] = True
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        asr.SILENCE_GUARD = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idle', type=Path, required=True)
    parser.add_argument('--audio', type=Path, action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    main(parser.parse_args())
