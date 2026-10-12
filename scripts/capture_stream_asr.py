"""Capture production-cadence ASR messages for repeatable matcher evaluation.

Example: python scripts/capture_stream_asr.py --audio asa.mp3 --seconds 600 --output /tmp/asa10
Writes PCM, gzip JSONL messages and a summary JSON beside the output prefix.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(args):
    import server_asr_stream as asr

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = out.with_suffix('.f32')
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', args.audio]
    if args.seconds is not None:
        if args.seconds <= 0:
            raise ValueError('--seconds must be positive')
        cmd += ['-t', str(args.seconds)]
    subprocess.run(cmd + ['-ac', '1', '-ar', '16000', '-f', 'f32le', str(pcm)], check=True)
    audio = np.memmap(pcm, dtype='<f4', mode='r')
    recognizer = asr.StreamingRecognizer()
    timings = []
    origin = last_log = time.perf_counter()
    step = asr.PARTIAL_INTERVAL_SAMPLES
    complete_samples = len(audio) // step * step
    with gzip.open(str(out) + '.jsonl.gz', 'wt', encoding='utf-8') as events:
        for start in range(0, complete_samples, step):
            before = time.perf_counter()
            state = recognizer.update(audio[start:start + step])
            timings.append(time.perf_counter() - before)
            events.write(json.dumps(state, ensure_ascii=False, separators=(',', ':')) + '\n')
            if time.perf_counter() - last_log > 30:
                print(json.dumps({'audio_seconds': state['window_end'],
                                  'total_seconds': len(audio) / 16000,
                                  'elapsed': time.perf_counter() - origin}), flush=True)
                last_log = time.perf_counter()
        # The production WebSocket flushes sub-800 ms terminal audio in the
        # final message, rather than emitting an undecoded partial snapshot.
        final = recognizer.finish(audio[complete_samples:])
        events.write(json.dumps(final, ensure_ascii=False, separators=(',', ':')) + '\n')
    elapsed = time.perf_counter() - origin
    report = {
        'audio': args.audio,
        'audio_sha256': hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
        'seconds': len(audio) / 16000, 'elapsed': elapsed,
        'rtf': elapsed / (len(audio) / 16000),
        'update_cost': {'median': float(np.median(timings)),
                        'p95': float(np.percentile(timings, 95)), 'max': max(timings)},
        'model': {'order': asr.kenlm_prior.order if asr.kenlm_prior else None,
                  'weight': asr.KENLM_WEIGHT, 'mode': asr.KENLM_MODE, 'beam': asr.BEAM_WIDTH,
                  'source_sha256': hashlib.sha256(Path(asr.__file__).read_bytes()).hexdigest()},
        'final': final, 'events': str(out) + '.jsonl.gz',
    }
    out.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('COMPLETE', out, elapsed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
