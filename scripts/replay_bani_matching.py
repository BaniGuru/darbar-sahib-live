"""Compare ordered bani following with the unchanged shabad matcher.

First capture real audio with capture_stream_asr.py. Replaying those same
causal messages isolates matcher changes from recognition changes. No human
reference is provided, so this reports matching behavior rather than WER.
"""
import argparse
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from bani_matching import OrderedBaniMatcher


def reset(routes, legacy):
    server._token_similarity.cache_clear()
    server._live_match = None
    server._live_match_at = 0
    server._active_shabad_id = None
    server._pending_shabad_id = None
    server._pending_shabad_hits = 0
    server._pending_pankti_id = None
    server._pending_pankti_hits = 0
    server._pending_pankti_audio_time = None
    server._debug_final_text = ''
    server._debug_partial_text = ''
    server._debug_timed_final_words = []
    server._debug_recompute_from = None
    server._pankti_timeline = []
    server._stream_wall_started_at = 0
    server._bani_matcher = None if legacy else OrderedBaniMatcher(
        routes, server._token_similarity, server._normalise_query)


async def replay(path, routes, legacy):
    reset(routes, legacy)
    trace, costs = [], []
    last_log = time.perf_counter()
    class Socket:
        async def __aiter__(self):
            nonlocal last_log
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for raw in f:
                    message = json.loads(raw)
                    before = time.perf_counter()
                    yield raw
                    costs.append(time.perf_counter() - before)
                    match = server._live_match
                    fields = {k: match.get(k) for k in (
                        'pankti_id', 'shabad_id', 'bani_id', 'bani_name', 'line_group',
                        'bani_position', 'match_mode', 'score'
                    )} if match else {}
                    matcher = server._bani_matcher
                    if matcher and matcher.family and matcher.selected:
                        route = matcher.routes[matcher.selected]
                        position = matcher.positions[matcher.selected]
                        fields['heard_pankti_id'] = route['lines'][position]['pankti_id']
                        fields['variant_evidence'] = dict(matcher.evidence)
                    trace.append({'audio_seconds': message['window_end'],
                                  'asr_tail': server._normalise_query(message['text']).split()[-12:],
                                  'displayed_text': match.get('punjabi', '') if match else '',
                                  **fields})
                    if time.perf_counter() - last_log > 30:
                        print('legacy' if legacy else 'current', message['window_end'], fields, flush=True)
                        last_log = time.perf_counter()
    origin = time.perf_counter()
    await server._handle_asr_messages(Socket())
    elapsed = time.perf_counter() - origin
    transitions = []
    previous = None
    for item in trace:
        key = tuple(item.get(k) for k in ('pankti_id', 'bani_id', 'bani_position', 'match_mode'))
        if key != previous:
            transitions.append(item)
        previous = key
    auto = [item for item in transitions if item.get('match_mode') == 'bani-auto-next']
    seconds = trace[-1]['audio_seconds']
    return {'elapsed_seconds': elapsed, 'rtf': elapsed / seconds,
            'updates': len(trace), 'matched_updates': sum(bool(t.get('pankti_id')) for t in trace),
            'bani_updates': sum(bool(t.get('bani_id')) for t in trace),
            'auto_next_transitions': len(auto),
            'distinct_heard_panktis': len({t['heard_pankti_id'] for t in trace if t.get('heard_pankti_id')}),
            'distinct_displayed_panktis': len({t['pankti_id'] for t in trace if t.get('pankti_id')}),
            'update_processing_seconds': {'median': float(np.median(costs)),
                                          'p95': float(np.percentile(costs, 95)), 'max': max(costs)},
            'transitions': transitions, 'trace': trace,
            'timeline': server._pankti_timeline}


async def main(args):
    server.logger.setLevel('ERROR')
    server._build_pankti_index()
    routes = server._bani_matcher.routes
    current = await replay(args.events, routes, False)
    legacy = await replay(args.events, routes, True) if args.baseline else None
    asr = json.loads(Path(args.asr_report).read_text()) if args.asr_report else None
    if asr:
        assert current['trace'][-1]['audio_seconds'] == asr['seconds']
        assert asr['final']['samples'] == round(asr['seconds'] * 16000)
    report = {'asr': asr, 'current': current, 'legacy': legacy,
              'reference_accuracy_measured': False,
              'note': 'Same real causal ASR messages; predicted lines are not counted as heard lines. Matching update coverage is not accuracy.',
              'sources_sha256': {name: hashlib.sha256((Path(server.__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ['server.py', 'bani_matching.py']}}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for name, result in [('current', current), ('legacy', legacy)]:
        if result:
            print(name, json.dumps({k:v for k,v in result.items() if k not in ['trace','transitions','timeline']}))
    print('Report:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', required=True)
    parser.add_argument('--asr-report')
    parser.add_argument('--report', required=True)
    parser.add_argument('--baseline', action='store_true')
    asyncio.run(main(parser.parse_args()))
