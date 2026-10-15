"""Replay captured production ASR with the recording's historical India clock.

This checks waiting/duty presentation, not human transcription accuracy. Capture
at least 16 seconds before and 12 seconds after the requested listener interval.
"""
import argparse
import asyncio
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from validate_pankti_timing import reset


async def replay(args):
    origin = datetime.fromisoformat(args.recording_start)
    if origin.tzinfo is None:
        raise ValueError('--recording-start needs an explicit timezone')
    epoch = origin.timestamp()
    capture = json.loads(Path(args.capture_report).read_text())
    start = float(capture['start_seconds'])
    assert start <= args.start < args.end <= start+capture['seconds']-10
    server._build_pankti_index()
    reset(server, server._bani_matcher.routes)
    server._playback_pankti_timeline = []
    server._stream_wall_started_at = epoch+start
    server.logger.setLevel('ERROR')
    sid = '11111111-1111-4111-8111-111111111111'
    server._audio_sessions[sid] = dict(last_seen=time.monotonic(), first_audio_at=epoch+start, generation=1)
    samples = []
    updates = []
    next_poll = args.start

    class Socket:
        async def __aiter__(self):
            nonlocal next_poll
            with gzip.open(args.events, 'rt', encoding='utf-8') as source:
                for raw in source:
                    message = json.loads(raw)
                    yield raw
                    received = start+float(message.get('received_until', message['window_end']))
                    updates.append(dict(received_file_seconds=received, audio_active=message.get('audio_active')))
                    while next_poll <= min(args.end, received-10):
                        state = server.live_state(sid, next_poll-start)
                        assert 'raagi' not in state and 'raagi_artwork_url' not in state
                        samples.append(dict(file_seconds=next_poll,
                            india_time=datetime.fromtimestamp(epoch+next_poll, ZoneInfo('Asia/Kolkata')).isoformat(),
                            **{key:state.get(key) for key in ['waiting_for_shabad', 'waiting_reason',
                                'pankti_id', 'pankti']}))
                        next_poll += 1

    await server._handle_asr_messages(Socket())
    assert len(samples) == int(args.end-args.start)+1, 'Incomplete listener interval'
    if args.expect_waiting_from is not None:
        assert args.start <= args.expect_waiting_from <= args.end
        assert all(row['waiting_for_shabad'] and row['pankti_id'] is None
                   for row in samples if row['file_seconds'] >= args.expect_waiting_from), 'Unexpected shabad during waiting interval'
    transitions = [dict(file_seconds=at-epoch, **line)
        for at, line in server._playback_pankti_timeline
        if args.start-16 <= at-epoch <= args.end+12]
    report = dict(recording_start=origin.isoformat(),
        clock_mapping='Filename states recording starts at 02:00 India; file offsets add to this clock.',
        capture_start_seconds=start, start_seconds=args.start, end_seconds=args.end,
        audio_sha256=capture['audio_sha256'], acoustic_settings=capture['model'],
        recognition_mode=capture['recognition_mode'], block_seconds=capture['block_seconds'],
        ardas_database_lines=len(server._ardas_pankti_ids),
        listener_samples=samples, transitions=transitions,
        asr_updates=updates, recognized_text=capture['final']['text'],
        human_accuracy_measured=False, no_raagi_fields_verified=True,
        waiting_verified_from=args.expect_waiting_from)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    changes=[]
    for row in samples:
        key=tuple(row[field] for field in ['waiting_for_shabad','waiting_reason','pankti_id'])
        if not changes or changes[-1][0] != key:
            changes.append((key,row))
    print(json.dumps(dict(listener_samples=len(samples),changes=[row for _,row in changes]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', required=True)
    parser.add_argument('--capture-report', required=True)
    parser.add_argument('--recording-start', required=True)
    parser.add_argument('--start', type=float, required=True, help='File offset in seconds')
    parser.add_argument('--end', type=float, required=True, help='File offset in seconds, inclusive')
    parser.add_argument('--report', required=True)
    parser.add_argument('--expect-waiting-from', type=float, help='Assert continuous waiting from this file offset through --end')
    asyncio.run(replay(parser.parse_args()))
