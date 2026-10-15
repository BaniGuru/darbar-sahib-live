"""Replay causal ASR messages against the current and a Git baseline matcher.

Scores exact, unambiguous word-pair following, not human transcription WER.
Listeners poll the shared timeline once a second, ten seconds behind input.
"""
import argparse
import asyncio
from bisect import bisect_right
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from bani_matching import OrderedBaniMatcher


def baseline_module(revision):
    root=Path(server.__file__).parent
    source=subprocess.check_output(['git','show',f'{revision}:server.py'],cwd=root,text=True)
    module=types.ModuleType('pankti_baseline')
    module.__file__=str(root/'server.py');sys.modules[module.__name__]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def reset(module, routes):
    for name,value in dict(_live_match=None,_live_match_at=0,_active_shabad_id=None,
            _pending_shabad_id=None,_pending_shabad_hits=0,_pending_shabad_update=None,
            _pending_pankti_id=None,_pending_pankti_hits=0,_pending_pankti_audio_time=None,
            _debug_final_text='',_debug_partial_text='',_debug_final_words=[],_debug_partial_words=[],
            _debug_timed_final_words=[],_debug_timed_partial_words=[],_debug_recompute_from=None,
            _pankti_timeline=[],_pankti_display_timeline=[],_display_dirty_from=None,
            _last_greeting_key=None,_greeting_audio_cutoff=None,_stream_wall_started_at=100000,
            _audio_sessions={},_audio_subscribers=set()).items():
        setattr(module,name,value)
    module._bani_matcher=OrderedBaniMatcher(routes,module._token_similarity,module._normalise_query)


async def replay(module, events, start, seconds):
    module._build_pankti_index();reset(module,module._bani_matcher.routes)
    module.logger.setLevel('ERROR');polls=[];costs=[];next_poll=start;last_log=time.perf_counter()
    sid='11111111-1111-4111-8111-111111111111'
    module._audio_sessions[sid]=dict(last_seen=time.monotonic(),first_audio_at=100000+start,generation=1)
    class Socket:
        async def __aiter__(self):
            nonlocal next_poll,last_log
            with gzip.open(events,'rt',encoding='utf-8') as source:
                for raw in source:
                    message=json.loads(raw);before=time.perf_counter()
                    yield raw
                    costs.append(time.perf_counter()-before)
                    if time.perf_counter()-last_log > 20:
                        print(module.__name__,message['window_end'], 'seconds, max update', round(max(costs),3),flush=True)
                        last_log=time.perf_counter()
                    available=float(message.get('received_until',message['window_end']))-10
                    while next_poll <= min(available,start+seconds):
                        state=module.live_state(sid,next_poll-start)
                        lines=state.get('display_panktis') or ([dict(pankti_id=state['pankti_id'])] if state['pankti_id'] else [])
                        polls.append(dict(audio_seconds=next_poll,pankti_id=state['pankti_id'],
                                          display_ids=[line['pankti_id'] for line in lines],
                                          pankti_timestamp=state['pankti_timestamp'],
                                          playhead_timestamp=state['playhead_timestamp']))
                        if state['pankti_timestamp'] is not None:
                            assert state['pankti_timestamp'] <= state['playhead_timestamp']
                        next_poll+=1
    before=time.perf_counter();await module._handle_asr_messages(Socket());elapsed=time.perf_counter()-before
    playback_timeline = (module._playback_pankti_timeline if hasattr(module, '_playback_pankti_timeline')
                         else module._pankti_timeline)
    timeline=[(at-100000,line) for at,line in playback_timeline]
    rows={row['pankti_id']:row for row in module._pankti_rows}
    transitions=[dict(audio_seconds=at,**line,speech=rows.get(line.get('pankti_id'),{}).get('speech',''))
                 for at,line in timeline if start <= at <= start+seconds]
    return dict(elapsed_seconds=elapsed,updates=len(costs),
                update_seconds=dict(median=statistics.median(costs),max=max(costs),p95=sorted(costs)[int(.95*(len(costs)-1))]),
                polls=polls,paired_polls=sum(len(p['display_ids'])==2 for p in polls),
                distinct_observed_panktis=len({line['pankti_id'] for at,line in timeline
                    if start<=at<=start+seconds and line.get('match_mode')!='bani-auto-next'}),
                transitions=transitions,timeline=timeline)


def at(timeline, seconds):
    index=bisect_right(timeline,seconds+1e-6,key=lambda item:item[0])-1
    return timeline[index][1] if index>=0 else {}


def word_pair_audit(current, baseline, words, rows, start, seconds):
    shabads={}
    for row in rows:
        shabads.setdefault(row['shabad_id'],[]).append(row)
    indexes={}
    for sid,lines in shabads.items():
        pairs={}
        for row in lines:
            for first,second in zip(row['_tokens'],row['_tokens'][1:]):
                if first!=second and min(len(first),len(second))>=2:
                    pairs.setdefault((first,second),set()).add(row['pankti_id'])
        indexes[sid]={pair:next(iter(ids)) for pair,ids in pairs.items() if len(ids)==1}
    observations=[];ambiguous=[]
    for word_index,(first,second) in enumerate(zip(words,words[1:])):
        when=float(second['start'])
        if not start<=when<=start+seconds:
            continue
        now=at(current['timeline'],when);old=at(baseline['timeline'],when)
        sid=now.get('shabad_id')
        # Compare within a selected shabad both implementations agree on.
        if not sid or sid!=old.get('shabad_id'):
            continue
        pair=tuple(server._normalise_query(word['text']) for word in (first,second))
        expected=indexes.get(sid,{}).get(pair)
        if expected:
            # Naam/japahu can span the end and start of a repeated refrain;
            # an interior pair in another line is not independent evidence
            # that the other line was sung. Audit those cases separately.
            third=server._normalise_query(words[word_index+2]['text']) if word_index+2<len(words) else ''
            expected_tokens=next(row['_tokens'] for row in shabads[sid] if row['pankti_id']==expected)
            expected_triples=[tuple(expected_tokens[i:i+3]) for i in range(len(expected_tokens)-2)]
            triple=pair+(third,)
            cycles=[tuple((row['_tokens'][-3:]*2)[i:i+3]) for row in shabads[sid]
                    if row['pankti_id']!=expected and len(row['_tokens'])>=3 for i in (1,2)]
            if pair!=tuple(expected_tokens[:2]) and triple not in expected_triples and triple in cycles:
                ambiguous.append(dict(audio_seconds=when,words=' '.join(triple),excluded_pankti_id=expected))
                continue
            result=dict(audio_seconds=when,words=' '.join(pair),shabad_id=sid,expected_pankti_id=expected)
            for label,run in [('current',current),('baseline',baseline)]:
                heard=at(run['timeline'],when)
                result[label+'_on_time']=heard.get('pankti_id')==expected
                later=next((ts for ts,line in run['timeline'] if when<ts<=when+12 and line.get('pankti_id')==expected),None)
                result[label+'_delay_seconds']=0 if result[label+'_on_time'] else (later-when if later else None)
            observations.append(result)
    return dict(exact_unique_pair_checks=len(observations),
                current_on_time=sum(o['current_on_time'] for o in observations),
                baseline_on_time=sum(o['baseline_on_time'] for o in observations),checks=observations,
                ambiguous_refrain_pairs=len(ambiguous),ambiguous_cases=ambiguous,
                note='ASR-to-pankti consistency within agreed selected shabads; not human accuracy or WER.')


async def main(args):
    baseline=baseline_module(args.baseline)
    old=await replay(baseline,args.events,args.start,args.seconds)
    current=await replay(server,args.events,args.start,args.seconds)
    asr=json.loads(Path(args.asr_report).read_text())
    assert len(current['polls'])==len(old['polls'])==int(args.seconds)+1, 'Missing one-second listener samples'
    words=asr['final']['committed_words']+asr['final']['partial_words']
    audit=word_pair_audit(current,old,words,server._pankti_rows,args.start,args.seconds)
    out=dict(baseline_revision=args.baseline,asr=dict(seconds=asr['seconds'],elapsed_seconds=asr['elapsed'],
             recognition_mode=asr['recognition_mode'],block_seconds=asr['block_seconds'],model=asr['model']),
             start_seconds=args.start,seconds=args.seconds,current=current,baseline=old,audit=audit,
             human_accuracy_measured=False,events_sha256=hashlib.sha256(Path(args.events).read_bytes()).hexdigest(),
             sources_sha256={name:hashlib.sha256((Path(server.__file__).parent/name).read_bytes()).hexdigest()
                             for name in ['server.py','bani_matching.py','pankti_display.py']})
    Path(args.report).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in value.items() if k not in ['polls','transitions','timeline','checks']}
                     for key,value in [('current',current),('baseline',old),('audit',audit)]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',required=True);parser.add_argument('--asr-report',required=True)
    parser.add_argument('--report',required=True);parser.add_argument('--baseline',default='f48d81a')
    parser.add_argument('--start',type=float,default=16);parser.add_argument('--seconds',type=float,required=True)
    asyncio.run(main(parser.parse_args()))
