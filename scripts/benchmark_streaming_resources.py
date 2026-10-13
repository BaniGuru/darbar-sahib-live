"""Profile causal streaming inference with declared Linux cgroup limits.

Run under Docker --cpus=2 --memory=4g to enforce the requested hardware
budget. The lag simulation uses measured update costs; it is not a paced
network measurement. Input is mono 16 kHz PCM16 WAV or float32 .f32.
"""
import argparse,collections,hashlib,json,os,resource,sys,time,wave
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--audio',action='append',required=True);p.add_argument('--output',required=True);p.add_argument('--seconds',type=float);args=p.parse_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
if args.seconds is not None and args.seconds <= 0:
 p.error('--seconds must be positive')
started=time.perf_counter();import server_asr_stream as asr
initialization=time.perf_counter()-started
stats=collections.defaultdict(lambda:{'calls':0,'seconds':0.0})
def measured(name,fn):
 def call(*a,**k):
  start=time.perf_counter()
  try:return fn(*a,**k)
  finally:stats[name]['calls']+=1;stats[name]['seconds']+=time.perf_counter()-start
 return call
if hasattr(asr,'IncrementalLanguageScorer'):
 asr.IncrementalLanguageScorer.extend=measured('incremental_prior_step',asr.IncrementalLanguageScorer.extend)
for name in ('preprocess_audio','run_encoder','decode_encoder_beams','token_prefix_score'):
 setattr(asr,name,measured(name,getattr(asr,name)))
if asr.kenlm_prior is not None:asr.kenlm_prior.token_score=measured('kenlm_token_score',asr.kenlm_prior.token_score)
asr.decoder_session.run=measured('decoder_run',asr.decoder_session.run)
if hasattr(asr.decoder_session,'run_hypotheses'):
 asr.decoder_session.run_hypotheses=measured('decoder_hypotheses',asr.decoder_session.run_hypotheses)
for name in ('encoder','predictor','joint'):
 part=getattr(asr.decoder_session,name,None)
 if part is not None:part.run=measured('decoder_'+name,part.run)
def distribution(values):
 if not values:return {'median':None,'p95':None,'max':None}
 return {'median':float(np.median(values)),'p95':float(np.percentile(values,95)),'max':max(values)}
report={'completed':False,'initialization_seconds':initialization,'cgroup':{name:Path('/sys/fs/cgroup/'+name).read_text().strip() for name in ('cpu.max','memory.max','memory.swap.max')},'affinity':sorted(os.sched_getaffinity(0)),'settings':{'recognition_mode':asr.RECOGNITION_MODE,'block_seconds':asr.BLOCK_SECONDS,'blas_threads':asr.BLAS_THREADS,'decoder_mode':asr.DECODER_MODE,'enable_language_model':asr.enable_language_model,'cpu_budget':asr.CPU_BUDGET,'encoder_threads':asr.ENCODER_THREADS,'beam':asr.BEAM_WIDTH,'kenlm_weight':asr.KENLM_WEIGHT if asr.kenlm_prior else 0,'corpus_weight':asr.LANGUAGE_PRIOR_WEIGHT if asr.language_prior else 0},'source_sha256':hashlib.sha256(Path(asr.__file__).read_bytes()).hexdigest(),'cases':[]}
for audio in args.audio:
 path=Path(audio)
 if path.suffix=='.f32':x=np.fromfile(path,'<f4')
 else:
  with wave.open(str(path)) as w:
   assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2
   x=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.float32)/32768
 if args.seconds is not None:x=x[:round(args.seconds*16000)]
 if not len(x):p.error('Audio must contain at least one sample')
 stats.clear();r=asr.new_recognizer();events=[];costs=[];virtual_end=0;started=time.perf_counter();last=started
 step=12800
 for i in range(0,len(x)//step*step,step):
  t=time.perf_counter();s=r.update(x[i:i+step]);cost=time.perf_counter()-t
  virtual_end=max(virtual_end,s.get('received_until',s['window_end']))+cost;costs.append(cost)
  events.append({'audio_seconds':s['window_end'],'cost':cost,'simulated_lag':virtual_end-s['window_end'],'text':s['text'],'partial_text':s['partial_text']})
  if time.perf_counter()-last>=15:
   print(path.name,s['window_end'],'audio seconds, cost',round(cost,3),'lag',round(virtual_end-s['window_end'],3),flush=True);last=time.perf_counter()
 t=time.perf_counter();f=r.finish(x[len(x)//step*step:]);finish_cost=time.perf_counter()-t;elapsed=time.perf_counter()-started
 assert f['samples']==len(x) and f['encoded_frames']==(len(x)+639)//640
 lag=[e['simulated_lag'] for e in events]
 case={'audio':path.name,'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seconds':len(x)/16000,'elapsed':elapsed,'rtf':elapsed/(len(x)/16000),'finish_cost':finish_cost,'update_cost':distribution(costs),'simulated_lag':distribution(lag),'profile':dict(stats),'final':f,'events':events,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cgroup_memory_peak_bytes':int(Path('/sys/fs/cgroup/memory.peak').read_text())}
 report['cases'].append(case);Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print('COMPLETE',path.name,case['rtf'],case['simulated_lag'],flush=True)
report['completed']=True;Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
