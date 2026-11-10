"""Remote fixed-seed scheduler; no outcome-based admissions or retry."""
from pathlib import Path
import datetime,hashlib,json,math,os,signal,subprocess,sys,time
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 p=R/'attempt'/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def load(p):return json.loads(p.read_text())
active={}
def halt():
 for seed,item in list(active.items()):
  proc=item['proc']
  if proc.poll() is None:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 deadline=time.monotonic()+5
 while time.monotonic()<deadline and any(x['proc'].poll() is None for x in active.values()):time.sleep(.1)
 for item in active.values():
  proc=item['proc']
  if proc.poll() is None:
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:pass
  item['log'].close()
def main():
 assert str(R)=='/workspace/easylab-repetitions200-v1' and os.environ.get('EASYLAB_EXECUTION_SCOPE')=='vast-repetitions200-v1'
 p=load(R/'protocol.json');a=load(R/'attempt/admission.json');assert a['approved'] and a['reserve_USD']==15 and a['new_bound_USD']<3 and a['cumulative_allocation_USD']<20
 assert not (R/'attempt/scheduler-started.json').exists(),'No execution restart'
 cores=len(os.sched_getaffinity(0));quota=None
 cpumax=Path('/sys/fs/cgroup/cpu.max')
 if cpumax.exists():
  q,period=cpumax.read_text().split()
  if q!='max':quota=int(q)/int(period)
 workers=min(int(a['CPU_workers']),cores,int(quota) if quota is not None else 16)
 assert 8<=workers<=16,'Insufficient actual CPU allocation'
 save('scheduler-started.json',{'at_UTC':stamp(),'pid':os.getpid(),'affinity_CPU_count':cores,'cgroup_quota_cores':quota,'workers':workers,'original_rental_deadline':a['hard_deadline_epoch_s']})
 for n,h in load(R/'pins.json').items():assert sha(R/n)==h,n
 remaining=a['science_deadline_epoch_s']-time.time();assert 0<remaining<=9000
 q=subprocess.run([sys.executable,str(R/'qualify.py')],timeout=min(600,remaining-10));assert q.returncode==0,'Original first-context parity or numerical qualification failed'
 pending=list(p['seeds']);done=[];started=time.monotonic();firstwave=set(pending[:workers]);wave_pass=False
 (R/'attempt/seeds').mkdir(exist_ok=True)
 while pending or active:
  assert time.time()<a['science_deadline_epoch_s']-180,'Science time cap (evaluation/collection reserve)'
  # Only the first wave is admitted until the measured throughput gate passes.
  while pending and len(active)<workers and (wave_pass or pending[0] in firstwave):
   seed=pending.pop(0);assert seed not in done and seed not in active
   log=(R/'attempt'/f'seed-{seed}.log').open('xb')
   proc=subprocess.Popen([sys.executable,str(R/'run_seed.py'),str(seed)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   active[seed]={'proc':proc,'log':log,'start':time.monotonic()}
  for seed,item in list(active.items()):
   rc=item['proc'].poll()
   if rc is None:
    assert time.monotonic()-item['start']<p['per_seed_wall_cap_s']+15,'Seed exceeded bound'
    continue
   item['log'].close()
   if rc!=0:
    try:os.killpg(item['proc'].pid,signal.SIGTERM)
    except ProcessLookupError:pass
   del active[seed]
   assert rc==0,f'Seed {seed} failed rc={rc}; no retry'
   seal=load(R/'attempt/seeds'/str(seed)/'selection-seal.json');assert seal['seed']==seed and seal['episodes']==151
   done.append(seed)
  if not wave_pass and firstwave<=set(done):
   wall=time.monotonic()-started;remaining=a['science_deadline_epoch_s']-time.time()
   estimate=math.ceil(len(pending)/workers)*wall*p['first_wave_forecast_factor']+180
   receipt={'at_UTC':stamp(),'completed_new_seeds':workers,'completed_new_episodes':workers*151,'actual_first_wave_wall_s':wall,
            'forecast_remaining_science_s':estimate,'actual_remaining_s':remaining,'factor':1.5,'pass':estimate<remaining}
   save('first-wave-runtime-gate.json',receipt);assert receipt['pass'],'Full fixed extension cannot fit current clock; stop without subset or retry'
   wave_pass=True
  partial=0;stages=[]
  for seed,item in active.items():
   f=R/'attempt/seeds'/str(seed)/'progress.json';s=load(f) if f.exists() else {}
   partial+=s.get('completed_episodes',0);stages.append({'seed':seed,'pid':item['proc'].pid,'completed_episodes':s.get('completed_episodes',0),'arm':s.get('current'),'elapsed_s':time.monotonic()-item['start']})
  save('progress.json',{'at_UTC':stamp(),'phase':'REPLAY','new_completed_seeds':len(done),'new_completed_episodes':len(done)*151+partial,
       'new_planned_episodes':27180,'reused_original_episodes':3020,'planned_combined_episodes':30200,
       'completed_seed_ids':sorted(done),'pending_seed_count':len(pending),'active':stages,'elapsed_s':time.monotonic()-started,
       'all_new_selections_sealed':False,'full_truth_evaluation_started':False,'first_wave_gate_pass':wave_pass,'actual_workers':workers})
  if pending or active:time.sleep(2)
 assert sorted(done)==p['seeds'] and len(done)==180
 seals=[]
 for seed in p['seeds']:
  d=R/'attempt/seeds'/str(seed);s=load(d/'selection-seal.json')
  assert s['selection_journal_SHA256']==sha(d/'selections.jsonl') and s['episodes_SHA256']==sha(d/'episodes-unevaluated.json')
  seals.append({'seed':seed,'selection_seal_SHA256':sha(d/'selection-seal.json'),'journal_SHA256':s['selection_journal_SHA256'],'episodes_SHA256':s['episodes_SHA256']})
 save('all-new-selection-seal.json',{'at_UTC':stamp(),'new_seeds':180,'new_episodes':27180,'all_completed_before_full_truth':True,
      'original_result_SHA256':sha(R/'original20-results.json'),'protocol_SHA256':sha(R/'protocol.json'),'seeds':seals})
 remaining=a['science_deadline_epoch_s']-time.time();assert remaining>10
 q=subprocess.run([sys.executable,str(R/'evaluate_extension.py')],timeout=remaining);assert q.returncode==0,'Final one-time evaluation failed'
 save('progress.json',{'at_UTC':stamp(),'phase':'COMPLETE','new_completed_seeds':180,'new_completed_episodes':27180,
      'combined_starts':200,'combined_episodes':30200,'reused_original_episodes':3020,'all_new_selections_sealed':True,
      'result_SHA256':sha(R/'attempt/final-results.json'),'elapsed_s':time.monotonic()-started})
 print('EXTENSION_COMPLETE',30200,flush=True)
def stopped(sig,frame):raise SystemExit('Remote execution stopped')
if __name__=='__main__':
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 try:main()
 except BaseException as e:
  save('failure.json',{'at_UTC':stamp(),'error':type(e).__name__+':'+str(e),'no_automatic_retry':True,'enlarged_comparison_incomplete':True});raise
 finally:halt()
