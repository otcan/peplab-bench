"""One new shared seed, all151 arms. Calls original unmodified learner/selector functions."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
import numpy as np
import replay as old
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(d,name,obj):
 p=d/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,allow_nan=False)+'\n');t.replace(p)
def main():
 seed=int(sys.argv[1]);p=json.loads((R/'protocol.json').read_text());a=json.loads((R/'attempt/admission.json').read_text())
 assert str(R)=='/workspace/easylab-repetitions200-v1' and os.environ.get('EASYLAB_EXECUTION_SCOPE')=='vast-repetitions200-v1'
 assert seed in p['seeds'] and seed not in p['original_seeds']
 d=R/'attempt/seeds'/str(seed);d.mkdir(parents=True,exist_ok=False)
 old.g.DEADLINE=min(a['science_deadline_epoch_s'],time.time()+p['per_seed_wall_cap_s'])
 old.CACHE.clear();old.FITS.update(GP=0,XGBoost=0)
 # Original Oracle.main runs unchanged with only this shard's seed admitted.
 for name in ['oracle.py','development-labels.json']:
  os.link(R/name,d/name)
 (d/'protocol.json').write_text(json.dumps({'candidate_ids':p['candidate_ids'],'arms':p['arms'],'seeds':[seed]}))
 (d/'attempt').mkdir()
 sf=np.load(R/'screen-features.npz',allow_pickle=False)
 st=json.loads((R/'structure-features.json').read_text());en=json.loads((R/'cached-energy-features.json').read_text())
 assert st['ids']==en['ids']==p['candidate_ids'] and len(p['candidate_ids'])==160
 f={'S':sf['ESM'],'C':sf['chemistry'],'K':np.asarray(st['features36'],float),'E':np.asarray(en['values'],float)}
 assert [f[k].shape for k in 'SCKE']==[(160,320),(160,2058),(160,36),(160,2)] and all(np.isfinite(v).all() for v in f.values())
 ids=p['candidate_ids'];config={c['id']:c for c in p['configurations']};assert len(config)==150
 oracle=subprocess.Popen([sys.executable,str(d/'oracle.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 def ask(q):
  oracle.stdin.write(json.dumps(q)+'\n');oracle.stdin.flush();reply=json.loads(oracle.stdout.readline());assert reply.pop('ok'),reply;return reply
 started=time.monotonic();episodes=[]
 try:
  order=np.random.default_rng(seed).permutation(160).tolist();initial=order[:8]
  with (d/'selections.jsonl').open('x') as journal:
   for arm in p['arms']:
    old.checktime();tick=time.monotonic();before=old.FITS.copy();ask({'op':'start','arm':arm,'seed':seed})
    observed=list(initial);y=list(ask({'op':'reveal','ids':[ids[i] for i in observed]})['values']);history=[observed.copy()];settings=[]
    for rnd in range(3):
     query=np.array([i for i in range(160) if i not in set(observed)]);train=np.array(observed);yy=np.asarray(y,float);innerseed=seed+1000+rnd
     if arm=='random':chosen=[i for i in order if i not in set(observed)][:8];means=sd=None;setting={}
     else:
      c=config[arm];(means,cov,noise,setting),cached=old.model(f,train,yy,query,c['sources'],c['predictor'],innerseed)
      sd=np.sqrt(np.maximum(0,np.diag(cov)));chosen=query[old.select(means,cov,noise,yy,c['selector'],innerseed)].tolist();setting={**setting,'cache_reused_identical_visible_fit':cached}
     assert len(chosen)==len(set(chosen))==8 and not set(chosen)&set(observed)
     record={'at_UTC':stamp(),'arm':arm,'seed':seed,'round':rnd+1,'observed_ids':[ids[i] for i in observed],'query_ids':[ids[i] for i in query],
             'predictions':None if means is None else means.tolist(),'uncalibrated_sd':None if sd is None else sd.tolist(),
             'selected_ids':[ids[i] for i in chosen],'settings':setting,'selected_before_batch_reveal':True}
     journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
     reveal=ask({'op':'reveal','ids':record['selected_ids']});observed+=chosen;y+=reveal['values'];history.append(observed.copy());settings.append(setting)
     assert len(observed)==reveal['observed_count']==8+(rnd+1)*8
    episodes.append({'arm':arm,'seed':seed,'initial_ids':[ids[i] for i in initial],'observed_ids':[ids[i] for i in observed],'observed_values':y,
                     'history':[[ids[i] for i in v] for v in history],'settings':settings,'elapsed_s':time.monotonic()-tick,
                     'actual_fit_counts':{k:old.FITS[k]-before[k] for k in old.FITS}})
    ask({'op':'seal','selection_journal_SHA256':sha(d/'selections.jsonl')})
    save(d,'progress.json',{'seed':seed,'at_UTC':stamp(),'completed_episodes':len(episodes),'planned':151,'current':arm,'elapsed_s':time.monotonic()-started,'fit_counts':old.FITS.copy(),'full_truth_requested':False})
  assert len(episodes)==151 and all(len(set(x['observed_ids']))==32 for x in episodes)
  save(d,'episodes-unevaluated.json',episodes)
  save(d,'selection-seal.json',{'at_UTC':stamp(),'seed':seed,'episodes':151,'selection_journal_SHA256':sha(d/'selections.jsonl'),
       'episodes_SHA256':sha(d/'episodes-unevaluated.json'),'oracle_audit_SHA256':sha(d/'attempt/oracle-audit.jsonl'),
       'elapsed_s':time.monotonic()-started,'fit_counts':old.FITS.copy(),'full_truth_evaluation_requested':False})
 finally:
  oracle.stdin.close();oracle.wait(timeout=10)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  seed=sys.argv[1];d=R/'attempt/seeds'/seed;d.mkdir(parents=True,exist_ok=True)
  save(d,'failure.json',{'at_UTC':stamp(),'error':type(e).__name__+':'+str(e),'no_retry':True});raise
