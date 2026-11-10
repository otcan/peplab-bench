"""Remote numerical and original-first-context parity gates before new seeds."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import csv,hashlib,json,subprocess,time
import numpy as np
import replay as old
from oracle import Oracle
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert str(R)=='/workspace/easylab-repetitions200-v1' and os.environ.get('EASYLAB_EXECUTION_SCOPE')=='vast-repetitions200-v1'
 adm=json.loads((R/'attempt/admission.json').read_text());old.g.DEADLINE=adm['science_deadline_epoch_s']
 for name,digest in json.loads((R/'pins.json').read_text()).items():assert sha(R/name)==digest,name
 p=json.loads((R/'protocol.json').read_text());assert len(p['seeds'])==180 and len(p['combined_seeds'])==200 and len(p['arms'])==151
 assert not set(p['seeds'])&set(p['original_seeds']) and len(p['configurations'])==150
 # Independent protocol fixtures: no duplicate reveal, no missing-episode evaluation.
 toy={str(i):float(i) for i in range(160)};o=Oracle(toy,list(toy),['random'],[1]);o.request({'op':'start','arm':'random','seed':1})
 for request in [{'op':'evaluate'},{'op':'reveal','ids':['0']*8}]:
  try:o.request(request)
  except AssertionError:pass
  else:raise AssertionError('Visibility fixture was not rejected')
 o.request({'op':'reveal','ids':[str(i) for i in range(8)]})
 try:o.request({'op':'reveal','ids':[str(i) for i in range(8)]})
 except AssertionError:pass
 else:raise AssertionError('Duplicate reveal not rejected')
 old.numerical_fixtures()
 sf=np.load(R/'screen-features.npz',allow_pickle=False);st=json.loads((R/'structure-features.json').read_text());en=json.loads((R/'cached-energy-features.json').read_text())
 energyseal=json.loads((R/'cached-energy-feature-seal.json').read_text());assert energyseal['energy_features_SHA256']==sha(R/'cached-energy-features.json') and energyseal['labels_read']==0
 assert st['ids']==en['ids']==p['candidate_ids'] and en['status']=='PASS_ALL160_LABEL_FREE_ENERGY_FEATURES'
 f={'S':sf['ESM'],'C':sf['chemistry'],'K':np.array(st['features36'],float),'E':np.array(en['values'],float)}
 labels=json.loads((R/'development-labels.json').read_text());assert set(labels)==set(p['candidate_ids'])
 seed=p['original_seeds'][0];initial=np.random.default_rng(seed).permutation(160)[:8];query=np.array([i for i in range(160) if i not in set(initial)])
 y=np.array([labels[p['candidate_ids'][i]] for i in initial]);old.CACHE.clear();old.FITS.update(GP=0,XGBoost=0)
 config={x['id']:x for x in p['configurations']}
 cases=['S__GP__greedy','C__XGB_bootstrap16__greedy','SCK__GP__greedy','CE__XGB_bootstrap16__greedy']
 with (R/'reference-first-seed-scores.csv').open() as h:ref=list(csv.DictReader(h))
 receipt=[]
 for arm in cases:
  c=config[arm];(mu,cov,noise,setting),_=old.model(f,initial,y,query,c['sources'],c['predictor'],seed+1000)
  sd=np.sqrt(np.maximum(0,np.diag(cov)));rows=[x for x in ref if x['arm']==arm and x['round']=='1'];assert len(rows)==len(query)==152
  assert [x['candidate_id'] for x in rows]==[p['candidate_ids'][i] for i in query]
  err=float(np.max(abs(mu-np.array([float(x['predicted_pIC50']) for x in rows]))))
  errsd=float(np.max(abs(sd-np.array([float(x['uncalibrated_SD_pIC50']) for x in rows]))))
  assert err<=1e-10 and errsd<=1e-10,(arm,err,errsd)
  chosen=[p['candidate_ids'][query[j]] for j in old.select(mu,cov,noise,y,c['selector'],seed+1000)]
  expected=[x['candidate_id'] for x in sorted([x for x in rows if x['selected_in_batch']=='True'],key=lambda x:int(x['selection_order']))]
  assert chosen==expected,(arm,chosen,expected)
  receipt.append({'arm':arm,'max_mean_error':err,'max_SD_error':errsd,'selected8_exact':True})
 old.CACHE.clear()
 result={'status':'PASS','software_fixtures':True,'reference_contexts':receipt,'reference_only_fit_counts':old.FITS,
         'new_seed_endpoints_evaluated':0,'reserved_values_opened':0,'actual_libraries':{},'CPU_threads_per_worker':1,
         'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout.strip(),
         'GPU_compute_required_for_this_CPU_replay':False}
 import scipy,xgboost
 assert np.__version__=='1.26.4' and scipy.__version__=='1.13.1' and xgboost.__version__=='3.4.1'
 result['actual_libraries']={'numpy':np.__version__,'scipy':scipy.__version__,'xgboost':xgboost.__version__}
 (R/'attempt/qualification.json').write_text(json.dumps(result,indent=2)+'\n');print('QUALIFICATION_PASS',flush=True)
if __name__=='__main__':main()
