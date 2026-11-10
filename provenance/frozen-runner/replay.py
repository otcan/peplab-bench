"""Fully crossed development replay, remote only. No reserved responses accessible."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
import numpy as np
from scipy.special import ndtr
import baseline_gp as g
from screen_models import predict
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FITS={'GP':0,'XGBoost':0};CACHE={}
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,v):
 p=R/'attempt'/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def checktime():g.checktime()
def model(f,train,y,query,keys,kind,seed):
 key=(kind,tuple(keys),tuple(train.tolist()),tuple(y.tolist()),seed)
 if key in CACHE:return CACHE[key],True
 if kind=='GP':
  before=g.FITS['GP'];out=g.fit_gp(f,train,y,query,keys,seed);FITS['GP']+=g.FITS['GP']-before
 else:
  x=np.concatenate([f[k] for k in keys],axis=1);blocks=[];start=0
  for k in keys:blocks.append((start,start+f[k].shape[1]));start+=f[k].shape[1]
  def call(t,yy,q,d):
   checktime();FITS['XGBoost']+=1
   return predict(x[t],yy,x[q],blocks,'XGBoost',d)
  losses=[]
  for depth in [1,2,3]:
   errors=[]
   for valid in g.inner_folds(len(train),seed):
    fit=np.array([i for i in range(len(train)) if i not in set(valid.tolist())]);errors.extend(abs(call(train[fit],y[fit],train[valid],depth)-y[valid]).tolist())
   losses.append(float(np.mean(errors)))
  depth=[1,2,3][int(np.argmin(losses))];rng=np.random.default_rng(seed)
  draws=[]
  for _ in range(16):
   sample=rng.integers(0,len(train),len(train));draws.append(call(train[sample],y[sample],query,depth))
  draws=np.asarray(draws);mu=draws.mean(0);c=np.cov(draws,rowvar=False,ddof=1)
  out=(mu,c,.01*max(float(y.std()),1e-8)**2,{'depth':depth,'inner_MAE':losses,'bootstrap_members':16,'keys':keys,'uncertainty':'UNCALIBRATED_GAUSSIAN_BOOTSTRAP_APPROXIMATION'})
 mu,c,n,setting=out
 assert np.isfinite(mu).all() and np.isfinite(c).all() and np.isfinite(n)
 CACHE[key]=out
 return out,False

def joint_draws(mu,cov,n,rng):
 c=(cov+cov.T)/2;scale=max(float(np.diag(c).max()),1e-16)
 # Preserve covariance up to declared numerical jitter, fail on substantive negative spectrum.
 for jitter in [1e-12,1e-10,1e-8]:
  try:l=np.linalg.cholesky(c+scale*jitter*np.eye(len(c)));break
  except np.linalg.LinAlgError:continue
 else:raise AssertionError('Invalid posterior covariance')
 return mu[None,:]+rng.standard_normal((n,len(mu)))@l.T

def select(mu,cov,noise,visible,kind,seed):
 if kind=='greedy':return np.argsort(-mu,kind='stable')[:8].tolist()
 rng=np.random.default_rng(seed)
 if kind=='TS':return np.argsort(-joint_draws(mu,cov,1,rng)[0],kind='stable')[:8].tolist()
 if kind=='top8utility':
  draws=joint_draws(mu,cov,256,rng);best=np.tile(np.sort(visible)[-8:],(256,1));selected=[]
  for _ in range(8):
   marginal=np.maximum(draws-best[:,0,None],0.).mean(0);marginal[selected]=-np.inf;i=int(np.argmax(marginal));selected.append(i)
   best=np.sort(np.column_stack([best,draws[:,i]]),axis=1)[:,-8:]
  return selected
 assert kind in ['EI','UCB'];c=cov.copy();selected=[]
 for _ in range(8):
  sd=np.sqrt(np.maximum(0.,np.diag(c)));score=g.expected_improvement(mu,sd,max(visible)) if kind=='EI' else mu+2*sd
  score[selected]=-np.inf;i=int(np.argmax(score));selected.append(i)
  col=c[:,i].copy();c-=np.outer(col,col)/max(float(c[i,i])+noise,1e-16);c=(c+c.T)*.5
 return selected

def numerical_fixtures():
 mu=np.linspace(0.,1.,12);cov=np.eye(12)*.2;y=np.arange(8)/10
 for kind in ['greedy','EI','UCB','TS','top8utility']:
  a=select(mu,cov,.01,y,kind,123);assert len(a)==len(set(a))==8 and a==select(mu,cov,.01,y,kind,123)
 assert select(mu,cov,.01,y,'greedy',123)==list(range(11,3,-1))
 a=np.array([1.,-1.]);assert np.allclose(g.expected_improvement(a,np.zeros(2),0),[1.,0.])
 features={'S':np.arange(30.).reshape(10,3)};aa,bb=g.blocks_scaled(features,np.arange(4),np.array([4]),['S']);features['S'][9]*=100
 a2,b2=g.blocks_scaled(features,np.arange(4),np.array([4]),['S']);assert np.array_equal(aa[0],a2[0]) and np.array_equal(bb[0],b2[0])
 # Task utility differs from best-one EI: a certain sub-best candidate can improve the eighth-best slot.
 certain=np.array([.85,.7,.6,.5,.4,.3,.2,.1,0.,-.1]);c=np.zeros((10,10));u=select(certain,c,1e-8,np.array([1.,.9,.8,.7,.6,.5,.4,.3]),'top8utility',1)
 assert u[0]==0 and g.expected_improvement(certain,np.zeros(10),1.).sum()==0
 # Fit both actual implementations on synthetic arrays only; visible training scaler invariance.
 toy={'S':np.arange(60.,dtype=float).reshape(20,3)};tr=np.arange(8);q=np.arange(8,20);yy=np.sin(np.arange(8))
 for kind in ['GP','XGB_bootstrap16']:
  (m,c,n,st),_=model(toy,tr,yy,q,['S'],kind,123);assert m.shape==(12,) and c.shape==(12,12) and np.isfinite(c).all()
 save('numerical-fixtures.json',{'status':'PASS','groups':10,'synthetic_only':True,'at_UTC':stamp()});CACHE.clear();FITS.update(GP=0,XGBoost=0)

def main():
 assert str(R)=='/workspace/easylab-source-selector-cross-v1' and os.environ.get('EASYLAB_EXECUTION_SCOPE')=='vast-source-selector-cross-v1'
 assert not (R/'attempt/selections.jsonl').exists() and not (R/'attempt/final-results.json').exists(),'No rerun'
 adm=json.loads((R/'attempt/admission.json').read_text());assert adm['approved'] and adm['reserve_USD']==15 and adm['new_bound_USD']<1.30 and adm['cumulative_allocation_USD']<20
 g.DEADLINE=adm['science_deadline_epoch_s'];checktime()
 pins=json.loads((R/'pins.json').read_text())
 for n,h in pins.items():assert sha(R/n)==h,n
 p=json.loads((R/'protocol.json').read_text());meta=json.loads((R/'feature-metadata.json').read_text());ids=p['candidate_ids']
 assert ids==meta['ids'] and len(ids)==len(set(ids))==160 and not set(ids)&set(meta['reserved_ids'])
 seal=json.loads((R/'attempt/energy-feature-seal.json').read_text());assert seal['labels_read']==0 and seal['energy_features_SHA256']==sha(R/'attempt/energy-features.json')
 energy=json.loads((R/'attempt/energy-features.json').read_text());assert energy['ids']==ids and energy['status']=='PASS_ALL160_LABEL_FREE_ENERGY_FEATURES'
 sf=np.load(R/'screen-features.npz',allow_pickle=False);st=json.loads((R/'structure-features.json').read_text())
 f={'S':sf['ESM'],'C':sf['chemistry'],'K':np.asarray(st['features36'],float),'E':np.asarray(energy['values'],float)}
 assert [f[k].shape for k in 'SCKE']==[(160,320),(160,2058),(160,36),(160,2)] and all(np.isfinite(v).all() for v in f.values())
 numerical_fixtures();oracle=subprocess.Popen([sys.executable,str(R/'oracle.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 def ask(q):
  oracle.stdin.write(json.dumps(q)+'\n');oracle.stdin.flush();reply=json.loads(oracle.stdout.readline());assert reply.pop('ok'),reply;return reply
 config={x['id']:x for x in p['configurations']};assert len(config)==150
 started=time.monotonic();episodes=[];runtime=[]
 try:
  with (R/'attempt/selections.jsonl').open('x') as journal:
   for sno,seed in enumerate(p['seeds']):
    seedstart=time.monotonic();CACHE.clear();order=np.random.default_rng(seed).permutation(160).tolist();initial=order[:8]
    for arm in p['arms']:
     checktime();tick=time.monotonic();fits_before=FITS.copy();ask({'op':'start','arm':arm,'seed':seed});observed=list(initial);y=list(ask({'op':'reveal','ids':[ids[i] for i in observed]})['values']);history=[observed.copy()];settings=[]
     for rnd in range(3):
      query=np.array([i for i in range(160) if i not in set(observed)]);tr=np.array(observed);yy=np.asarray(y,float);innerseed=seed+1000+rnd
      if arm=='random':chosen=[i for i in order if i not in set(observed)][:8];means=sd=None;setting={}
      else:
       c=config[arm];(means,cov,noise,setting),cached=model(f,tr,yy,query,c['sources'],c['predictor'],innerseed)
       sd=np.sqrt(np.maximum(0,np.diag(cov)));chosen=query[select(means,cov,noise,yy,c['selector'],innerseed)].tolist();setting={**setting,'cache_reused_identical_visible_fit':cached}
      assert len(chosen)==len(set(chosen))==8 and not set(chosen)&set(observed)
      record={'at_UTC':stamp(),'arm':arm,'seed':seed,'round':rnd+1,'observed_ids':[ids[i] for i in observed],'query_ids':[ids[i] for i in query],'predictions':None if means is None else means.tolist(),'uncalibrated_sd':None if sd is None else sd.tolist(),'selected_ids':[ids[i] for i in chosen],'settings':setting,'selected_before_batch_reveal':True}
      journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
      reveal=ask({'op':'reveal','ids':record['selected_ids']});observed+=chosen;y+=reveal['values'];history.append(observed.copy());settings.append(setting)
      assert len(observed)==reveal['observed_count']==8+(rnd+1)*8
     episodes.append({'arm':arm,'seed':seed,'initial_ids':[ids[i] for i in initial],'observed_ids':[ids[i] for i in observed],'observed_values':y,'history':[[ids[i] for i in v] for v in history],'settings':settings,'elapsed_s':time.monotonic()-tick,'actual_fit_counts':{k:FITS[k]-fits_before[k] for k in FITS}})
     ask({'op':'seal','selection_journal_SHA256':sha(R/'attempt/selections.jsonl')})
     save('progress.json',{'at_UTC':stamp(),'completed_episodes':len(episodes),'planned':3020,'current':{'arm':arm,'seed':seed},'fit_counts':FITS.copy(),'elapsed_s':time.monotonic()-started,'energy_qualification':'PASS160','no_full_truth_yet':True})
    runtime.append(time.monotonic()-seedstart)
    if sno==0:
     remaining=g.DEADLINE-time.time();forecast=runtime[0]*19*1.5+300
     save('first-start-runtime-gate.json',{'at_UTC':stamp(),'actual_completed_episodes':151,'first_shared_start_s':runtime[0],'remaining_forecast_s':forecast,'actual_remaining_s':remaining,'pass':forecast<remaining,'no_subset_or_retry':True})
     assert forecast<remaining,'All150 configurations cannot fit frozen wall bound; stop before full-truth evaluation'
   assert len(episodes)==3020
  save('all-selection-seal.json',{'at_UTC':stamp(),'episodes':3020,'selection_journal_SHA256':sha(R/'attempt/selections.jsonl'),'energy_feature_seal_SHA256':sha(R/'attempt/energy-feature-seal.json'),'pins_SHA256':sha(R/'pins.json'),'all_episodes_sealed_before_full_truth':True})
  truth=ask({'op':'evaluate'})['all_development_truth'];assert set(truth)==set(ids)
  cutoff=sorted(truth.values())[-8];above={i for i in ids if truth[i]>cutoff};boundary={i for i in ids if truth[i]==cutoff}
  def count(seen):return len(set(seen)&above)+min(8-len(above),len(set(seen)&boundary))
  for ep in episodes:ep['curve']=[{'measurements':len(v),'true_top8_hits':count(v),'best_measured_pIC50':max(truth[i] for i in v)} for v in ep['history']]
  summary=[]
  for arm in p['arms']:
   es=[e for e in episodes if e['arm']==arm];assert len(es)==20;values=[e['curve'][-1]['true_top8_hits'] for e in es]
   summary.append({'arm':arm,**config.get(arm,{'sources':[],'predictor':'none','selector':'random'}),'episodes':20,'mean_top8_hits_at32':float(np.mean(values)),'sd_across_initializations':float(np.std(values,ddof=1)),'mean_recall_at32':float(np.mean(values)/8),'mean_discovery_curve':[float(np.mean([e['curve'][j]['true_top8_hits'] for e in es])) for j in range(4)],'total_model_wall_s':sum(e['elapsed_s'] for e in es),'actual_fit_counts':{k:sum(e['actual_fit_counts'][k] for e in es) for k in FITS}})
  shortlist=sorted([s for s in summary if s['arm']!='random'],key=lambda s:(-s['mean_top8_hits_at32'],s['sd_across_initializations'],len(s['sources']),s['arm']))[:3]
  contrasts=[]
  for base in p['configurations']:
   for source in 'SCKE':
    if source in base['sources']:continue
    larger=''.join(k for k in 'SCKE' if k in base['sources']+[source])+'__'+base['predictor']+'__'+base['selector'];small=base['id']
    a={e['seed']:e for e in episodes if e['arm']==small};b={e['seed']:e for e in episodes if e['arm']==larger};d=[b[z]['curve'][-1]['true_top8_hits']-a[z]['curve'][-1]['true_top8_hits'] for z in p['seeds']]
    contrasts.append({'base':small,'added_source':source,'larger':larger,'paired_mean_delta_hits':float(np.mean(d)),'wins':sum(v>0 for v in d),'ties':sum(v==0 for v in d),'losses':sum(v<0 for v in d)})
  save('final-results.json',{'status':'COMPLETE_EXPLORATORY_SOURCE_MODEL_SELECTOR_CROSS_STUDY','decision':'DEVELOPMENT_SCREEN_ONLY_NO_CONFIRMATORY_CLAIM','summary':summary,'shortlist':shortlist,'matched_source_contrasts':contrasts,'episodes':episodes,'fit_counts':FITS,'elapsed_s':time.monotonic()-started,'source_feature_dimensions':{k:v.shape[1] for k,v in f.items()},'all_selection_seal_SHA256':sha(R/'attempt/all-selection-seal.json'),'energy_features_SHA256':sha(R/'attempt/energy-features.json'),'total_cached_measurement_reveals':3020*32,'new_assays':0,'reserved_values_opened':0,'new_MD_or_docking':0,'limits':p['limits'],'at_UTC':stamp()})
  print('CROSS_STUDY_COMPLETE',len(episodes),flush=True)
 finally:oracle.stdin.close();oracle.wait(timeout=10)
if __name__=='__main__':
 try:main()
 except BaseException as e:save('failure.json',{'phase':'REPLAY','error':type(e).__name__+':'+str(e),'no_automatic_retry':True,'at_UTC':stamp()});raise
