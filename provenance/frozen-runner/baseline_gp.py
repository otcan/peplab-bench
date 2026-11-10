"""One bounded DEVELOPMENT replay; all numerical fitting/evaluation runs on Vast."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'deps'))
import numpy as np
from scipy.special import ndtr
from screen_models import predict as old_predict

FITS={'GP':0,'XGBoost':0}
DEADLINE=None

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
    p=R/'attempt'/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def checktime():assert DEADLINE is not None and time.time()<DEADLINE-10,'Frozen execution wall bound'

def blocks_scaled(features,train,query,keys):
    aa,bb=[],[]
    for key in keys:
        x=features[key];mu=x[train].mean(0);sd=x[train].std(0);sd=np.where(sd<1e-8,1.,sd)
        aa.append((x[train]-mu)/sd/np.sqrt(x.shape[1]))
        bb.append((x[query]-mu)/sd/np.sqrt(x.shape[1]))
    return aa,bb

def kernel(a,b,length):
    result=0
    for x,z in zip(a,b):
        sq=np.maximum(0.,(x*x).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*x@z.T)
        result=result+np.exp(-sq/(2*length**2))/len(a)
    return result

def gp(features,train,y,query,keys,length,covariance=False):
    checktime();FITS['GP']+=1
    a,b=blocks_scaled(features,train,query,keys)
    center=float(y.mean());spread=max(float(y.std()),1e-8)
    l=np.linalg.cholesky(kernel(a,a,length)+(.01+1e-8)*np.eye(len(train)))
    cross=kernel(b,a,length);alpha=np.linalg.solve(l.T,np.linalg.solve(l,(y-center)/spread))
    mu=center+spread*(cross@alpha)
    if not covariance:return mu
    v=np.linalg.solve(l,cross.T)
    cov=spread**2*(kernel(b,b,length)-v.T@v);cov=(cov+cov.T)*.5
    assert np.isfinite(mu).all() and np.isfinite(cov).all() and np.min(np.diag(cov))>=-1e-8
    return mu,cov,spread**2*.01

def inner_folds(n,seed):return np.array_split(np.random.default_rng(seed).permutation(n),4)

def fit_gp(features,train,y,query,keys,seed):
    folds=inner_folds(len(train),seed);loss=[]
    for length in [.5,1.,2.]:
        errors=[]
        for valid in folds:
            fit=np.array([i for i in range(len(train)) if i not in set(valid.tolist())])
            p=gp(features,train[fit],y[fit],train[valid],keys,length)
            errors.extend(abs(p-y[valid]).tolist())
        loss.append(float(np.mean(errors)))
    length=[.5,1.,2.][int(np.argmin(loss))]
    return (*gp(features,train,y,query,keys,length,True),{'length':length,'inner_MAE':loss,'keys':keys})

def xgb(features,train,y,query,seed):
    x=features['chemistry'];loss=[];folds=inner_folds(len(train),seed)
    def call(fit,yy,q,depth):
        checktime();FITS['XGBoost']+=1
        return old_predict(x[fit],yy,x[q],[(0,x.shape[1])],'XGBoost',depth)
    for depth in [1,2,3]:
        errors=[]
        for valid in folds:
            fit=np.array([i for i in range(len(train)) if i not in set(valid.tolist())])
            errors.extend(abs(call(train[fit],y[fit],train[valid],depth)-y[valid]).tolist())
        loss.append(float(np.mean(errors)))
    depth=[1,2,3][int(np.argmin(loss))]
    return call(train,y,query,depth),{'depth':depth,'inner_MAE':loss}

def expected_improvement(mu,sd,best):
    improvement=mu-best;z=np.divide(improvement,sd,out=np.zeros_like(mu),where=sd>1e-12)
    ei=improvement*ndtr(z)+sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return np.where(sd>1e-12,np.maximum(ei,0.),np.maximum(improvement,0.))

def batch(mu,cov,noise,best,kind):
    if kind=='greedy':return np.argsort(-mu,kind='stable')[:8].tolist()
    c=cov.copy();selected=[]
    for _ in range(8):
        sd=np.sqrt(np.maximum(0,np.diag(c)))
        score=expected_improvement(mu,sd,best);score[selected]=-np.inf
        i=int(np.argmax(score));assert i not in selected
        selected.append(i)
        column=c[:,i].copy();denom=max(float(c[i,i])+noise,1e-16)
        c-=np.outer(column,column)/denom;c=(c+c.T)*.5
    return selected

def numerical_fixtures():
    # Artificial arrays only, remote-only. No molecular/old response fixtures.
    f={'chemistry':np.array([[0.,1.],[1.,0.],[2.,1.],[3.,0.],[4.,1.],[999.,-999.]])}
    tr=np.arange(4);q=np.array([4]);a,b=blocks_scaled(f,tr,q,['chemistry'])
    f2={'chemistry':f['chemistry'].copy()};f2['chemistry'][5]*=100
    a2,b2=blocks_scaled(f2,tr,q,['chemistry']);assert np.array_equal(a[0],a2[0]) and np.array_equal(b[0],b2[0])
    m,c,n=gp(f,tr,np.array([0.,1.,0.,1.]),np.array([4,5]),['chemistry'],1.,True)
    assert np.isfinite(m).all() and np.linalg.eigvalsh(c).min()>-1e-8
    assert np.allclose(expected_improvement(np.array([1.,-1.]),np.zeros(2),0.),[1.,0.])
    mu=np.linspace(0.,1.,12);cov=np.eye(12)
    s=batch(mu,cov,.01,1.,'EI');assert len(set(s))==8 and s==batch(mu,cov,.01,1.,'EI')
    assert batch(mu,cov,.01,1.,'greedy')==list(range(11,3,-1))
    save('numerical-fixtures.json',{'status':'PASS','groups':5,'synthetic_only':True,'at_UTC':stamp()})

def main():
    global DEADLINE
    assert str(R)=='/workspace/easylab-bayesian-replay-v1' and os.environ.get('EASYLAB_EXECUTION_SCOPE')=='vast-bayesian-replay-v1'
    (R/'attempt').mkdir(exist_ok=True)
    assert not (R/'attempt/selections.jsonl').exists() and not (R/'attempt/final-results.json').exists(),'No replay rerun'
    admission=json.loads((R/'attempt/admission.json').read_text());assert admission['approved'] and admission['cumulative_allocation_USD']<20 and admission['reserve_USD']==15
    DEADLINE=admission['science_deadline_epoch_s'];assert 0<DEADLINE-time.time()<=1800
    pins=json.loads((R/'pins.json').read_text())
    for n,v in pins.items():assert sha(R/n)==v,'Frozen input/code mismatch: '+n
    p=json.loads((R/'protocol.json').read_text());meta=json.loads((R/'feature-metadata.json').read_text())
    ids=meta['ids'];assert ids==p['candidate_ids'] and len(set(ids))==160 and not set(ids)&set(meta['reserved_ids'])
    f=np.load(R/'screen-features.npz',allow_pickle=False)
    st=json.loads((R/'structure-features.json').read_text());assert st['ids']==ids and st['all160_certified']
    features={'chemistry':f['chemistry'],'ESM':f['ESM'],'structure':np.array(st['features36'],float)}
    assert [features[k].shape for k in ['chemistry','ESM','structure']]==[(160,2058),(160,320),(160,36)]
    assert all(np.isfinite(v).all() for v in features.values())
    numerical_fixtures();FITS.update(GP=0,XGBoost=0)
    oracle=subprocess.Popen([sys.executable,str(R/'oracle.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def ask(q):
        oracle.stdin.write(json.dumps(q)+'\n');oracle.stdin.flush()
        reply=json.loads(oracle.stdout.readline());assert reply.pop('ok'),reply
        return reply
    started=time.monotonic();episodes=[]
    try:
        with (R/'attempt/selections.jsonl').open('x') as journal:
            for seed in p['seeds']:
                order=np.random.default_rng(seed).permutation(160).tolist();initial=order[:8]
                for arm in p['arms']:
                    checktime();ask({'op':'start','arm':arm,'seed':seed})
                    observed=list(initial);reveal=ask({'op':'reveal','ids':[ids[i] for i in observed]});y=list(reveal['values'])
                    history=[list(observed)];settings=[]
                    for roundno in range(3):
                        query=np.array([i for i in range(160) if i not in set(observed)]);train=np.array(observed);yy=np.array(y,float)
                        assert len(query)==160-len(observed)
                        innerseed=seed+1000+roundno
                        if arm=='random':
                            chosen=[i for i in order if i not in set(observed)][:8];means=sd=None;setting={}
                        elif arm=='chemistry_XGB_greedy':
                            means,setting=xgb(features,train,yy,query,innerseed);sd=None
                            chosen=query[np.argsort(-means,kind='stable')[:8]].tolist()
                        else:
                            keys=['chemistry'] if arm.startswith('chemistry') else ['chemistry','ESM','structure']
                            means,cov,noise,setting=fit_gp(features,train,yy,query,keys,innerseed)
                            sd=np.sqrt(np.maximum(0,np.diag(cov)))
                            chosen=query[batch(means,cov,noise,max(y),'EI' if arm.endswith('EI') else 'greedy')].tolist()
                        assert len(chosen)==len(set(chosen))==8 and not set(chosen)&set(observed)
                        record={'at_UTC':stamp(),'arm':arm,'seed':seed,'round':roundno+1,'observed_ids':[ids[i] for i in observed],
                                'query_ids':[ids[i] for i in query],'predictions':None if means is None else means.tolist(),
                                'uncalibrated_GP_sd':None if sd is None else sd.tolist(),'selected_ids':[ids[i] for i in chosen],
                                'settings':setting,'selected_before_batch_reveal':True}
                        journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                        reveal=ask({'op':'reveal','ids':record['selected_ids']});observed+=chosen;y+=reveal['values']
                        assert len(observed)==reveal['observed_count']==8+(roundno+1)*8
                        history.append(list(observed));settings.append(setting)
                    episode={'arm':arm,'seed':seed,'initial_ids':[ids[i] for i in initial],'observed_ids':[ids[i] for i in observed],
                             'observed_values':y,'history':[[ids[i] for i in v] for v in history],'settings':settings}
                    episodes.append(episode)
                    ask({'op':'seal','selection_journal_SHA256':sha(R/'attempt/selections.jsonl')})
                    save('progress.json',{'at_UTC':stamp(),'completed_episodes':len(episodes),'planned':120,'current':{'arm':arm,'seed':seed},'fit_counts':FITS.copy(),'elapsed_s':time.monotonic()-started})
        assert len(episodes)==120 and all(len(set(e['observed_ids']))==32 for e in episodes)
        save('all-selection-seal.json',{'at_UTC':stamp(),'completed_episodes':120,'selection_journal_SHA256':sha(R/'attempt/selections.jsonl'),'pins_SHA256':sha(R/'pins.json'),'all_episodes_sealed_before_full_truth':True})
        truth=ask({'op':'evaluate'})['all_development_truth'];assert set(truth)==set(ids)
        cutoff=sorted(truth.values())[-8];above={i for i in ids if truth[i]>cutoff};boundary={i for i in ids if truth[i]==cutoff}
        def count(seen):return len(set(seen)&above)+min(8-len(above),len(set(seen)&boundary))
        for e in episodes:
            e['curve']=[{'measurements':len(v),'true_top8_hits':count(v),'best_measured_pIC50':max(truth[i] for i in v)} for v in e['history']]
        summary=[]
        for arm in p['arms']:
            part=[e for e in episodes if e['arm']==arm];assert len(part)==20
            values=[e['curve'][-1]['true_top8_hits'] for e in part]
            summary.append({'arm':arm,'episodes':20,'mean_top8_hits_at32':float(np.mean(values)),'sd_across_initializations':float(np.std(values,ddof=1)),
                            'mean_recall_at32':float(np.mean(values)/8),'mean_discovery_curve':[float(np.mean([e['curve'][j]['true_top8_hits'] for e in part])) for j in range(4)]})
        s={v['arm']:v for v in summary};f=s['fusion_GP_EI']['mean_top8_hits_at32']
        useful=f>s['chemistry_GP_EI']['mean_top8_hits_at32'] and f>s['chemistry_XGB_greedy']['mean_top8_hits_at32']
        save('final-results.json',{'status':'COMPLETE_DEVELOPMENT_REPLAY','decision':'PROMISING_DEVELOPMENT_ONLY' if useful else 'NO_DEMONSTRATED_ADDED_VALUE_ON_DEVELOPMENT',
                                 'summary':summary,'episodes':episodes,'total_cached_measurement_reveals':3840,'new_assay_measurements':0,
                                 'all_selection_seal_SHA256':sha(R/'attempt/all-selection-seal.json'),'fit_counts':FITS,'elapsed_s':time.monotonic()-started,
                                 'reserved_values_opened':0,'no_new_docking_or_MD':True,'uncertainty':'UNCALIBRATED_GP_MODEL_UNCERTAINTY',
                                 'limits':p['limits'],'at_UTC':stamp()})
        print('BAYESIAN_DEVELOPMENT_REPLAY_COMPLETE',useful,flush=True)
    finally:
        oracle.stdin.close();oracle.wait(timeout=10)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (R/'attempt').mkdir(exist_ok=True)
        save('failure.json',{'at_UTC':stamp(),'error':type(e).__name__+':'+str(e),'no_automatic_retry':True})
        raise
