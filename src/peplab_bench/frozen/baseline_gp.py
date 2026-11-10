"""One bounded DEVELOPMENT replay; all numerical fitting/evaluation runs on Vast."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time
R=Path(__file__).resolve().parent
import numpy as np
from scipy.special import ndtr
from .screen_models import predict as old_predict

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

