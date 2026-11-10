"""Fully crossed development replay, remote only. No reserved responses accessible."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
import numpy as np
from scipy.special import ndtr
from . import baseline_gp as g
from .screen_models import predict
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

