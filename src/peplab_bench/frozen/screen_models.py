"""Small fixed learner menu. Every scaler is fitted inside the current fit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys
from pathlib import Path
import numpy as np
from xgboost import XGBRegressor
def transform(train,query,blocks):
 mu=train.mean(0);sd=train.std(0);sd=np.where(sd<1e-8,1.,sd)
 a=(train-mu)/sd;b=(query-mu)/sd
 for lo,hi in blocks:
  a[:,lo:hi]/=np.sqrt((hi-lo)*len(blocks));b[:,lo:hi]/=np.sqrt((hi-lo)*len(blocks))
 return a,b
def predict(train,labels,query,blocks,learner,param):
 assert len(train)==len(labels) and len(train)>1
 a,b=transform(np.asarray(train,float),np.asarray(query,float),blocks)
 labels=np.asarray(labels,float);center=labels.mean();spread=max(float(labels.std()),1e-8);y=(labels-center)/spread
 if learner=='ridge':
  alpha=np.linalg.solve(a@a.T+float(param)*np.eye(len(a)),y);p=b@a.T@alpha
 elif learner=='GP':
  def k(x,z):
   sq=np.maximum(0.,(x*x).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*x@z.T)
   return np.exp(-sq/(2*float(param)**2))
  matrix=k(a,a)+(.01+1e-8)*np.eye(len(a));l=np.linalg.cholesky(matrix)
  p=k(b,a)@np.linalg.solve(l.T,np.linalg.solve(l,y))
 elif learner=='XGBoost':
  model=XGBRegressor(n_estimators=100,max_depth=int(param),learning_rate=.05,subsample=1.,colsample_bytree=1.,reg_lambda=1.,min_child_weight=1,tree_method='hist',n_jobs=1,random_state=20261103,objective='reg:squarederror',device='cpu',verbosity=0)
  model.fit(a,y);p=model.predict(b)
 else:raise ValueError(learner)
 p=center+spread*np.asarray(p,float);assert np.isfinite(p).all();return p
