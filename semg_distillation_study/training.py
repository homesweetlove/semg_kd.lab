"""Source-only teacher training utilities and safe tensor checkpoints."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def grouped_trial_split(subject,trial,people,seed=20261113):
 subject=np.asarray(subject);trial=np.asarray(trial);validation_trials=[]
 for sid in people:
  choices=sorted(np.unique(trial[subject==sid]).tolist())
  if len(choices)<2:raise ValueError('Need at least two eligible records per source person')
  local=int(hashlib.sha256((str(seed)+':'+str(sid)).encode()).hexdigest()[:16],16)
  validation_trials.append(choices[int(np.random.default_rng(local).integers(len(choices)))])
 source=np.isin(subject,people);val=source&np.isin(trial,validation_trials);train=source&~val
 ti=np.flatnonzero(train);vi=np.flatnonzero(val)
 if set(subject[ti])!=set(people) or set(subject[vi])!=set(people):raise ValueError('Source-person coverage incomplete')
 if set(trial[ti])&set(trial[vi]):raise ValueError('Whole-record split overlapped')
 return ti,vi,validation_trials

def metric_rows(y,prediction,subject):
 y=np.asarray(y,float);prediction=np.asarray(prediction,float);subject=np.asarray(subject)
 if y.shape!=prediction.shape or len(y)!=len(subject) or not len(y) or not np.isfinite(prediction).all():raise ValueError('Invalid predictions')
 rows=[]
 for sid in sorted(np.unique(subject)):
  r=prediction[subject==sid]-y[subject==sid]
  rows.append(dict(subject=str(sid),n=len(r),mae=float(np.abs(r).mean()),rmse=float(np.sqrt(np.square(r).mean()))))
 return rows

def macro(rows):return {key:float(np.mean([r[key] for r in rows])) for key in ('mae','rmse')}

def train_epoch(model,optimizer,X,y,epoch,batch_size=512,seed=20261113):
 model.train();g=torch.Generator().manual_seed(seed+epoch);order=torch.randperm(len(y),generator=g)
 loss_sum=0.;started=time.perf_counter()
 for start in range(0,len(order),batch_size):
  ix=order[start:start+batch_size];xb=X[ix];yb=y[ix]
  optimizer.zero_grad(set_to_none=True);prediction=model(xb);loss=F.mse_loss(prediction,yb)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  loss_sum+=float(loss.detach())*len(ix)
 return loss_sum/len(y),time.perf_counter()-started

def predict(model,X,batch_size=1024):
 model.eval();parts=[]
 with torch.inference_mode():
  for start in range(0,len(X),batch_size):parts.append(model(X[start:start+batch_size]).cpu().numpy())
 return np.concatenate(parts)

def atomic_torch(path,payload):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temp);temp.replace(path)

def checkpoint(model,optimizer,scaler,metadata):
 return dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
  optimizer_state=optimizer.state_dict() if optimizer is not None else None,
  scaler={k:torch.as_tensor(v).clone() for k,v in scaler.items()},metadata=metadata)

def load_checkpoint(path,model,optimizer=None):
 pack=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(pack['model_state'])
 if optimizer is not None and pack['optimizer_state'] is not None:optimizer.load_state_dict(pack['optimizer_state'])
 return pack

def latency(model,X,repeats=200,warmup=30):
 model.eval();one=X[:1].contiguous();elapsed=[]
 with torch.inference_mode():
  for i in range(warmup+repeats):
   begin=time.perf_counter_ns();model(one);end=time.perf_counter_ns()
   if i>=warmup:elapsed.append((end-begin)/1e6)
 return dict(batch_size=1,warmup=warmup,repeats=repeats,median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),
  scope='Trained model forward only; preprocessing/sensor acquisition excluded; CPU float32')
