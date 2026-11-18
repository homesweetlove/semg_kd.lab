"""Exact past-only21x12 sequences and source-person-only normalization."""
from pathlib import Path
import hashlib,json
import numpy as np

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def history_indices(subject,trial,t,steps=21):
 subject=np.asarray(subject);trial=np.asarray(trial);t=np.asarray(t,float)
 if len(subject)!=len(trial) or len(t)!=len(subject) or not np.isfinite(t).all():raise ValueError('Invalid row metadata')
 ticks=np.rint(t*10).astype(np.int64)
 if not np.allclose(t,ticks/10,atol=1e-8,rtol=0):raise ValueError('Timestamps must lie on10Hz grid')
 groups={}
 for row,key in enumerate(zip(subject,trial)):groups.setdefault(key,[]).append(row)
 current=[];history=[]
 for ids in groups.values():
  ix=np.asarray(ids);tt=ticks[ix]
  if np.any(np.diff(tt)<=0):raise ValueError('Trial time must increase strictly')
  desired=tt[:,None]-np.arange(steps-1,-1,-1)[None,:]
  pos=np.searchsorted(tt,desired);safe=np.minimum(pos,len(tt)-1)
  present=(pos<len(tt))&(tt[safe]==desired)
  keep=present.all(axis=1)&(tt>=30)
  current.extend(ix[keep]);history.append(ix[safe[keep]])
 if not current:raise ValueError('No complete causal histories')
 current=np.asarray(current,np.int64);history=np.concatenate(history).astype(np.int32)
 order=np.argsort(current);return current[order],history[order]

def prepare_data(features,cache_root):
 features=Path(features);fingerprint=dict(features_sha256=sha(features),data_code_sha256=sha(__file__),steps=21,columns=12)
 key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:20]
 cache=Path(cache_root)/key;cache.mkdir(parents=True,exist_ok=True);manifest=cache/'manifest.json'
 if manifest.exists():
  saved=json.loads(manifest.read_text(encoding='utf-8'))
  if saved['configuration']!=fingerprint:raise ValueError('Different cached preprocessing configuration')
  for name,digest in saved['sha256'].items():
   if sha(cache/name)!=digest:raise ValueError('Cached sequence fingerprint mismatch')
 else:
  with np.load(features,allow_pickle=False) as z:
   current=np.array(z['X'][:,:12],dtype=np.float32);y=z['y'];subject=z['subject'];trial=z['trial'];t=z['t']
  if not np.isfinite(current).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training features/labels')
  rows,history=history_indices(subject,trial,t)
  X=current[history]
  np.save(cache/'X.npy',X,allow_pickle=False)
  np.savez_compressed(cache/'rows.npz',y=np.array(y[rows],np.float32),subject=subject[rows],trial=trial[rows],t=t[rows],row_ids=rows)
  saved=dict(configuration=fingerprint,n_rows=len(rows),input_rows=len(current),startup_or_incomplete_rows_removed=len(current)-len(rows),shape=list(X.shape),
   sha256={name:sha(cache/name) for name in ('X.npy','rows.npz')})
  manifest.write_text(json.dumps(saved,indent=2),encoding='utf-8')
 with np.load(cache/'rows.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 d['X']=np.load(cache/'X.npy',mmap_mode='r',allow_pickle=False);d['cache_manifest']=saved;d['cache_path']=str(cache)
 return d

def fit_scaler(X,y,subject,train_people):
 train=np.flatnonzero(np.isin(subject,list(train_people)))
 if not len(train) or set(np.asarray(subject)[train])!=set(train_people):raise ValueError('Missing source people')
 xx=np.asarray(X[train],np.float64);yy=np.asarray(y[train],np.float64)
 mean=xx.mean(axis=(0,1));std=xx.std(axis=(0,1));ym=float(yy.mean());ys=float(yy.std())
 return dict(mean=mean.astype(np.float32),std=np.maximum(std,1e-6).astype(np.float32),ymean=ym,ystd=max(ys,1e-6))

def normalize_X(X,scaler):return (np.asarray(X,np.float32)-scaler['mean'])/scaler['std']

def choose_source_people(people,n=30,seed=20261113):
 people=[str(p) for p in people]
 if len(set(people))!=len(people) or len(people)<n:raise ValueError('Need30 unique source people')
 order=np.random.default_rng(seed).permutation(sorted(people)).tolist()
 return order[:n],order[n:]

def source_folds(people):
 if len(people)!=30 or len(set(people))!=30:raise ValueError('Expected30 unique development people')
 return [dict(train=[p for p in people if p not in set(v.tolist())],validation=v.tolist()) for v in np.array_split(np.array(people),5)]
