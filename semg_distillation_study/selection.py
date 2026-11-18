"""Person-disjoint development fitting; never uses the unused/test person."""
import csv,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from models import FAMILIES,make_model,parameter_count
from data import sha,source_folds,fit_scaler,normalize_X
from training import atomic_torch,checkpoint,load_checkpoint,predict,metric_rows,macro

BASE=Path(__file__).resolve().parent

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)

def write_csv(path,rows):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 temp.replace(path)

def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def digest_obj(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def row_hash(rows):return hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest()

def weight_hash(state):
 h=hashlib.sha256()
 for name in sorted(state):h.update(name.encode());h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def person_folds(source,subject):
 source=[str(x) for x in source];subject=np.asarray(subject)
 if len(source)!=30 or len(set(source))!=30 or not set(source)<=set(subject):raise ValueError('Need30 unique present source people')
 folds=[]
 for k,fold in enumerate(source_folds(source)):
  ti=np.flatnonzero(np.isin(subject,fold['train']));vi=np.flatnonzero(np.isin(subject,fold['validation']))
  if set(subject[ti])&set(subject[vi]) or len(set(subject[ti]))!=24 or len(set(subject[vi]))!=6:raise ValueError('Invalid person split')
  folds.append(dict(fold=k,train_people=fold['train'],validation_people=fold['validation'],train_indices=ti,validation_indices=vi))
 return folds

def rank_families(rows,people):
 expected=set(people);ranked=[]
 for family in FAMILIES:
  local=[r for r in rows if r['family']==family]
  if len(local)!=len(expected) or len({r['subject'] for r in local})!=len(local) or {r['subject'] for r in local}!=expected:raise ValueError('Missing/duplicate/extra source-person metrics:'+family)
  scores=macro(local)
  if not np.isfinite(list(scores.values())).all():raise ValueError('Nonfinite selection metric')
  ranked.append(dict(family=family,**scores,parameters=parameter_count(make_model(family,'teacher'))))
 if any(r['family'] not in FAMILIES for r in rows):raise ValueError('Unknown family')
 ranked.sort(key=lambda r:(r['mae'],r['rmse'],r['parameters'],FAMILIES.index(r['family'])))
 return [dict(rank=k+1,**r) for k,r in enumerate(ranked)]

def validate_teacher(metadata,train_people,validation_people,family):
 if metadata.get('role')!='teacher' or metadata.get('family')!=family or set(metadata.get('source_people',[]))!=set(train_people) or set(metadata.get('source_people',[]))&set(validation_people):raise ValueError('Teacher identity/family leak or mismatch')
 if set(metadata.get('validation_people',[]))!=set(validation_people):raise ValueError('Teacher validation partition mismatch')

def one_epoch(model,optimizer,X,y,epoch,batch_size,seed,targets=None,weight=0.):
 if not 0<=weight<=1 or (weight and (targets is None or targets.shape!=y.shape)):raise ValueError('Invalid distillation targets')
 model.train();order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+epoch));total=0.;start=time.perf_counter()
 for offset in range(0,len(order),batch_size):
  ix=order[offset:offset+batch_size];optimizer.zero_grad(set_to_none=True);p=model(X[ix]);loss=(p-y[ix]).square().mean()
  if weight:loss=(1-weight)*loss+weight*(p-targets[ix]).square().mean()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite optimization loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(ix)
 return total/len(y),time.perf_counter()-start

def verify_proof(folder):
 folder=Path(folder);proof=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
 if proof['status']!='complete':raise ValueError('Incomplete model')
 for name,digest in proof['output_sha256'].items():
  if sha(folder/name)!=digest:raise ValueError('Completed model fingerprint mismatch:'+str(folder/name))
 return proof

def fit_model(folder,family,role,X,y,V,yv,pv,scaler,config,metadata,targets=None,weight=0.,row_ids=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 signature=digest_obj(dict(config=config,metadata=metadata,family=family,role=role,weight=weight))
 if (folder/'complete.json').exists():
  result=verify_proof(folder)
  if result['signature']!=signature:raise ValueError('Changed completed model configuration')
  print('VERIFIED_SKIP',folder,flush=True);return result
 torch.manual_seed(config['seed']);model=make_model(family,role);initial_hash=weight_hash(model.state_dict())
 optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
 begin=1;best=float('inf');best_epoch=0;stale=0;epochs=[];optimization_s=0.;validation_s=0.
 if (folder/'last.pt').exists():
  pack=load_checkpoint(folder/'last.pt',model,optimizer);m=pack['metadata']
  if m['fit_signature']!=signature:raise ValueError('Resume code/config/split fingerprint mismatch')
  begin=m['epoch']+1;best=m['best_mae'];best_epoch=m['best_epoch'];stale=m['stale'];epochs=m['epoch_rows'];optimization_s=m['optimization_seconds'];validation_s=m['validation_seconds']
 for epoch in range(begin,config['max_epochs']+1):
  if epoch>config['min_epochs'] and stale>=config['patience']:break
  lr=1e-5+(config['lr']-1e-5)*.5*(1+np.cos(np.pi*(epoch-1)/max(1,config['max_epochs']-1)))
  for group in optimizer.param_groups:group['lr']=float(lr)
  loss,seconds=one_epoch(model,optimizer,X,y,epoch,config['batch_size'],config['seed'],targets,weight);optimization_s+=seconds
  start=time.perf_counter();p=predict(model,V)*scaler['ystd']+scaler['ymean'];metric=macro(metric_rows(yv,p,pv));validation_s+=time.perf_counter()-start
  if metric['mae']<best-1e-8:best=metric['mae'];best_epoch=epoch;stale=0;improved=True
  else:stale+=1;improved=False
  epochs.append(dict(epoch=epoch,lr=float(lr),train_standardized_mse=loss,validation_mae=metric['mae'],validation_rmse=metric['rmse'],optimization_seconds=seconds))
  m=dict(**metadata,family=family,role=role,fit_signature=signature,initial_weights_sha256=initial_hash,epoch=epoch,best_mae=best,best_epoch=best_epoch,stale=stale,
   epoch_rows=epochs,optimization_seconds=optimization_s,validation_seconds=validation_s,distillation_weight=weight)
  if improved:atomic_torch(folder/'best.pt',checkpoint(model,None,scaler,m))
  atomic_torch(folder/'last.pt',checkpoint(model,optimizer,scaler,m));write_csv(folder/'epochs.csv',epochs)
  save(folder/'progress.json',dict(status='training',family=family,role=role,epoch=epoch,best_epoch=best_epoch,best_mae=best,stale=stale))
  print(folder.name,family,role,'epoch',epoch,'MAE',round(metric['mae'],4),'best',round(best,4),'opt_s',round(seconds,1),flush=True)
 # Recreate derived reports from authoritative checkpoints, including interrupted finalization.
 write_csv(folder/'epochs.csv',epochs);selected=load_checkpoint(folder/'best.pt',model)
 p=predict(model,V)*scaler['ystd']+scaler['ymean'];per=metric_rows(yv,p,pv)
 row_ids=np.arange(len(yv),dtype=np.int64) if row_ids is None else np.asarray(row_ids,np.int64)
 np.savez_compressed(folder/'validation_predictions.npz',prediction=p,y=yv,subject=pv,row_ids=row_ids)
 write_csv(folder/'per_subject.csv',per)
 examples=min(64,len(V));np.savez_compressed(folder/'reload_examples.npz',X=V[:examples].numpy(),prediction=p[:examples],row_ids=row_ids[:examples])
 summary=dict(family=family,role=role,distillation_weight=weight,parameters=parameter_count(model),epochs_executed=len(epochs),selected_epoch=best_epoch,
  **macro(per),optimization_seconds=optimization_s,validation_seconds=validation_s,n_train_rows=len(X),n_validation_rows=len(V),n_train_people=len(metadata['source_people']),n_validation_people=len(metadata['validation_people']))
 save(folder/'progress.json',dict(status='complete',**summary))
 artifacts=('best.pt','last.pt','epochs.csv','per_subject.csv','validation_predictions.npz','reload_examples.npz','progress.json')
 result=dict(status='complete',signature=signature,summary=summary,per_subject=per,output_sha256={name:sha(folder/name) for name in artifacts})
 save(folder/'complete.json',result);return result

def normalized_fold(data,fold):
 ti=fold['train_indices'];vi=fold['validation_indices']
 scaler=fit_scaler(data['X'][ti],data['y'][ti],data['subject'][ti],fold['train_people'])
 X=torch.from_numpy(normalize_X(data['X'][ti],scaler));y=torch.from_numpy(np.asarray((data['y'][ti]-scaler['ymean'])/scaler['ystd'],np.float32))
 V=torch.from_numpy(normalize_X(data['X'][vi],scaler))
 return X,y,V,data['y'][vi],data['subject'][vi],scaler

def baselines(folder,X,y,V,yv,pv,scaler,row_ids):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return verify_proof(folder)
 start=time.perf_counter();dim=int(np.prod(X.shape[1:]));gram=np.zeros((dim+1,dim+1));rhs=np.zeros(dim+1)
 for offset in range(0,len(X),4096):
  a=np.column_stack((X[offset:offset+4096].numpy().reshape(-1,dim).astype(float),np.ones(min(4096,len(X)-offset))))
  gram+=a.T@a;rhs+=a.T@y[offset:offset+4096].numpy()
 penalty=np.eye(dim+1)*100.;penalty[-1,-1]=0.;beta=np.linalg.solve(gram+penalty,rhs)
 prediction=(V.numpy().reshape(-1,dim)@beta[:-1]+beta[-1])*scaler['ystd']+scaler['ymean'];constant=np.repeat(scaler['ymean'],len(V));per=[]
 for method,p in (('constant',constant),('ridge_alpha100',prediction)):
  per.extend(dict(family=method,**r) for r in metric_rows(yv,p,pv))
 write_csv(folder/'per_subject.csv',per);np.savez_compressed(folder/'predictions.npz',ridge=prediction,constant=constant,y=yv,subject=pv,row_ids=row_ids,beta=beta)
 proof=dict(status='complete',per_subject=per,seconds=time.perf_counter()-start,output_sha256={n:sha(folder/n) for n in ('per_subject.csv','predictions.npz')})
 save(folder/'complete.json',proof);return proof
