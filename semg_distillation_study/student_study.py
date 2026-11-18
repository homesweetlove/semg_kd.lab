"""Paired direct/output-KD students and frozen-epoch final source30 fits."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from models import make_model,parameter_count
from data import sha,prepare_data,fit_scaler,normalize_X
from training import atomic_torch,checkpoint,load_checkpoint,predict,metric_rows,macro
from selection import BASE,save,write_csv,read_csv,digest_obj,weight_hash,row_hash,verify_proof,validate_teacher,person_folds,normalized_fold,fit_model,one_epoch

def teacher_targets(folder,path,train_people,validation_people,family,scaler,X):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=Path(path)
 pack=torch.load(path,map_location='cpu',weights_only=True);validate_teacher(pack['metadata'],train_people,validation_people,family)
 if pack['metadata']['models_code_sha256']!=sha(BASE/'models.py'):raise ValueError('Teacher architecture fingerprint mismatch')
 for name,value in scaler.items():
  if not torch.equal(pack['scaler'][name],torch.as_tensor(value)):raise ValueError('Teacher/student normalization mismatch')
 signature=digest_obj(dict(teacher_sha256=sha(path),input_sha256=row_hash_bytes(X.numpy()),train_people=list(train_people),validation_people=list(validation_people)))
 if (folder/'complete.json').exists():
  proof=verify_proof(folder)
  if proof['signature']!=signature:raise ValueError('Cached teacher target provenance changed')
  with np.load(folder/'targets.npz',allow_pickle=False) as z:target=torch.from_numpy(z['prediction'].copy())
  return target,proof
 model=make_model(family,'teacher').eval();model.requires_grad_(False);model.load_state_dict(pack['model_state']);before=weight_hash(model.state_dict());start=time.perf_counter()
 target=predict(model,X).astype(np.float32);seconds=time.perf_counter()-start
 if before!=weight_hash(model.state_dict()):raise ValueError('Frozen teacher weights changed')
 np.savez_compressed(folder/'targets.npz',prediction=target)
 proof=dict(status='complete',signature=signature,n_training_targets=len(target),teacher_sha256=sha(path),target_generation_seconds=seconds,
  teacher_weights_unchanged=True,output_sha256={'targets.npz':sha(folder/'targets.npz')})
 save(folder/'complete.json',proof);return torch.from_numpy(target),proof

def row_hash_bytes(array):
 import hashlib
 return hashlib.sha256(np.asarray(array,dtype='<f4').tobytes()).hexdigest()

def fit_final(folder,family,role,epochs,X,y,scaler,metadata,seed=20261113,targets=None,weight=0.):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 signature=digest_obj(dict(family=family,role=role,epochs=epochs,metadata=metadata,seed=seed,weight=weight))
 if (folder/'complete.json').exists():
  proof=verify_proof(folder)
  if proof['signature']!=signature:raise ValueError('Final model configuration changed')
  return proof
 torch.manual_seed(seed);model=make_model(family,role);initial=weight_hash(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 begin=1;history=[];seconds=0.
 if (folder/'last.pt').exists():
  pack=load_checkpoint(folder/'last.pt',model,optimizer);m=pack['metadata']
  if m['fit_signature']!=signature:raise ValueError('Final resume fingerprint mismatch')
  begin=m['epoch']+1;history=m['epoch_rows'];seconds=m['optimization_seconds']
 for epoch in range(begin,epochs+1):
  lr=1e-5+(.001-1e-5)*.5*(1+np.cos(np.pi*(epoch-1)/max(epochs-1,1)))
  for g in optimizer.param_groups:g['lr']=float(lr)
  loss,elapsed=one_epoch(model,optimizer,X,y,epoch,512,seed,targets,weight);seconds+=elapsed
  history.append(dict(epoch=epoch,lr=float(lr),train_standardized_loss=loss,optimization_seconds=elapsed))
  meta=dict(**metadata,family=family,role=role,epoch=epoch,fit_signature=signature,epoch_rows=history,optimization_seconds=seconds,
   initial_weights_sha256=initial,distillation_weight=weight,initialization_seed=seed,epoch_rule='fixed source-CV median selected epoch;no final evaluation labels')
  atomic_torch(folder/'last.pt',checkpoint(model,optimizer,scaler,meta));write_csv(folder/'epochs.csv',history)
  print('FINAL',family,role,'KDweight',weight,'epoch',epoch,'/',epochs,flush=True)
 # Resume at the exact final epoch also rebuilds inference/derived artifacts.
 pack=load_checkpoint(folder/'last.pt',model);atomic_torch(folder/'best.pt',checkpoint(model,None,scaler,pack['metadata']));write_csv(folder/'epochs.csv',history)
 p=predict(model,X[:64])*scaler['ystd']+scaler['ymean'];np.savez_compressed(folder/'reload_examples.npz',X=X[:64].numpy(),prediction=p)
 proof=dict(status='complete',signature=signature,family=family,role=role,distillation_weight=weight,epochs_executed=epochs,parameters=parameter_count(model),
  optimization_seconds=seconds,initial_weights_sha256=initial,final_weights_sha256=weight_hash(model.state_dict()),independent_test_executed=False,
  output_sha256={name:sha(folder/name) for name in ('best.pt','last.pt','epochs.csv','reload_examples.npz')})
 save(folder/'complete.json',proof);return proof

def protocol(args):
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);parent=Path(args.teachers).resolve()
 selected=json.loads((parent/'selection.json').read_text(encoding='utf-8'));manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
 if manifest['status']!='complete' or manifest['n_trained_models']!=30:raise ValueError('Finish all30fold teachers before students')
 from select_teachers import verify_tree
 verify_tree(parent,manifest)
 config=dict(code_sha256={n:sha(BASE/n) for n in ('models.py','data.py','training.py','selection.py','student_study.py')},
  teacher_manifest_sha256=sha(parent/'manifest.json'),selection_sha256=sha(parent/'selection.json'),features_sha256=sha(args.features),
  source_people=selected['source_people'],unused_people=selected['unused_people'],selected_families=selected['selected_families'],
  training=manifest['configuration']['training'],threads=args.threads,jobs=args.jobs,distillation_weight=.5,weight_rule='fixed before student fits;no lambda search',
  student_pairing='same architecture,initial seed,batch order,maximum budget and validation checkpoint rule;fold-specific24-person teacher',
  final_epoch_rule='per-family/per-method median selected epoch across5source folds;refit all source30records without heldout selection',
  scope='source-development paired comparison;teacher/family/epoch selection uses these folds;not independent10/20test;one seed')
 path=out/'protocol.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf-8'))['configuration']!=config:raise ValueError('Student protocol/code/input changed;new run required')
 else:save(path,dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=config))
 return out,parent,selected,config

def fold_worker(args):
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 out,parent,selected,config=protocol(args);folder=out/f'fold{args.fold}';folder.mkdir(exist_ok=True)
 if (folder/'complete.json').exists():verify_proof(folder);return
 d=prepare_data(args.features,args.cache);fold=person_folds(selected['source_people'],d['subject'])[args.fold];ti=fold['train_indices'];vi=fold['validation_indices']
 X,y,V,yv,pv,scaler=normalized_fold(d,fold)
 training=dict(config['training'],seed=config['training']['seed']+1009*args.fold);per=[];summaries=[];costs=[]
 common=dict(source_people=fold['train_people'],validation_people=fold['validation_people'],fold=args.fold,models_code_sha256=sha(BASE/'models.py'),
  configuration_sha256=sha(out/'protocol.json'),train_rows_sha256=row_hash(d['row_ids'][ti]),validation_rows_sha256=row_hash(d['row_ids'][vi]))
 for family in selected['selected_families']:
  teacher=parent/f'fold{args.fold}'/family/'best.pt';targets,proof=teacher_targets(folder/family/'targets',teacher,fold['train_people'],fold['validation_people'],family,scaler,X)
  costs.append(dict(family=family,fold=args.fold,**proof))
  for method,weight in (('direct',0.),('kd',.5)):
   meta=dict(common,teacher_sha256=sha(teacher) if weight else None,teacher_target_sha256=sha(folder/family/'targets/targets.npz') if weight else None)
   result=fit_model(folder/family/method,family,'student',X,y,V,yv,pv,scaler,training,meta,targets if weight else None,weight,d['row_ids'][vi])
   summaries.append(dict(fold=args.fold,method=method,**result['summary']));per.extend(dict(fold=args.fold,family=family,method=method,**r) for r in result['per_subject'])
  a=torch.load(folder/family/'direct/best.pt',weights_only=True);b=torch.load(folder/family/'kd/best.pt',weights_only=True)
  if a['metadata']['initial_weights_sha256']!=b['metadata']['initial_weights_sha256']:raise ValueError('Student pair initialization mismatch')
 write_csv(folder/'summary.csv',summaries);write_csv(folder/'per_subject.csv',per);save(folder/'teacher_target_costs.json',costs)
 files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p!=folder/'complete.json' and p.suffix!='.tmp'}
 save(folder/'complete.json',dict(status='complete',summary=summaries,per_subject=per,target_costs=costs,output_sha256=files))

def final_fit(out,parent,selected,config,args,summaries):
 torch.set_num_threads(args.threads)
 d=prepare_data(args.features,args.cache);ix=np.flatnonzero(np.isin(d['subject'],selected['source_people']))
 scaler=fit_scaler(d['X'][ix],d['y'][ix],d['subject'][ix],selected['source_people'])
 X=torch.from_numpy(normalize_X(d['X'][ix],scaler));y=torch.from_numpy(np.asarray((d['y'][ix]-scaler['ymean'])/scaler['ystd'],np.float32))
 common=dict(source_people=selected['source_people'],validation_people=[],models_code_sha256=sha(BASE/'models.py'),configuration_sha256=sha(out/'protocol.json'),train_rows_sha256=row_hash(d['row_ids'][ix]))
 rows=[]
 for family in selected['selected_families']:
  folder=out/'final30'/family;epochs=selected['final_teacher_epoch_budget'][family]
  teacher=fit_final(folder/'teacher',family,'teacher',epochs,X,y,scaler,common)
  target,proof=teacher_targets(folder/'targets',folder/'teacher/best.pt',selected['source_people'],[],family,scaler,X)
  rows.append(dict(method='teacher',**{k:v for k,v in teacher.items() if k not in ('output_sha256','signature')}))
  for method,weight in (('direct',0.),('kd',.5)):
   epoch=int(np.median([r['selected_epoch'] for r in summaries if r['family']==family and r['method']==method]))
   meta=dict(common,teacher_sha256=sha(folder/'teacher/best.pt') if weight else None,teacher_target_sha256=sha(folder/'targets/targets.npz') if weight else None)
   result=fit_final(folder/method,family,'student',epoch,X,y,scaler,meta,targets=target if weight else None,weight=weight)
   rows.append(dict(method=method,**{k:v for k,v in result.items() if k not in ('output_sha256','signature')}))
 write_csv(out/'final30/summary.csv',rows)
 return rows

def aggregate(out,parent,selected,config,args):
 per=[];summaries=[];costs=[]
 for k in range(5):
  proof=verify_proof(out/f'fold{k}');per.extend(proof['per_subject']);summaries.extend(proof['summary']);costs.extend(proof['target_costs'])
 write_csv(out/'per_subject.csv',per);write_csv(out/'fold_summary.csv',summaries);save(out/'teacher_target_costs.json',costs)
 rows=[];paired=[];rng=np.random.default_rng(20261113)
 for family in selected['selected_families']:
  local={method:sorted([r for r in per if r['family']==family and r['method']==method],key=lambda r:r['subject']) for method in ('direct','kd')}
  if any(len(v)!=30 or len({r['subject'] for r in v})!=30 or {r['subject'] for r in v}!=set(selected['source_people']) for v in local.values()):raise ValueError('Student source-person metrics incomplete')
  for method,v in local.items():rows.append(dict(family=family,method=method,**macro(v)))
  delta=np.array([a['mae']-b['mae'] for a,b in zip(local['direct'],local['kd'])]);boot=delta[rng.integers(0,30,size=(5000,30))].mean(axis=1)
  paired.append(dict(family=family,mae_improvement_direct_minus_kd=float(delta.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),people_improved=int((delta>0).sum()),n_people=30,
   scope='conditional source-development paired bootstrap;selection optimism and one seed;not independent confirmation'))
 write_csv(out/'summary.csv',rows);write_csv(out/'paired_improvement.csv',paired)
 finals=final_fit(out,parent,selected,config,args,summaries)
 files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('manifest.json','progress.json') and p.suffix not in ('.log','.tmp')}
 save(out/'manifest.json',dict(status='complete',n_fold_student_fits=30,n_final_models=len(finals),selected_families=selected['selected_families'],configuration=config,output_sha256=files,
  source_development_only=True,new_person_test10or20_executed=False))
 state=json.loads((BASE/'research_state.json').read_text(encoding='utf-8'));state['executed'].update(student_training=True,student_source_person_cv=True,final30_teacher_student_fitting=True)
 state['student_study_run']=dict(directory=out.relative_to(BASE).as_posix(),status='complete',scope='source-person development only;new-person test10/20not executed')
 state['next_action']='Audit paired source-development comparisons and trained model timing; independent10/20-person testing still needs compatible data.'
 for task in state['tasks']:
  if task['id']=='distill_selected_three':task.update(status='complete',evidence=out.relative_to(BASE).as_posix()+'/manifest.json')
 save(BASE/'research_state.json',state);print('PAIRED_STUDENTS_AND_FINAL30_COMPLETE',flush=True)

def main(args):
 if args.fold is not None:return fold_worker(args)
 out,parent,selected,config=protocol(args)
 if (out/'manifest.json').exists():
  from select_teachers import verify_tree
  verify_tree(out,json.loads((out/'manifest.json').read_text(encoding='utf-8')));print('VERIFIED_STUDENT_STUDY_SKIP');return
 pending=list(range(5));active={};logs={}
 try:
  while pending or active:
   while pending and len(active)<args.jobs:
    k=pending.pop(0);log=(out/f'fold{k}.log').open('a',encoding='utf-8');logs[k]=log
    command=[sys.executable,'-B',str(BASE/'student_study.py'),'--fold',str(k),'--out',str(out),'--teachers',str(parent),'--cache',str(Path(args.cache).resolve()),'--features',str(Path(args.features).resolve()),'--threads',str(args.threads),'--jobs',str(args.jobs)]
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads))
    active[k]=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   for k,p in list(active.items()):
    code=p.poll()
    if code is None:continue
    logs[k].close();del active[k]
    if code:raise RuntimeError(f'Studentfold{k}failed;read{out/f"fold{k}.log"}')
   n=sum((out/f'fold{k}'/f/m/'complete.json').exists() for k in range(5) for f in selected['selected_families'] for m in ('direct','kd'))
   save(out/'progress.json',dict(status='running',completed_student_fits=n,expected=30,active_folds=sorted(active)))
   if active:time.sleep(3)
  aggregate(out,parent,selected,config,args);save(out/'progress.json',dict(status='complete',completed_student_fits=30,final_models=9))
 finally:
  for p in active.values():
   if p.poll() is None:p.terminate()
  for log in logs.values():log.close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teachers',type=Path,default=BASE/'runs/teacher_personcv_v1');p.add_argument('--out',type=Path,default=BASE/'runs/student_personcv_v1')
 p.add_argument('--features',type=Path,default=BASE.parent/'semg_experiment/data/features.npz');p.add_argument('--cache',type=Path,default=BASE.parent.parent/'work/sequence_cache')
 p.add_argument('--threads',type=int,default=4);p.add_argument('--jobs',type=int,default=3);p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args();main(args)
