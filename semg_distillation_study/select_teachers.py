"""Fit six fresh teacher families in each of five source-person folds."""
import argparse,datetime,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from models import FAMILIES
from data import prepare_data,sha
from selection import BASE,save,write_csv,person_folds,normalized_fold,fit_model,baselines,rank_families,row_hash,verify_proof

CODE=('models.py','data.py','training.py','selection.py','select_teachers.py')
DEFAULT_OUT=BASE/'runs/teacher_personcv_v1'

def verify_tree(folder,manifest):
 for name,digest in manifest['output_sha256'].items():
  if sha(Path(folder)/name)!=digest:raise ValueError('Fold/run artifact changed:'+name)

def configuration(args):
 prior=BASE/'runs/teachers_source30_v1/split.json'
 split=json.loads(prior.read_text(encoding='utf-8'))
 return dict(code_sha256={n:sha(BASE/n) for n in CODE},features_sha256=sha(args.features),prior_split_sha256=sha(prior),
  source_people=split['source_people'],unused_people=split['unused_people'],families=list(FAMILIES),
  training=dict(seed=20261113,max_epochs=12,min_epochs=6,patience=4,batch_size=512,lr=.001,weight_decay=.0001),
  fold_seed_rule='20261113+1009*fold_index; shared across all six families',n_folds=5,n_train_people_each=24,n_validation_people_each=6,
  threads=args.threads,jobs=args.jobs,device='cpu',precision='float32',
  ranking_rule='30 equally weighted person MAE,then macroRMSE,parameter count,registry order',
  checkpoint_rule='minimum6 executed epochs;best validation macroMAE;4 stale epochs stop;12 epoch cap',
  controls='source mean constant and flattened252-feature Ridge alpha100 with fitted intercept;train24 only',
  scope='Exploratory source-development selection;checkpoint/family selection uses these validation people;NOT independent test10/20',
  full30_pretrained_initialization=False,student_training=False)

def prepare(args):
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);config=configuration(args);protocol=out/'protocol.json'
 if protocol.exists():
  if json.loads(protocol.read_text(encoding='utf-8'))['configuration']!=config:raise ValueError('Changed person-CV protocol/code/input; use new run directory')
 else:save(protocol,dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=config))
 d=prepare_data(args.features,args.cache);folds=person_folds(config['source_people'],d['subject']);splits=[]
 for fold in folds:
  ti=fold['train_indices'];vi=fold['validation_indices']
  splits.append(dict(fold=fold['fold'],train_people=fold['train_people'],validation_people=fold['validation_people'],n_train_rows=len(ti),n_validation_rows=len(vi),
   train_rows_sha256=row_hash(d['row_ids'][ti]),validation_rows_sha256=row_hash(d['row_ids'][vi]),
   train_trials=sorted(set(d['trial'][ti])),validation_trials=sorted(set(d['trial'][vi]))))
 value=dict(source_people=config['source_people'],unused_people=config['unused_people'],folds=splits,sequence_cache=d['cache_manifest'])
 split_path=out/'splits.json'
 if split_path.exists():
  if json.loads(split_path.read_text(encoding='utf-8'))!=value:raise ValueError('Person splits changed')
 else:save(split_path,value)
 return out,config,d,folds,splits

def worker(args):
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 out,config,d,folds,splits=prepare(args);fold=folds[args.fold];folder=out/f'fold{args.fold}';folder.mkdir(exist_ok=True)
 if (folder/'complete.json').exists():
  proof=json.loads((folder/'complete.json').read_text(encoding='utf-8'));verify_tree(folder,proof);print('VERIFIED_FOLD_SKIP',args.fold,flush=True);return
 started=time.perf_counter();X,y,V,yv,pv,scaler=normalized_fold(d,fold);vi=fold['validation_indices']
 meta=dict(source_people=fold['train_people'],validation_people=fold['validation_people'],fold=args.fold,
  source_features_sha256=config['features_sha256'],models_code_sha256=config['code_sha256']['models.py'],
  configuration_sha256=sha(out/'protocol.json'),splits_sha256=sha(out/'splits.json'),
  train_rows_sha256=splits[args.fold]['train_rows_sha256'],validation_rows_sha256=splits[args.fold]['validation_rows_sha256'],
  initialization='fresh deterministic random weights;no30person pretrained checkpoint')
 training=dict(config['training'],seed=config['training']['seed']+1009*args.fold)
 baseline=baselines(folder/'baselines',X,y,V,yv,pv,scaler,d['row_ids'][vi])
 summaries=[];per=[]
 for family in FAMILIES:
  result=fit_model(folder/family,family,'teacher',X,y,V,yv,pv,scaler,training,meta,row_ids=d['row_ids'][vi])
  summaries.append(dict(fold=args.fold,**result['summary']));per.extend(dict(fold=args.fold,family=family,**r) for r in result['per_subject'])
  print('FINISHED_FAMILY',args.fold,family,result['summary']['mae'],flush=True)
 write_csv(folder/'summary.csv',summaries);write_csv(folder/'per_subject.csv',per)
 files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='complete.json' and p.suffix!='.tmp'}
 # Nested completion proofs are also protected, unlike the root proof itself.
 for family in FAMILIES:files[family+'/complete.json']=sha(folder/family/'complete.json')
 files['baselines/complete.json']=sha(folder/'baselines/complete.json')
 save(folder/'complete.json',dict(status='complete',fold=args.fold,n_models=6,summary=summaries,per_subject=per,baseline_per_subject=baseline['per_subject'],
  elapsed_s=time.perf_counter()-started,output_sha256=files))
 print('FOLD_COMPLETE',args.fold,flush=True)

def aggregate(out,config):
 summaries=[];people=[];controls=[]
 for k in range(5):
  folder=out/f'fold{k}';proof=json.loads((folder/'complete.json').read_text(encoding='utf-8'));verify_tree(folder,proof)
  summaries.extend(proof['summary']);people.extend(proof['per_subject']);controls.extend(dict(fold=k,**r) for r in proof['baseline_per_subject'])
 ranking=rank_families(people,config['source_people']);write_csv(out/'fold_summary.csv',summaries);write_csv(out/'per_subject.csv',people);write_csv(out/'baseline_per_subject.csv',controls);write_csv(out/'ranking.csv',ranking)
 selected=[r['family'] for r in ranking[:3]]
 epoch_budget={f:int(np.median([r['selected_epoch'] for r in summaries if r['family']==f])) for f in selected}
 save(out/'selection.json',dict(status='complete',selected_families=selected,ranking=ranking,final_teacher_epoch_budget=epoch_budget,
  source_people=config['source_people'],unused_people=config['unused_people'],protocol_sha256=sha(out/'protocol.json'),
  selection_scope='Fivefold24/6source-person DEVELOPMENT selection;selected checkpoints and families evaluated on same validation;not independent confirmation',
  independent_test10or20_executed=False,student_training_executed=False))
 files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('manifest.json','progress.json') and p.suffix not in ('.tmp','.log')}
 save(out/'manifest.json',dict(status='complete',n_folds=5,n_families=6,n_trained_models=30,configuration=config,output_sha256=files,
  source_person_cv_executed=True,new_person_test10or20_executed=False,student_training_executed=False))
 state_path=BASE/'research_state.json';state=json.loads(state_path.read_text(encoding='utf-8'))
 state['executed'].update(teacher_ranking=True,source_person_cross_validation=True)
 state['teacher_selection_run']=dict(directory=out.relative_to(BASE).as_posix(),status='complete',selected_families=selected,scope='source-person development validation,not independent10/20test')
 for task in state['tasks']:
  if task['id']=='source_only_six_model_selection':task.update(status='complete',evidence=out.relative_to(BASE).as_posix()+'/selection.json')
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();state['next_action']='Audit person-CV selection,then paired small direct/KD fits using only matching24-person teachers;fresh10/20test remains external.';save(state_path,state)
 print('ALL_30_PERSON_CV_MODELS_COMPLETE',json.dumps(ranking),flush=True)

def main(args):
 if args.fold is not None:return worker(args)
 out,config,d,folds,splits=prepare(args);del d,folds,splits
 if args.prepare_only:print('PERSON_CV_PROTOCOL_FROZEN',out,flush=True);return
 if (out/'manifest.json').exists():
  manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
  if manifest.get('status')=='complete':verify_tree(out,manifest);print('VERIFIED_COMPLETE_PERSON_CV_SKIP');return
 save(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cpu_count=os.cpu_count(),platform=platform.platform(),threads_per_worker=args.threads,concurrent_workers=args.jobs,cuda=False))
 state=json.loads((BASE/'research_state.json').read_text(encoding='utf-8'));state['teacher_selection_run']=dict(directory=out.relative_to(BASE).as_posix(),status='running',expected_models=30,source_only=True)
 save(BASE/'research_state.json',state)
 waiting=list(range(5));active={};completed=[];last_count=-1;logs={}
 try:
  while waiting or active:
   while waiting and len(active)<args.jobs:
    k=waiting.pop(0);log=(out/f'fold{k}.log').open('a',encoding='utf-8');logs[k]=log
    command=[sys.executable,'-B',str(BASE/'select_teachers.py'),'--fold',str(k),'--out',str(out),'--cache',str(Path(args.cache).resolve()),'--features',str(Path(args.features).resolve()),'--threads',str(args.threads),'--jobs',str(args.jobs)]
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads))
    active[k]=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   for k,p in list(active.items()):
    code=p.poll()
    if code is None:continue
    logs[k].close();del active[k]
    if code:raise RuntimeError(f'Fold{k}failed,exit{code};read{out/f"fold{k}.log"}')
    completed.append(k)
   n=sum((out/f'fold{k}'/family/'complete.json').exists() for k in range(5) for family in FAMILIES)
   save(out/'progress.json',dict(status='running',completed_models=n,expected_models=30,completed_folds=completed,active_folds=sorted(active),waiting_folds=waiting))
   if n!=last_count:print('PERSON_CV_PROGRESS',n,'/30',flush=True);last_count=n
   if active:time.sleep(3)
  aggregate(out,config);save(out/'progress.json',dict(status='complete',completed_models=30,expected_models=30))
 finally:
  for p in active.values():
   if p.poll() is None:p.terminate()
  for log in logs.values():log.close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--cache',type=Path,default=BASE.parent.parent/'work/sequence_cache')
 p.add_argument('--features',type=Path,default=BASE.parent/'semg_experiment/data/features.npz');p.add_argument('--threads',type=int,default=4);p.add_argument('--jobs',type=int,default=3)
 p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
 if not 1<=args.jobs<=5 or not 1<=args.threads<=16:p.error('Use1–5jobs and1–16threads')
 main(args)
