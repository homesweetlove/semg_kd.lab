"""Three student seeds × four KD weights, reusing proven original controls."""
import argparse,datetime,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from data import sha,prepare_data
from models import make_model
from selection import BASE,save,write_csv,read_csv,digest_obj,row_hash,weight_hash,person_folds,normalized_fold,fit_model,verify_proof
from student_study import teacher_targets
from resume_personcv import check_checkpoints
from select_teachers import verify_tree
from robustness import condition_grid,summarize,select_weights,assert_baseline_runtime

SEEDS=[20261113,20261129,20261207];WEIGHTS=[0.,.25,.5,.75]
CODE=('models.py','data.py','training.py','selection.py','student_study.py','resume_personcv.py','robustness.py','run_robustness.py')

def prepare(args):
 out=Path(args.out).resolve()
 if not out.is_relative_to(BASE):raise ValueError('Output must be inside project')
 out.mkdir(parents=True,exist_ok=True);teachers=BASE/'runs/teacher_personcv_v1';students=BASE/'runs/student_personcv_v1'
 tm=json.loads((teachers/'manifest.json').read_text(encoding='utf-8'));sm=json.loads((students/'manifest.json').read_text(encoding='utf-8'))
 for root,m in ((teachers,tm),(students,sm)):
  verify_tree(root,m)
  for name,digest in m['configuration']['code_sha256'].items():assert sha(BASE/name)==digest
 selected=json.loads((teachers/'selection.json').read_text(encoding='utf-8'));families=selected['selected_families'];grid=condition_grid(families,SEEDS,WEIGHTS)
 runtime=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform(),cpu_count=os.cpu_count())
 baseline_runtime=json.loads((BASE/'runs/runtime_guards/student_personcv_v1.json').read_text(encoding='utf-8'))
 baseline_runtime['numpy']=json.loads((BASE/'environment_versions.json').read_text(encoding='utf-8'))['packages']['numpy']
 assert_baseline_runtime(baseline_runtime,runtime)
 config=dict(code_sha256={n:sha(BASE/n) for n in CODE},features_sha256=sha(args.features),teacher_manifest_sha256=sha(teachers/'manifest.json'),baseline_student_manifest_sha256=sha(students/'manifest.json'),
  families=families,source_people=selected['source_people'],unused_people=selected['unused_people'],student_seeds=SEEDS,weights=WEIGHTS,
  training=tm['configuration']['training'],threads=args.threads,jobs=args.jobs,n_conditions=len(grid),n_reused_conditions=sum(r['reuse_baseline'] for r in grid),
  runtime=runtime,baseline_runtime_verified=True,
  teacher_mode='Fixed original fold24teachers and fixed family selection;NOTteacher-seed/fullpipeline robustness',
  statistics='n30people,not90;average pairedseed errors withinperson before conditional person bootstrap;seed variability separate',
  scope='Exploratory source-development lambda/seed sensitivity with selection reuse;NOTindependent10/20test')
 path=out/'protocol.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf-8'))['configuration']!=config:raise ValueError('Changed frozen grid/runtime/code/input;use new run')
 else:save(path,dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=config))
 return out,config,teachers,students,grid

def verify_baseline(folder,family,weight,fold,training,scaler,row_ids):
 proof=verify_proof(folder);check_checkpoints(folder);pack=torch.load(folder/'best.pt',weights_only=True,map_location='cpu');meta=pack['metadata']
 assert meta['family']==family and meta['role']=='student' and meta['distillation_weight']==weight
 assert set(meta['source_people'])==set(fold['train_people']) and set(meta['validation_people'])==set(fold['validation_people'])
 assert meta['configuration_sha256']==sha(BASE/'runs/student_personcv_v1/protocol.json')
 torch.manual_seed(training['seed']);initial=make_model(family,'student');assert weight_hash(initial.state_dict())==meta['initial_weights_sha256']
 for key in scaler:assert torch.equal(pack['scaler'][key],torch.as_tensor(scaler[key]))
 with np.load(folder/'validation_predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['row_ids'],row_ids)
 old_meta={k:meta[k] for k in ('source_people','validation_people','fold','models_code_sha256','configuration_sha256','train_rows_sha256','validation_rows_sha256','teacher_sha256','teacher_target_sha256')}
 signature=digest_obj(dict(config=training,metadata=old_meta,family=family,role='student',weight=weight))
 assert signature==proof['signature']==meta['fit_signature']
 return proof

def worker(args):
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 out,cfg,teachers,students,grid=prepare(args);folder=out/f'fold{args.fold}';folder.mkdir(exist_ok=True)
 if (folder/'complete.json').exists():verify_proof(folder);print('VERIFIED_FOLD_SKIP',args.fold,flush=True);return
 d=prepare_data(args.features,args.cache);fold=person_folds(cfg['source_people'],d['subject'])[args.fold];ti=fold['train_indices'];vi=fold['validation_indices'];X,y,V,yv,pv,scaler=normalized_fold(d,fold)
 records=[];initials={}
 for family in cfg['families']:
  teacher=teachers/f'fold{args.fold}'/family/'best.pt';target_folder=students/f'fold{args.fold}'/family/'targets'
  targets,cost=teacher_targets(target_folder,teacher,fold['train_people'],fold['validation_people'],family,scaler,X)
  assert cost['teacher_sha256']==sha(teacher)
  for condition in [r for r in grid if r['fold']==args.fold and r['family']==family]:
   seed=condition['seed'];weight=condition['weight'];training=dict(cfg['training'],seed=seed+1009*args.fold)
   local=folder/family/f's{seed}'/f'w{int(weight*100):03d}';local.mkdir(parents=True,exist_ok=True)
   if condition['reuse_baseline']:
    model_folder=students/f'fold{args.fold}'/family/('direct' if weight==0. else 'kd')
    result=verify_baseline(model_folder,family,weight,fold,training,scaler,d['row_ids'][vi]);print('REUSED',args.fold,family,seed,weight,flush=True)
   else:
    model_folder=local/'model'
    if (model_folder/'last.pt').exists():check_checkpoints(model_folder)
    meta=dict(source_people=fold['train_people'],validation_people=fold['validation_people'],fold=args.fold,models_code_sha256=sha(BASE/'models.py'),configuration_sha256=sha(out/'protocol.json'),
     train_rows_sha256=row_hash(d['row_ids'][ti]),validation_rows_sha256=row_hash(d['row_ids'][vi]),teacher_sha256=sha(teacher) if weight else None,teacher_target_sha256=sha(target_folder/'targets.npz') if weight else None)
    result=fit_model(model_folder,family,'student',X,y,V,yv,pv,scaler,training,meta,targets if weight else None,weight,d['row_ids'][vi]);check_checkpoints(model_folder)
   selected=torch.load(model_folder/'best.pt',map_location='cpu',weights_only=True);identity=selected['metadata']['initial_weights_sha256']
   key=(family,seed)
   if key in initials:assert initials[key]==identity,'Unpaired initialization'
   initials[key]=identity
   record=dict(**condition,model_folder=model_folder.relative_to(BASE).as_posix(),model_proof_sha256=sha(model_folder/'complete.json'),checkpoint_sha256=sha(model_folder/'best.pt'),summary=result['summary'],per_subject=result['per_subject'])
   save(local/'condition.json',record);records.append(record)
   print('CONDITION_COMPLETE',args.fold,family,seed,weight,result['summary']['mae'],flush=True)
 save(folder/'conditions.json',records)
 files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p!=folder/'complete.json' and p.suffix!='.tmp'}
 save(folder/'complete.json',dict(status='complete',n_conditions=len(records),records=records,output_sha256=files))

def aggregate(out,cfg):
 records=[];rows=[];fold_rows=[]
 for k in range(5):records.extend(verify_proof(out/f'fold{k}')['records'])
 expected={(r['fold'],r['family'],r['seed'],r['weight']) for r in condition_grid(cfg['families'],SEEDS,WEIGHTS)}
 assert len(records)==len(expected) and {(r['fold'],r['family'],r['seed'],r['weight']) for r in records}==expected
 for r in records:
  path=BASE/r['model_folder'];assert sha(path/'complete.json')==r['model_proof_sha256'] and sha(path/'best.pt')==r['checkpoint_sha256'];verify_proof(path)
  rows.extend(dict(family=r['family'],seed=r['seed'],weight=r['weight'],fold=r['fold'],**p) for p in r['per_subject'])
  fold_rows.append(dict(fold=r['fold'],family=r['family'],seed=r['seed'],weight=r['weight'],reused=r['reuse_baseline'],**{k:v for k,v in r['summary'].items() if k not in ('family',)}))
 result=summarize(rows,cfg['families'],SEEDS,WEIGHTS,cfg['source_people'])
 for name,values in (('per_person_seed.csv',rows),('fold_summary.csv',fold_rows),('summary.csv',result['summary']),('per_seed.csv',result['per_seed']),('paired_improvement.csv',result['paired'])):write_csv(out/name,values)
 choices=select_weights(result['summary']);save(out/'selection.json',dict(status='complete',selected_weights=choices,rule='mean source-person MAE over3studentseeds,thenRMSE,thenlowerKDweight',teacher_fixed=True,independent_test_executed=False))
 files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('manifest.json','progress.json') and p.suffix not in ('.tmp','.log')}
 save(out/'manifest.json',dict(status='complete',n_conditions=180,n_reused_conditions=30,n_new_fold_fits=150,configuration=cfg,output_sha256=files,independent_test10or20_executed=False))
 state=json.loads((BASE/'research_state.json').read_text(encoding='utf-8'));state['student_robustness_run']=dict(directory=out.relative_to(BASE).as_posix(),status='complete',teacher_fixed=True,student_seeds=3,weights=WEIGHTS,new_fits=150,reused_fits=30)
 state['next_action']='Audit student-seed/weight robustness;refit fixed source-selected students and report allsettings.Independent10/20test still requires data.';save(BASE/'research_state.json',state)
 print('ALL_180_CONDITIONS_COMPLETE',json.dumps(choices),flush=True)

def main(args):
 if args.fold is not None:return worker(args)
 out,cfg,teachers,students,grid=prepare(args)
 if args.prepare_only:
  prepare_data(args.features,args.cache);print('GRID_PROTOCOL_FROZEN',out,flush=True);return
 if (out/'manifest.json').exists():verify_tree(out,json.loads((out/'manifest.json').read_text(encoding='utf-8')));print('VERIFIED_GRID_SKIP');return
 prepare_data(args.features,args.cache)
 state=json.loads((BASE/'research_state.json').read_text(encoding='utf-8'));state['student_robustness_run']=dict(directory=out.relative_to(BASE).as_posix(),status='running',expected_conditions=180,new_fits=150,reused=30,teacher_fixed=True)
 state['next_action']='Finish3studentseeds×4weights sensitivity;do not claim teacherseed repetitions or independent90people.';save(BASE/'research_state.json',state)
 pending=list(range(5));active={};logs={};last=-1
 try:
  while pending or active:
   while pending and len(active)<args.jobs:
    k=pending.pop(0);log=(out/f'fold{k}.log').open('a',encoding='utf-8');logs[k]=log
    command=[sys.executable,'-B',str(BASE/'run_robustness.py'),'--fold',str(k),'--out',str(out),'--cache',str(Path(args.cache).resolve()),'--features',str(Path(args.features).resolve()),'--threads',str(args.threads),'--jobs',str(args.jobs)]
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads))
    active[k]=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   for k,p in list(active.items()):
    code=p.poll()
    if code is None:continue
    logs[k].close();del active[k]
    if code:raise RuntimeError(f'Robustness fold{k}failed;read fold{k}.log')
   n=len(list(out.glob('fold*/**/condition.json')))
   save(out/'progress.json',dict(status='running',completed_conditions=n,expected_conditions=180,active_folds=sorted(active),pending_folds=pending))
   if n!=last:print('GRID_PROGRESS',n,'/180',flush=True);last=n
   if active:time.sleep(3)
  aggregate(out,cfg);save(out/'progress.json',dict(status='complete',completed_conditions=180,expected_conditions=180))
 finally:
  for p in active.values():
   if p.poll() is None:p.terminate()
  for log in logs.values():log.close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=BASE/'runs/student_robustness_v2');p.add_argument('--features',type=Path,default=BASE.parent/'semg_experiment/data/features.npz');p.add_argument('--cache',type=Path,default=BASE.parent.parent/'work/sequence_cache')
 p.add_argument('--threads',type=int,default=4);p.add_argument('--jobs',type=int,default=3);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
 if not 1<=args.jobs<=5 or not 1<=args.threads<=16:p.error('Invalid worker/thread count')
 main(args)
