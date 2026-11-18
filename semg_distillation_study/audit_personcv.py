"""Independently reload and recompute person-CV and paired-student artifacts."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
from models import FAMILIES,make_model
from data import sha,prepare_data,fit_scaler
from training import metric_rows,macro
from selection import BASE,save,read_csv,person_folds,row_hash,weight_hash,rank_families,validate_teacher
from select_teachers import verify_tree

def benchmark(model,x,path,previous=None):
 digest=sha(path)
 if previous is not None:
  assert previous['checkpoint_sha256']==digest and previous['threads']==4 and len(previous['raw_ms'])==200
  raw=previous['raw_ms']
 else:
  raw=[]
  with torch.inference_mode():
   for i in range(230):
    start=time.perf_counter_ns();model(x);elapsed=(time.perf_counter_ns()-start)/1e6
    if i>=30:raw.append(elapsed)
 assert np.isfinite(raw).all() and min(raw)>0
 return dict(checkpoint_sha256=digest,threads=4,batch_size=1,precision='float32',device='cpu',warmup=30,repeats=200,raw_ms=raw,median_ms=float(np.median(raw)),p95_ms=float(np.quantile(raw,.95)))

def check_reloaded(folder,family,role):
 pack=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);meta=pack['metadata']
 last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)['metadata']
 for key in ('fit_signature','family','role','source_people','validation_people'):assert meta.get(key)==last.get(key)
 assert meta['epoch']==last.get('best_epoch',last['epoch'])
 assert meta['family']==family and meta['role']==role and meta['models_code_sha256']==sha(BASE/'models.py')
 model=make_model(family,role);model.load_state_dict(pack['model_state']);model.eval()
 assert weight_hash(model.state_dict())!=meta['initial_weights_sha256']
 with np.load(folder/'reload_examples.npz',allow_pickle=False) as z:
  with torch.inference_mode():p=model(torch.from_numpy(z['X'].copy())).numpy()*float(pack['scaler']['ystd'])+float(pack['scaler']['ymean'])
  difference=float(np.max(np.abs(p-z['prediction'])));assert difference<5e-4
 return pack,model,difference

def recompute_predictions(folder,expected_row_ids,data,vi):
 with np.load(folder/'validation_predictions.npz',allow_pickle=False) as z:
  np.testing.assert_array_equal(z['row_ids'],expected_row_ids);np.testing.assert_array_equal(z['subject'],data['subject'][vi]);np.testing.assert_array_equal(z['y'],data['y'][vi])
  rows=metric_rows(z['y'],z['prediction'],z['subject'])
 saved=read_csv(folder/'per_subject.csv');assert len(rows)==len(saved)
 for r,s in zip(rows,saved):
  assert r['subject']==s['subject'] and r['n']==int(s['n'])
  assert abs(r['mae']-float(s['mae']))<1e-10 and abs(r['rmse']-float(s['rmse']))<1e-10
 ep=read_csv(folder/'epochs.csv');chosen=min(ep,key=lambda e:float(e['validation_mae']))
 return rows,int(chosen['epoch'])

def audit(args):
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 teachers=Path(args.teachers);tm=json.loads((teachers/'manifest.json').read_text(encoding='utf-8'));assert tm['status']=='complete' and tm['n_trained_models']==30
 verify_tree(teachers,tm)
 assert sha(args.features)==tm['configuration']['features_sha256']
 for name,digest in tm['configuration']['code_sha256'].items():assert sha(BASE/name)==digest
 selection=json.loads((teachers/'selection.json').read_text(encoding='utf-8'));d=prepare_data(args.features,args.cache)
 assert len(selection['source_people'])==30 and not set(selection['source_people'])&set(selection['unused_people'])
 folds=person_folds(selection['source_people'],d['subject']);splits=json.loads((teachers/'splits.json').read_text(encoding='utf-8'))
 teacher_rows=[];checks=[];controls=[]
 for fold in folds:
  k=fold['fold'];ti=fold['train_indices'];vi=fold['validation_indices'];description=splits['folds'][k]
  assert row_hash(d['row_ids'][ti])==description['train_rows_sha256'] and row_hash(d['row_ids'][vi])==description['validation_rows_sha256']
  assert not set(d['subject'][ti])&set(d['subject'][vi]) and not set(d['trial'][ti])&set(d['trial'][vi])
  fitted=fit_scaler(d['X'][ti],d['y'][ti],d['subject'][ti],fold['train_people'])
  for family in FAMILIES:
   folder=teachers/f'fold{k}'/family;pack,model,error=check_reloaded(folder,family,'teacher');meta=pack['metadata'];validate_teacher(meta,fold['train_people'],fold['validation_people'],family)
   torch.manual_seed(tm['configuration']['training']['seed']+1009*k);initial=make_model(family,'teacher');assert weight_hash(initial.state_dict())==meta['initial_weights_sha256']
   for key in fitted:assert torch.equal(pack['scaler'][key],torch.as_tensor(fitted[key]))
   assert meta['configuration_sha256']==sha(teachers/'protocol.json') and meta['splits_sha256']==sha(teachers/'splits.json')
   assert meta['train_rows_sha256']==description['train_rows_sha256'] and meta['validation_rows_sha256']==description['validation_rows_sha256']
   rows,epoch=recompute_predictions(folder,d['row_ids'][vi],d,vi);assert epoch==meta['epoch'];teacher_rows.extend(dict(family=family,fold=k,**r) for r in rows)
   checks.append(dict(fold=k,family=family,role='teacher',max_reload_difference_deg=error,validated_people=len(rows),selected_epoch=epoch))
  with np.load(teachers/f'fold{k}/baselines/predictions.npz',allow_pickle=False) as z:
   np.testing.assert_array_equal(z['row_ids'],d['row_ids'][vi]);np.testing.assert_array_equal(z['y'],d['y'][vi])
   for name,key in (('constant','constant'),('ridge_alpha100','ridge')):controls.extend(dict(family=name,fold=k,**r) for r in metric_rows(z['y'],z[key],z['subject']))
 ranked=rank_families(teacher_rows,selection['source_people'])
 for a,b in zip(ranked,selection['ranking']):assert a['family']==b['family'] and abs(a['mae']-b['mae'])<1e-10 and abs(a['rmse']-b['rmse'])<1e-10
 assert selection['selected_families']==[r['family'] for r in ranked[:3]]
 result=dict(status='passed',audit_code_sha256=sha(__file__),teacher_manifest_sha256=sha(teachers/'manifest.json'),n_fold_teacher_models=30,n_teacher_person_metrics=len(teacher_rows),
  n_source_people=30,unused_people=selection['unused_people'],all_fold_people_disjoint=True,scalers_recomputed_from24train_people=True,fresh_initialization_verified=True,checks=checks,
  controls={name:macro([r for r in controls if r['family']==name]) for name in ('constant','ridge_alpha100')},teacher_ranking=ranked,new_person_test10or20_executed=False)
 if args.students:
  students=Path(args.students);sm=json.loads((students/'manifest.json').read_text(encoding='utf-8'));assert sm['status']=='complete' and sm['n_fold_student_fits']==30 and sm['n_final_models']==9
  verify_tree(students,sm)
  assert sm['configuration']['features_sha256']==tm['configuration']['features_sha256']
  for name,digest in sm['configuration']['code_sha256'].items():assert sha(BASE/name)==digest
  student_rows=[];initials={};cv_timing_path=students/'source_cv_timing.json'
  cv_prior=json.loads(cv_timing_path.read_text(encoding='utf-8')) if cv_timing_path.exists() else {};cv_timings={}
  for fold in folds:
   k=fold['fold'];vi=fold['validation_indices'];ti=fold['train_indices']
   for family in selection['selected_families']:
    teacher_path=teachers/f'fold{k}'/family/'best.pt';teacher_pack=torch.load(teacher_path,weights_only=True);teacher_scaler=teacher_pack['scaler']
    target_folder=students/f'fold{k}'/family/'targets';target_proof=json.loads((target_folder/'complete.json').read_text(encoding='utf-8'))
    assert target_proof['teacher_sha256']==sha(teacher_path) and target_proof['n_training_targets']==len(ti)
    assert target_proof['output_sha256']['targets.npz']==sha(target_folder/'targets.npz')
    tmodel=make_model(family,'teacher');tmodel.load_state_dict(teacher_pack['model_state']);tmodel.eval()
    with np.load(teachers/f'fold{k}'/family/'reload_examples.npz',allow_pickle=False) as z:timing_x=torch.from_numpy(z['X'][:1].copy())
    timing_key=f'{k}/{family}/teacher';cv_timings[timing_key]=benchmark(tmodel,timing_x,teacher_path,cv_prior.get(timing_key))
    tx=(np.asarray(d['X'][ti[:64]],np.float32)-teacher_scaler['mean'].numpy())/teacher_scaler['std'].numpy()
    with torch.inference_mode():expected=tmodel(torch.from_numpy(tx)).numpy()
    with np.load(target_folder/'targets.npz',allow_pickle=False) as z:np.testing.assert_allclose(expected,z['prediction'][:64],atol=5e-5,rtol=0)
    for method in ('direct','kd'):
     folder=students/f'fold{k}'/family/method;pack,model,error=check_reloaded(folder,family,'student');meta=pack['metadata']
     timing_key=f'{k}/{family}/{method}';cv_timings[timing_key]=benchmark(model,timing_x,folder/'best.pt',cv_prior.get(timing_key))
     assert set(meta['source_people'])==set(fold['train_people']) and set(meta['validation_people'])==set(fold['validation_people'])
     for key in teacher_scaler:assert torch.equal(pack['scaler'][key],teacher_scaler[key])
     if method=='kd':assert meta['teacher_sha256']==sha(teacher_path) and meta['teacher_target_sha256']==sha(target_folder/'targets.npz') and meta['distillation_weight']==.5
     else:assert meta['distillation_weight']==0. and meta['teacher_sha256'] is None
     initials[k,family,method]=meta['initial_weights_sha256'];rows,epoch=recompute_predictions(folder,d['row_ids'][vi],d,vi);assert epoch==meta['epoch']
     student_rows.extend(dict(family=family,method=method,fold=k,**r) for r in rows)
    assert initials[k,family,'direct']==initials[k,family,'kd']
  for saved in read_csv(students/'summary.csv'):
   actual=macro([r for r in student_rows if r['family']==saved['family'] and r['method']==saved['method']])
   for key in actual:assert abs(actual[key]-float(saved[key]))<1e-10
  save(cv_timing_path,cv_timings)
  # Controlled timing after every worker has completed; every raw repeat retained.
  timing_path=students/'trained_timing.json';prior=json.loads(timing_path.read_text(encoding='utf-8')) if timing_path.exists() else {};timings={};final_checks=[]
  for family in selection['selected_families']:
   for method in ('teacher','direct','kd'):
    folder=students/'final30'/family/method;role='teacher' if method=='teacher' else 'student';pack,model,error=check_reloaded(folder,family,role)
    assert set(pack['metadata']['source_people'])==set(selection['source_people']) and pack['metadata']['validation_people']==[]
    with np.load(folder/'reload_examples.npz',allow_pickle=False) as z:x=torch.from_numpy(z['X'][:1].copy())
    key=family+'/'+method;digest=sha(folder/'best.pt')
    timings[key]=benchmark(model,x,folder/'best.pt',prior.get(key))
    final_checks.append(dict(family=family,method=method,max_reload_difference_deg=error,source_people=30,independent_test=False))
   a=torch.load(students/'final30'/family/'direct/best.pt',weights_only=True);b=torch.load(students/'final30'/family/'kd/best.pt',weights_only=True)
   assert a['metadata']['initial_weights_sha256']==b['metadata']['initial_weights_sha256']
  save(timing_path,timings)
  result.update(student_manifest_sha256=sha(students/'manifest.json'),n_fold_student_models=30,n_student_person_metrics=len(student_rows),paired_student_initializations_verified=True,
   teacher_training_targets_recomputed=True,n_final30_models=9,final_checks=final_checks,trained_timing_sha256=sha(timing_path),source_cv_timing_sha256=sha(cv_timing_path),n_benchmarked_checkpoints=54,student_development_summary=read_csv(students/'summary.csv'))
 result['verified_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(args.out,result);print(json.dumps({k:v for k,v in result.items() if k not in ('checks','final_checks')},indent=2))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teachers',type=Path,default=BASE/'runs/teacher_personcv_v1');p.add_argument('--students',type=Path)
 p.add_argument('--features',type=Path,default=BASE.parent/'semg_experiment/data/features.npz');p.add_argument('--cache',type=Path,default=BASE.parent.parent/'work/sequence_cache')
 p.add_argument('--out',type=Path,default=BASE/'runs/personcv_verification.json');audit(p.parse_args())
