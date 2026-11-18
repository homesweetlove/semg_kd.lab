"""Guarded entry point for person-CV/student resumes and state recovery.

The live fitting implementation stays frozen. This outer tool rejects mixed
checkpoint identities/runtimes before invoking it and reconciles publication
after verifying completed artifacts. Use this command for subsequent resumes.
"""
import argparse,datetime,json,os,platform,subprocess,sys
from pathlib import Path
import torch
from selection import BASE,save

def validate_destination(base,out):
 base=Path(base).resolve();out=Path(out).resolve()
 if not out.is_relative_to(base):raise ValueError('Output must stay inside the study directory')
 return out

def runtime_identity():return dict(python=sys.version,torch=str(torch.__version__),cpu_count=os.cpu_count(),platform=platform.platform())

def runtime_guard(path,current):
 path=Path(path)
 if path.exists():
  old=json.loads(path.read_text(encoding='utf-8'))
  if old!=current:raise ValueError('Runtime identity changed;use a new versioned run rather than mixing results')
 else:save(path,current)

def check_checkpoints(out):
 count=0
 for last in Path(out).rglob('last.pt'):
  folder=last.parent;a=torch.load(last,map_location='cpu',weights_only=True);ma=a['metadata'];best=folder/'best.pt'
  if not best.exists():raise ValueError('Checkpoint identity recovery requires selected best.pt:'+str(folder))
  b=torch.load(best,map_location='cpu',weights_only=True);mb=b['metadata']
  for key in ('fit_signature','family','role','source_people','validation_people'):
   if ma.get(key)!=mb.get(key):raise ValueError('Checkpoint identity mismatch:'+str(folder)+'/'+key)
  if mb['epoch']!=ma.get('best_epoch',ma['epoch']):raise ValueError('Checkpoint identity selected-epoch mismatch:'+str(folder))
  for key in a['scaler']:
   if not torch.equal(a['scaler'][key],b['scaler'][key]):raise ValueError('Checkpoint identity scaler mismatch')
  count+=1
 return count

def publish(base,out,kind):
 base=Path(base);out=validate_destination(base,out);manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
 if manifest['status']!='complete':raise ValueError('Cannot publish incomplete results')
 state_path=base/'research_state.json';state=json.loads(state_path.read_text(encoding='utf-8'));relative=out.relative_to(base).as_posix()
 if kind=='teachers':
  selected=json.loads((out/'selection.json').read_text(encoding='utf-8'))['selected_families'];state['executed'].update(teacher_ranking=True,source_person_cross_validation=True)
  state['teacher_selection_run']=dict(directory=relative,status='complete',selected_families=selected,scope='source-development24/6personCV;not independent10/20test')
  task_id='source_only_six_model_selection';evidence=relative+'/selection.json'
  progress=dict(status='complete',completed_models=30,expected_models=30)
  if not state['executed'].get('student_training'):state['next_action']='Run paired direct/KD students with corresponding fold24teachers;then fixed-epoch final30 fits. Fresh10/20test still requires data.'
 else:
  state['executed'].update(student_training=True,student_source_person_cv=True,final30_teacher_student_fitting=True)
  state['student_study_run']=dict(directory=relative,status='complete',scope='source-development only;independent10/20test not executed')
  task_id='distill_selected_three';evidence=relative+'/manifest.json';progress=dict(status='complete',completed_student_fits=30,final_models=9)
  state['next_action']='Verify and report paired source-development results and trained timing;obtain compatible fresh10/20people for independent testing.'
 for task in state['tasks']:
  if task['id']==task_id:task.update(status='complete',evidence=evidence)
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(state_path,state);save(out/'progress.json',progress)

def main(args):
 out=validate_destination(BASE,args.out);guard=BASE/'runs/runtime_guards'/(out.name+'.json');current=runtime_identity()
 environment=out/'environment.json'
 if environment.exists():
  old=json.loads(environment.read_text(encoding='utf-8'))
  for key in ('python','torch','cpu_count','platform'):
   if key in old and old[key]!=current[key]:raise ValueError('Runtime differs from original environment record:'+key)
 runtime_guard(guard,current);checked=check_checkpoints(out) if out.exists() else 0
 command=[sys.executable,'-B',str(BASE/('select_teachers.py' if args.kind=='teachers' else 'student_study.py')),'--out',str(out),'--cache',str(Path(args.cache).resolve()),'--features',str(Path(args.features).resolve()),'--threads',str(args.threads),'--jobs',str(args.jobs)]
 if args.kind=='students':command+=['--teachers',str(Path(args.teachers).resolve())]
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();result=subprocess.run(command,cwd=BASE)
 session=dict(started_utc=start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime=current,checked_checkpoint_pairs=checked,exit_code=result.returncode,command=command)
 name=start.replace(':','').replace('.','').replace('+','_')+'.json';save(BASE/'runs/runtime_sessions'/name,session)
 if result.returncode:raise SystemExit(result.returncode)
 from select_teachers import verify_tree
 verify_tree(out,json.loads((out/'manifest.json').read_text(encoding='utf-8')));check_checkpoints(out);publish(BASE,out,args.kind)
 print('GUARDED_COMPLETE_AND_STATE_RECONCILED',args.kind,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('teachers','students'),default='teachers');p.add_argument('--out',type=Path,default=BASE/'runs/teacher_personcv_v1')
 p.add_argument('--teachers',type=Path,default=BASE/'runs/teacher_personcv_v1');p.add_argument('--cache',type=Path,default=BASE.parent.parent/'work/sequence_cache');p.add_argument('--features',type=Path,default=BASE.parent/'semg_experiment/data/features.npz')
 p.add_argument('--threads',type=int,default=4);p.add_argument('--jobs',type=int,default=3);main(p.parse_args())
