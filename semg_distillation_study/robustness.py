"""Fixed-teacher student stochastic robustness; source people remain n30."""
import itertools
import numpy as np

def assert_baseline_runtime(expected,current):
 if expected!=current:raise ValueError('Baseline runtime differs;do not mix reused and new fits')

def condition_grid(families,seeds,weights):
 if not families or not seeds or not weights or len(set(families))!=len(families) or len(set(seeds))!=len(seeds) or len(set(weights))!=len(weights):raise ValueError('Unique nonempty factors required')
 if 0. not in weights or not all(np.isfinite(w) and 0<=w<=1 for w in weights):raise ValueError('Need direct control and valid KD weights')
 return [dict(fold=fold,family=family,seed=int(seed),weight=float(weight),reuse_baseline=seed==20261113 and weight in (0.,.5)) for fold,family,seed,weight in itertools.product(range(5),families,seeds,weights)]

def summarize(rows,families,seeds,weights,people,bootstrap=5000):
 expected=set(itertools.product(families,seeds,weights,people));lookup={}
 if len(expected)!=len(families)*len(seeds)*len(weights)*len(people):raise ValueError('Duplicate factor levels')
 for r in rows:
  key=(r['family'],int(r['seed']),float(r['weight']),str(r['subject']))
  if key not in expected or key in lookup:raise ValueError('Unexpected/duplicate seed-person metric')
  values=[float(r['mae']),float(r['rmse'])]
  if not np.isfinite(values).all() or min(values)<0 or int(r['n'])<1:raise ValueError('Invalid metric')
  lookup[key]=dict(mae=values[0],rmse=values[1])
 if set(lookup)!=expected:raise ValueError('Incomplete seed-person grid')
 summary=[];per_seed=[];paired=[];rng=np.random.default_rng(20261117)
 for family in families:
  for weight in weights:
   mseeds=[];rseeds=[]
   for seed in seeds:
    ma=float(np.mean([lookup[family,seed,weight,p]['mae'] for p in people]));rm=float(np.mean([lookup[family,seed,weight,p]['rmse'] for p in people]))
    mseeds.append(ma);rseeds.append(rm);per_seed.append(dict(family=family,seed=seed,weight=weight,mae=ma,rmse=rm,n_people=len(people)))
   summary.append(dict(family=family,weight=weight,mae=float(np.mean(mseeds)),rmse=float(np.mean(rseeds)),seed_macro_mae_sd=float(np.std(mseeds,ddof=1)) if len(seeds)>1 else 0.,
    seed_macro_mae_min=min(mseeds),seed_macro_mae_max=max(mseeds),n_people=len(people),n_student_seeds=len(seeds)))
   if weight:
    # Average seed-paired differences WITHIN each person before resampling people.
    delta=np.array([np.mean([lookup[family,s,0.,p]['mae']-lookup[family,s,weight,p]['mae'] for s in seeds]) for p in people])
    samples=delta[rng.integers(0,len(people),size=(bootstrap,len(people)))].mean(axis=1)
    paired.append(dict(family=family,weight=weight,mae_improvement=float(delta.mean()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),
     people_improved=int((delta>0).sum()),n_people=len(people),n_student_seeds=len(seeds),scope='Conditional person bootstrap after averaging student seeds;fixed teachers;source-development selection reuse;not independent confirmation'))
 return dict(summary=summary,per_seed=per_seed,paired=paired)

def select_weights(summary):
 result={}
 for family in sorted({r['family'] for r in summary}):
  best=min([r for r in summary if r['family']==family],key=lambda r:(r['mae'],r['rmse'],r['weight']));result[family]=best['weight']
 return result
