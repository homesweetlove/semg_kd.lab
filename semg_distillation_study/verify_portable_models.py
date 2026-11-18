"""Verify hashes and 64 real input/prediction examples for all portable weights."""
import json
from pathlib import Path
import numpy as np
import torch
from load_distilled import ElbowAngleModel
from data import sha

def validate_portable_manifest(manifest):
 expected={(f,m) for f in ('gru','mlp','transformer') for m in ('teacher','direct','selected')};records=manifest.get('models',[])
 if manifest.get('status')!='complete' or manifest.get('n_models')!=9 or len(records)!=9 or {(r['family'],r['method']) for r in records}!=expected:raise ValueError('Need exact nine unique portable models')
 files=manifest.get('files',{})
 if 'examples.npz' not in files:raise ValueError('Examples missing from protected inventory')
 for r in records:
  if r['path']!=f"{r['family']}/{r['method']}/best.pt" or r['example_key']!=f"{r['family']}__{r['method']}" or r['path'] not in files:raise ValueError('Portable identity/path/inventory mismatch')
 return records

def validate_example_arrays(examples,records):
 X=np.asarray(examples['X'])
 if X.shape!=(64,21,12) or not np.isfinite(X).all():raise ValueError('Need64 finite21x12 input examples')
 for r in records:
  p=np.asarray(examples[r['example_key']])
  if p.shape!=(64,) or not np.isfinite(p).all():raise ValueError('Need64 finite scalar reference predictions')

def validate_prediction_pair(prediction,reference):
 prediction=np.asarray(prediction);reference=np.asarray(reference)
 if prediction.shape!=(64,) or reference.shape!=(64,) or not np.isfinite(prediction).all() or not np.isfinite(reference).all():raise ValueError('Nonfinite or malformed portable predictions')
 difference=float(np.max(np.abs(prediction.astype(np.float64)-reference.astype(np.float64))))
 if not np.isfinite(difference) or difference>=5e-4:raise ValueError('Portable predictions changed')
 return difference

def run():
 torch.set_num_threads(4)
 base=Path(__file__).resolve().parent/'portable_models';manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
 records=validate_portable_manifest(manifest)
 for name,digest in manifest['files'].items():
  if sha(base/name)!=digest:raise ValueError('Portable artifact fingerprint mismatch: '+name)
 checks=[]
 with np.load(base/'examples.npz',allow_pickle=False) as z:
  validate_example_arrays(z,records)
  for record in records:
   p=ElbowAngleModel(base/record['path']).predict_degrees(z['X']);difference=validate_prediction_pair(p,z[record['example_key']])
   checks.append(dict(family=record['family'],method=record['method'],n_examples=len(p),max_difference_deg=difference))
 print(json.dumps(dict(status='passed',n_models=len(checks),checks=checks,examples_are_seen_source_inputs=True,independent_test=False),indent=2,allow_nan=False))
 return checks

if __name__=='__main__':run()
