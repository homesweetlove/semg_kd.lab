"""Portable teacher/student inference from the saved role and source scaler."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import models
from models import make_model

class ElbowAngleModel:
 def __init__(self,path):
  pack=torch.load(Path(path),map_location='cpu',weights_only=True);self.metadata=pack['metadata'];self.scaler=pack['scaler']
  self.family=self.metadata['family'];self.role=self.metadata['role']
  if hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest()!=self.metadata['models_code_sha256']:raise ValueError('Architecture fingerprint mismatch')
  self.model=make_model(self.family,self.role).eval();self.model.load_state_dict(pack['model_state'])

 def predict_degrees(self,feature_sequences,batch_size=1024):
  data=np.asarray(feature_sequences,dtype=np.float32)
  if data.ndim!=3 or data.shape[1:]!=(21,12) or not len(data) or not np.isfinite(data).all() or batch_size<1:raise ValueError('Need finite nonempty[N,21,12] log-feature histories')
  normalized=(data-self.scaler['mean'].numpy())/self.scaler['std'].numpy();parts=[]
  with torch.inference_mode():
   for start in range(0,len(data),batch_size):parts.append(self.model(torch.from_numpy(normalized[start:start+batch_size])).numpy())
  return np.concatenate(parts)*float(self.scaler['ystd'])+float(self.scaler['ymean'])
