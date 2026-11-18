"""Six proposed regression families, each with large teacher/small student.

Shared causal-history input[N,21,12]; output[N]. Architectural definitions only;
these initial weights are NOT pretrained and carry no performance claim.
"""
import torch
from torch import nn

FAMILIES=('mlp','cnn1d','lstm','gru','tcn','transformer')
STEPS=21
FEATURES=12

class RegressionModel(nn.Module):
 def check(self,x):
  if x.ndim!=3 or tuple(x.shape[1:])!=(STEPS,FEATURES) or len(x)<1:
   raise ValueError('Expected nonempty[N,21,12] past-only feature sequences')

class MLP(RegressionModel):
 def __init__(self,large):
  super().__init__();a,b=(256,128) if large else (32,16)
  self.net=nn.Sequential(nn.Flatten(),nn.Linear(STEPS*FEATURES,a),nn.ReLU(),nn.Linear(a,b),nn.ReLU(),nn.Linear(b,1))
 def forward(self,x):self.check(x);return self.net(x).squeeze(-1)

class CNN(RegressionModel):
 def __init__(self,large):
  super().__init__();a,b=(64,128) if large else (8,16)
  self.net=nn.Sequential(nn.Conv1d(FEATURES,a,5,padding=2),nn.ReLU(),nn.Conv1d(a,b,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(b,1))
 def forward(self,x):self.check(x);return self.net(x.transpose(1,2)).squeeze(-1)

class Recurrent(RegressionModel):
 def __init__(self,kind,large):
  super().__init__();width=128 if large else 24;depth=2 if large else 1
  self.encoder=(nn.LSTM if kind=='lstm' else nn.GRU)(FEATURES,width,depth,batch_first=True)
  self.head=nn.Linear(width,1)
 def forward(self,x):
  self.check(x);h,_=self.encoder(x);return self.head(h[:,-1]).squeeze(-1)

class CausalBlock(nn.Module):
 def __init__(self,cin,cout,dilation):
  super().__init__();self.padding=2*dilation
  self.conv=nn.Conv1d(cin,cout,3,padding=self.padding,dilation=dilation)
  self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
 def forward(self,x):return torch.relu(self.conv(x)[...,:-self.padding]+self.skip(x))

class TCN(RegressionModel):
 def __init__(self,large):
  super().__init__();width=128 if large else 16;layers=[];cin=FEATURES
  for dilation in (1,2,4,8):layers.append(CausalBlock(cin,width,dilation));cin=width
  self.encoder=nn.Sequential(*layers);self.head=nn.Linear(width,1)
 def forward(self,x):self.check(x);return self.head(self.encoder(x.transpose(1,2))[...,-1]).squeeze(-1)

class Transformer(RegressionModel):
 def __init__(self,large):
  super().__init__();width=128 if large else 32;depth=2 if large else 1
  self.project=nn.Linear(FEATURES,width);self.position=nn.Parameter(torch.zeros(1,STEPS,width))
  layer=nn.TransformerEncoderLayer(width,4,dim_feedforward=2*width,dropout=0.,batch_first=True)
  self.encoder=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False);self.head=nn.Linear(width,1)
 def forward(self,x):
  self.check(x);h=self.encoder(self.project(x)+self.position)
  return self.head(h[:,-1]).squeeze(-1)

def make_model(family,role):
 if family not in FAMILIES or role not in ('teacher','student'):raise ValueError('Unknown family/role')
 large=role=='teacher'
 if family=='mlp':return MLP(large)
 if family=='cnn1d':return CNN(large)
 if family in ('lstm','gru'):return Recurrent(family,large)
 if family=='tcn':return TCN(large)
 return Transformer(large)

def parameter_count(model):return sum(p.numel() for p in model.parameters())
