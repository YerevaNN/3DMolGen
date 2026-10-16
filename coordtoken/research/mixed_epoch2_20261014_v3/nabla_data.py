"""Adapter over immutable original train_full arrays; no mixed-corpus input."""
import json
from pathlib import Path
import numpy as np
import torch
class OriginalNabla:
 def __init__(self,root,split):
  if split not in {'train','val'}:raise ValueError('Training-derived inputs only')
  self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
  m=self.manifest
  if m['status']!='complete' or not m['all_original_records_retained'] or m['provenance']['source_split']!='dataset_train_full':raise ValueError('Original train_full required')
  self.rows=m['rows'];self.vocab=m['vocab'];self.dsmap={'nablaDFT':0};self.datasets=['nablaDFT']
  self.tokens=np.load(self.root/'tokens.npy',mmap_mode='r');self.coords=np.load(self.root/'coords.npy',mmap_mode='r');self.atoms=np.load(self.root/'atoms.npy',mmap_mode='r');self.keys=np.load(self.root/'keys.npy',mmap_mode='r')
 def metadata(self,ids,field):
  ids=np.asarray(ids,dtype=np.int64)
  if field=='dataset':return np.zeros(len(ids),dtype=np.int16)
  if field=='key':return self.keys[ids]
  if field=='atoms':return self.atoms[ids].sum(1)
  if field=='length':return (self.tokens[ids]>=0).sum(1)
  raise ValueError(field)
 def batch(self,ids):
  ids=np.asarray(ids,dtype=np.int64)
  if not len(ids) or np.any(ids<0) or np.any(ids>=self.rows):raise ValueError('Invalid rows')
  tokens=self.tokens[ids];length=(int((tokens>=0).sum(1).max())+7)//8*8
  return {'tokens':torch.from_numpy(np.array(tokens[:,:length],dtype=np.int64)), 'coords':torch.from_numpy(np.array(self.coords[ids,:length],dtype=np.float32)), 'atoms':torch.from_numpy(np.array(self.atoms[ids,:length],dtype=bool))}
