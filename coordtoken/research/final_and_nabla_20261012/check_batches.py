"""Check distributed tail masks, full coverage and atom-weighted gradients."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import Batches
from prepare import atomic
from integrity import OUT
class Fixture:
 def metadata(self,ids,field):
  assert field=='atoms';return np.asarray(ids)%9+1
 def batch(self,ids):
  ids=np.asarray(ids);n=ids%9+1;mask=np.arange(9)[None,:]<n[:,None]
  return {'tokens':torch.as_tensor(ids.copy()),'atoms':torch.as_tensor(mask)}
def main():
 torch.set_num_threads(1)
 for rows in [5003,513,540]:
  order=np.random.default_rng(123).permutation(rows)
  for world,batch,acc in [(8,64,1),(1,128,2),(1,128,4),(1,128,8)]:
   size=world*batch*acc
   for start in [0,1]:
    got=[];gradients={}
    for rank in range(world):
     loader=DataLoader(Batches(Fixture(),order,batch,rank,world,start*acc,acc),batch_size=None,num_workers=2)
     for item in loader:
      group=int(item['step'])//acc;ids=order[group*size:(group+1)*size];den=int((ids%9+1).sum());assert item['global_atoms']==den
      active=item['atoms'].any(1);localids=item['tokens'][active].numpy();got.extend(localids.tolist())
      # L=(w*x)^2 with atom weights; mimic DDP averaging exactly.
      localgrad=float((2*(localids+1)**2*(localids%9+1)).sum())*world/den
      gradients[group]=gradients.get(group,0.)+localgrad/world
    np.testing.assert_array_equal(np.sort(got),np.sort(order[start*size:]))
    for group,value in gradients.items():
     ids=order[group*size:(group+1)*size];expected=float((2*(ids+1)**2*(ids%9+1)).sum())/float((ids%9+1).sum());assert np.isclose(value,expected,rtol=1e-12)
 atomic(OUT/'batch_checks.json',{'status':'passed','eight_rank_tail_and_dummy_mask':True,'global_atom_gradient_normalization':True,'resume_order_two_workers':True,'batches':[256,512,1024],'test_used':False})
if __name__=='__main__':main()
