"""Actual selected 202M model: exact 8-rank stop/resume check on saved fixture."""
import json,subprocess,sys
from pathlib import Path
import torch
from integrity import ROOT,HERE,OUT,verify_sources
from prepare import atomic

def main():
 verify_sources();torch.set_num_threads(1)
 fixture=ROOT/'logs/atom_lr_batch_20261011/resume_fixture';case=json.loads((OUT/'cases/mixed.json').read_text());case.update(name='ddp_resume_fixture',data=str(fixture),indices=str(fixture/'indices'))
 dest=OUT/'ddp_check';dest.mkdir(exist_ok=True);atomic(dest/'case.json',case)
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',str(HERE/'train_run.py'),'--case',str(dest/'case.json')]
 a,b=dest/'uninterrupted',dest/'resumed'
 subprocess.run(command+['--output',str(a)],check=True)
 subprocess.run(command+['--output',str(b),'--stop-after','1'],check=True)
 subprocess.run(command+['--output',str(b),'--resume'],check=True)
 x=torch.load(a/'last.pt',map_location='cpu',weights_only=False);y=torch.load(b/'last.pt',map_location='cpu',weights_only=False)
 assert x['step']==y['step']==2
 diff=max(float((x['model'][k]-y['model'][k]).abs().max()) for k in x['model']);assert diff==0,diff
 for k,d in x['optimizer']['state'].items():
  for name,value in d.items():
   other=y['optimizer']['state'][k][name]
   if torch.is_tensor(value):assert torch.equal(value,other),(k,name)
   else:assert value==other
 assert len(x['rng'])==len(y['rng'])==8
 for r in range(8):
  for name in ['cpu','cuda']:assert torch.equal(x['rng'][r][name],y['rng'][r][name]),(r,name)
 atomic(OUT/'ddp_resume_check.json',{'status':'passed','world':8,'steps':2,'max_parameter_difference':diff,'all_optimizer_states_equal':True,'all_rank_rng_equal':True,'fixture':str(fixture),'test_used':False})
if __name__=='__main__':main()
