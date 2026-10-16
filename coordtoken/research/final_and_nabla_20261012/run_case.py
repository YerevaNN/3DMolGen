"""Run a frozen case; full training includes a checkpointed 8-GPU gate."""
import argparse,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
from integrity import HERE,OUT,verify_sources,verify_case,sha
from prepare import atomic

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['mixed','grid','dropout']);p.add_argument('index',type=int,nargs='?',default=0);a=p.parse_args();verify_sources()
 if a.stage=='mixed':casepath=OUT/'cases/mixed.json'
 else:
  listing=json.loads((OUT/(a.stage+'_cases.json')).read_text());casepath=Path(listing[a.index])
 case=json.loads(casepath.read_text())
 if a.stage in {'mixed','grid'}:
  registered=json.loads((HERE/'registered_cases.json').read_text());assert registered[str(casepath)]==case,'Registered case changed'
 verify_case(case);dest=OUT/'runs'/case['name'];dest.mkdir(parents=True,exist_ok=True)
 with (dest/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (dest/'result.json').exists() and json.loads((dest/'result.json').read_text())['status']=='complete':return
  atomic(dest/'case_receipt.json',{'case':case,'case_sha256':sha(casepath),'job':os.environ.get('SLURM_JOB_ID'),'array':os.environ.get('SLURM_ARRAY_TASK_ID')})
  command=[sys.executable]
  if a.stage=='mixed':command+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8']
  command+=[str(HERE/'train_run.py'),'--case',str(casepath),'--output',str(dest),'--resume']
  try:
   if a.stage=='mixed' and not (dest/'ddp_gate.json').exists():subprocess.run(command+['--stop-after','100'],check=True)
   if a.stage=='mixed':
    gate=json.loads((dest/'ddp_gate.json').read_text());assert gate['status']=='passed' and gate['world']==8
   subprocess.run(command,check=True)
  except Exception as exc:
   atomic(dest/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'stage':a.stage,'job':os.environ.get('SLURM_JOB_ID')})
   if a.stage!='mixed' and (dest/'unstable.json').exists():return
   raise
if __name__=='__main__':main()
