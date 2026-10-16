import json,subprocess,sys,os,fcntl
from integrity import HERE,OUT,verify_sources,verify_case,sha

def main():
 verify_sources();case=OUT/'case.json';verify_case(json.loads(case.read_text()))
 import torch
 assert int(os.environ['SLURM_JOB_NUM_NODES'])==1 and torch.cuda.device_count()==8
 dest=OUT/'run';dest.mkdir(exist_ok=True)
 with (dest/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (dest/'result.json').exists() and json.loads((dest/'result.json').read_text()).get('status')=='complete':return
  cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',str(HERE/'train_run.py'),'--case',str(case),'--output',str(dest),'--resume']
  for step in [100,200]:
   p=dest/f'ddp_gate_{step}.json'
   if not p.exists():subprocess.run(cmd+['--stop-after',str(step)],check=True)
   r=json.loads(p.read_text());assert r['status']=='passed' and r['world']==8 and r['resumed_at_step']==(0 if step==100 else 100)
  subprocess.run(cmd,check=True)
  subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
if __name__=='__main__':main()
