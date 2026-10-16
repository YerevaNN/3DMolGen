"""Compact durable progress report; never reads test outputs."""
import argparse,json,datetime,time
from pathlib import Path
from integrity import OUT

def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None

def update():
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();lines=['# Full training and nabla search', '', 'Updated '+now, '', '**Mixed full:** 12-bit atom codes + one register; width1024, 8+8 layers, global batch512, LR5e-5, dropout0.05. One epoch from scratch on8GPUs.', '']
 mixed=read(OUT/'cases/mixed.json');run=OUT/'runs'/mixed['name'];p=read(run/'progress.json')
 if p:
  step=p.get('step',p.get('steps',0));total=p['total_steps'];seconds=p['training_seconds'];eta=seconds/max(1,step)*(total-step)/3600
  lines+=[f"Status: {p['status']}; {step:,}/{total:,} updates. Rough remaining training: {eta:.1f}h (excludes future queue/checkpoint/validation time).",'']
 else:lines+=['Status: awaiting checks/allocation.','']
 n=read(OUT/'nabla_indices/manifest.json')
 if n:lines+=[f"**Nabla:** {n['official_rows']:,} original train_full rows; {n['training_rows']:,} tuning-train, {n['holdout_rows']:,} held-out conformers; {n['validation_rows']:,} validation keys; zero full-key train/validation overlap.",'']
 else:lines+=['**Nabla:** original train_full hash/split checks pending.','']
 lines+=['One epoch per trial; widths512/1024, 8+8 layers,12/13bits, one register. Each width/bit has its own3×3 LR/batch grid. Dropout0.05 during grids;0/0.1 checked at each winner. Seed123.','', 'Scheduler: mixed run warms up over1,024,000examples, constant through90%, then cosine decay. Nabla uses10%warmup/80%constant/10%cosine within each one-epoch trial.','']
 cases=[read(Path(x)) for x in (read(OUT/'grid_cases.json') or [])];states=[]
 for width in [512,1024]:
  for bits in [12,13]:
   group=[c for c in cases if c['width']==width and c['bits']==bits]
   lines += [f'## Nabla width{width}, {bits}-bit', '', '| LR | Batch256 | Batch512 | Batch1024 |','|---|---:|---:|---:|']
   for lr in sorted({c['lr'] for c in group}):
    cells=[]
    for batch in [256,512,1024]:
     c=next(c for c in group if c['lr']==lr and c['global_batch']==batch);d=OUT/'runs'/c['name'];r=read(d/'result.json');pr=read(d/'progress.json');fail=read(d/'failure.json');unstable=read(d/'unstable.json')
     if r and r['status']=='complete':value=f"{r['validation']['mean_rmsd']:.5f}";status='complete'
     elif unstable:value='unstable';status='unstable'
     elif fail:value='failed';status='failed'
     elif pr:value=f"{100*pr.get('step',pr.get('steps',0))/pr['total_steps']:.0f}%";status='training'
     else:value='queued';status='queued'
     cells.append(value);states.append({'name':c['name'],'status':status})
    lines.append(f'| {lr:g} | '+' | '.join(cells)+' |')
   lines.append('')
 selection=read(OUT/'nabla_decision.json')
 if selection:
  lines+=['## Selected configurations','','| Width | Bits | LR | Batch | Dropout | RMSD, Å |','|---|---:|---:|---:|---:|---:|']
  for w in selection['selected_by_scale']:
   c=w['case'];r=w['result'];lines.append(f"| {c['width']} | {c['bits']} | {c['lr']:g} | {c['global_batch']} | {c['dropout']:g} | {r['validation']['mean_rmsd']:.5f} |")
 lines+=['','RMSD is unaligned mean per molecule; lower is better. Single-seed development evidence; no tests used. Nabla uses official training membership, not the mixed-corpus filtered subset. No44-epoch refit or extra mixed-corpus control is auto-launched.','', '[Protocol](../../research/final_and_nabla_20261012/protocol.json) · [Jobs](jobs.json)']
 tmp=OUT/'REPORT.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(OUT/'REPORT.md')
 return bool(selection and p and p['status']=='complete')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  complete=update()
  if complete or not a.watch:break
  time.sleep(120)
