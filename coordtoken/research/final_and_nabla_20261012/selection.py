"""Choose each scale/codebook independently; then compare tuned finalists."""
import argparse,json,math
from pathlib import Path
from integrity import OUT,sha,verify_sources
from prepare import atomic

def read(path):
 c=json.loads(Path(path).read_text());d=OUT/'runs'/c['name']
 if (d/'unstable.json').exists():return c,None
 if not (d/'result.json').exists():raise RuntimeError('Missing run result: '+c['name'])
 r=json.loads((d/'result.json').read_text());m=json.loads((d/'manifest.json').read_text());i=json.loads((Path(c['indices'])/'manifest.json').read_text())
 assert r['status']=='complete' and not r['test_evaluated']
 assert r['examples_seen']==i['training_rows'] and r['steps']==math.ceil(i['training_rows']/c['global_batch'])
 assert r['manifest_sha256']==sha(d/'manifest.json') and m['index_manifest_sha256']==sha(Path(c['indices'])/'manifest.json')
 assert m['arm']['lr']==c['lr'] and m['arm']['global_batch']==c['global_batch'] and m['config']['bits']==c['bits'] and m['config']['dropout']==c['dropout']
 assert r['validation']['failures']==0
 return c,r

def passes(result,reference):
 return all(result['validation'][k]<=reference['validation'][k]*1.05 for k in ['bond_mae','pair_rmse'])

def grid():
 runs=[read(p) for p in json.loads((OUT/'grid_cases.json').read_text())];chosen=[];fallbacks=[];dropouts=[]
 for width in [512,1024]:
  for bits in [12,13]:
   group=[(c,r) for c,r in runs if c['width']==width and c['bits']==bits and r is not None]
   if not group:raise RuntimeError('No stable cases')
   reference=next(((c,r) for c,r in group if c['lr']==1e-4 and c['global_batch']==512),None)
   if reference is None:
    reference=min(group,key=lambda x:(abs(math.log(x[0]['lr']/1e-4)),abs(x[0]['global_batch']-512)));fallbacks.append(reference[0]['name'])
   eligible=[(c,r) for c,r in group if passes(r,reference[1])];best=min(r['validation']['mean_rmsd'] for c,r in eligible)
   tied=[x for x in eligible if x[1]['validation']['mean_rmsd']<=best*1.01]
   c,r=min(tied,key=lambda x:(x[0]['lr'],abs(x[0]['global_batch']-512)))
   chosen.append({'case':c,'result':r,'reference':reference[0]['name']})
   for dropout in [0.,.1]:
    extra=dict(c,kind='nabla_dropout',dropout=dropout,name=c['name'].replace('drop005',f'drop{dropout:g}'))
    path=OUT/'cases'/f'dropout_{len(dropouts):02d}.json';atomic(path,extra);dropouts.append(str(path))
 atomic(OUT/'grid_selection.json',{'status':'selected','chosen':chosen,'fallback_references':fallbacks,'test_used':False})
 atomic(OUT/'dropout_cases.json',dropouts)

def final():
 selection=json.loads((OUT/'grid_selection.json').read_text());extras=[read(p) for p in json.loads((OUT/'dropout_cases.json').read_text())];winners=[]
 for entry in selection['chosen']:
  c,reference=entry['case'],entry['result'];group=[(c,reference)]+[(x,r) for x,r in extras if r is not None and x['width']==c['width'] and x['bits']==c['bits']]
  candidates=[x for x in group if passes(x[1],reference) and x[1]['validation']['mean_rmsd']<=reference['validation']['mean_rmsd']*.99]
  best=min(candidates,key=lambda x:x[1]['validation']['mean_rmsd']) if candidates else (c,reference)
  winners.append({'case':best[0],'result':best[1]})
 selected=[]
 for width in [512,1024]:
  a=next(w for w in winners if w['case']['width']==width and w['case']['bits']==12);b=next(w for w in winners if w['case']['width']==width and w['case']['bits']==13)
  use13=passes(b['result'],a['result']) and b['result']['validation']['mean_rmsd']<=.98*a['result']['validation']['mean_rmsd']
  selected.append(b if use13 else a)
 atomic(OUT/'nabla_decision.json',{'status':'selected','independently_tuned_finalists':winners,'selected_by_scale':selected,'test_used':False,'budget':'One epoch; no claim of multi-epoch optimality','full_nabla_refit_launched':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['grid','final']);a=p.parse_args();verify_sources();globals()[a.stage]()
