"""Post-selection whole-test evaluation of the separately trained new-data model."""
import csv,hashlib,json,re,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/weka/fgeikyan/CoordToken');OLD=ROOT/'research/final_and_nabla_20261012';sys.path.insert(0,str(OLD))
from model import Config,Tokenizer
from prepare import encode,atomic
from metrics import evaluate
from integrity import sha,verify_sources
OUT=ROOT/'logs/mixed_epoch2_test_20261015'
SOURCE=Path('/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_deduplicated_20261006/merged_test.csv')
CHECKPOINT=ROOT/'logs/mixed_epoch2_20261014/run/model.pt'
class Chunk:
 def __init__(self,items,vocab):
  self.items=items;self.vocab=vocab;self.atom_vocab=np.asarray([v.startswith('[') for v in vocab]);self.dsmap={v:i for i,v in enumerate(sorted({x[2] for x in items}))}
 def metadata(self,ids,field):
  assert field=='dataset';return np.asarray([self.dsmap[self.items[int(i)][2]] for i in ids])
 def batch(self,ids):
  selected=[self.items[int(i)] for i in ids];length=(max(len(x[0]) for x in selected)+7)//8*8;tokens=np.full((len(ids),length),-1,np.int64);coords=np.zeros((len(ids),length,3),np.float32)
  for j,(tok,xyz,group,name) in enumerate(selected):tokens[j,:len(tok)]=tok;coords[j,:len(tok)]=xyz
  return {'tokens':torch.from_numpy(tokens),'coords':torch.from_numpy(coords),'atoms':torch.from_numpy((tokens>=0)&self.atom_vocab[tokens.clip(0)])}
def main():
 verify_sources();OUT.mkdir(exist_ok=True);
 assert not (OUT/'result.json').exists(),'Completed evaluation already exists'
 assert sha(Path(__file__))==json.loads(Path(__file__).with_name('new_test_freeze.json').read_text())['files']['evaluate_new_test.py']
 torch.set_num_threads(4);torch.cuda.set_device(0);torch.set_float32_matmul_precision('high')
 assert sha(CHECKPOINT)=='29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f'
 manifest=json.loads((SOURCE.parent/'manifest.json').read_text());entry=next(x for x in manifest['files'] if Path(x['path']).name==SOURCE.name);assert SOURCE.stat().st_size==entry['bytes'];assert sha(SOURCE)==entry['sha256']
 ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);cfg={k:v for k,v in ckpt['config'].items() if k!='parameters'};model=Tokenizer(Config(**cfg));model.load_state_dict(ckpt['model'],strict=True);model.cuda().eval();vocab=ckpt['vocab']
 plan=json.loads((ROOT/'logs/full_release_20261008/packed/plan.json').read_text());assert vocab==plan['vocab'];mapping={v:i for i,v in enumerate(vocab)};regex=re.compile(plan['pattern'])
 total=0;sums=defaultdict(lambda:defaultdict(float));started=time.monotonic();completed=OUT/'per_conformer.csv';temp=OUT/'per_conformer.partial.csv'
 with SOURCE.open(newline='') as stream,temp.open('w',newline='') as output:
  reader=csv.DictReader(stream);writer=csv.writer(output);writer.writerow(['row','name','dataset_regime','rmsd_angstrom']);chunk=[]
  def consume():
   nonlocal total
   data=Chunk(chunk,vocab);metrics,errors=evaluate(model,data,np.arange(len(chunk)),batch_size=64)
   assert metrics['failures']==0 and np.isfinite(errors).all()
   for group,m in metrics['per_source'].items():
    assert all(np.isfinite(m[k]) for k in ['mean_rmsd','bond_mae','pair_rmse']);sums[group]['n']+=m['n']
    for k in ['mean_rmsd','bond_mae','pair_rmse']:sums[group][k]+=m[k]*m['n']
   for j,(item,error) in enumerate(zip(chunk,errors)):writer.writerow([total+j,item[3],item[2],float(error)])
   total+=len(chunk);atomic(OUT/'progress.json',{'status':'evaluating','rows':total,'total_rows':entry['rows'],'seconds':time.monotonic()-started,'test_evaluated':True,'used_for_selection':False});print('Evaluated',total,flush=True)
  for row in reader:
   group=row['dataset'];subset=row.get('test_subset','')
   if group=='nablaDFT':
    if not subset:raise ValueError('Missing nabla test regime')
    group+=' / '+subset
   tok,xyz,n=encode(row['enriched_text'],mapping,regex);chunk.append((tok,xyz,group,row['name']))
   if len(chunk)==8192:consume();chunk=[]
  if chunk:consume()
 assert total==entry['rows'];temp.replace(completed)
 groups={g:{'rows':int(s['n']),**{k:s[k]/s['n'] for k in ['mean_rmsd','bond_mae','pair_rmse']}} for g,s in sums.items()}
 pooled={k:sum(s[k] for s in sums.values())/total for k in ['mean_rmsd','bond_mae','pair_rmse']}
 result={'status':'complete','test_evaluated':True,'used_for_selection':False,'evaluator_sha256':sha(Path(__file__)),'checkpoint':str(CHECKPOINT.resolve()),'checkpoint_sha256':sha(CHECKPOINT),'source':str(SOURCE),'source_sha256':entry['sha256'],'rows':total,'parse_or_prediction_failures':0,'per_dataset_regime':groups,'pooled':pooled,'seconds':time.monotonic()-started,'metric':'Mean per-conformer atom RMSD, centered input, no alignment or symmetry matching; bf16 inference/FP32 output. Nabla regimes reported separately.','limitations':'Final merged test snapshot; GEOM-Revisited is a separate reference and is not included in this CSV. Existing upstream representation and cross-source conformer-independence limitations remain.'}
 atomic(OUT/'result.json',result)
 lines=['# CoordToken epoch2: full test evaluation','','Fixed epoch2 checkpoint; no further selection. Mean unaligned RMSD in Å.','','| Dataset / regime | Conformers | RMSD |','|---|---:|---:|']
 for g,m in sorted(groups.items()):lines.append(f"| {g} | {m['rows']:,} | {m['mean_rmsd']:.5f} |")
 lines+=['',f"Pooled across{total:,}conformers: **{pooled['mean_rmsd']:.5f}Å**. GEOM-Revisited excluded from this pooled CSV result.",'','All rows evaluated; no parse/prediction failures. No test-based hyperparameter changes.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
