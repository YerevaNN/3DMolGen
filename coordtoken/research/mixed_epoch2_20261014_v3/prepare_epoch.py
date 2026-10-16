"""Training-only epoch2 permutation and original checkpoint verification."""
import json,math
from pathlib import Path
import numpy as np
import torch
from integrity import ROOT,OUT,sha,verify_sources
from data import bucket_order
from prepare import atomic

def main():
 verify_sources();torch.set_num_threads(4)
 case=json.loads((OUT/'case.json').read_text());dest=OUT/'indices';old=ROOT/'logs/full_release_20261008/indices';meta=json.loads((old/'manifest.json').read_text())
 assert sha(Path(case['data'])/'manifest.json')==meta['data_manifest_sha256']=='8dd1d3a7e315a0439f9cba7a589a43285d9cf9efda77b700e1729895711e1cce'
 for item in json.loads((Path(case['data'])/'manifest.json').read_text())['sources'].values():
  st=Path(item['path']).stat();assert(st.st_size,st.st_mtime_ns)==(item['bytes'],item['mtime_ns'])
 assert sha(old/'train_lengths.npy')==meta['train_lengths.npy_sha256']
 assert sha(old/'validation.npy')==meta['validation.npy_sha256']
 parent=Path(case['parent_run']);pr=json.loads((parent/'result.json').read_text());assert pr['status']=='complete' and pr['examples_seen']==189792028
 ck=torch.load(parent/'last.pt',map_location='cpu',weights_only=False,mmap=True);model=torch.load(parent/'model.pt',map_location='cpu',weights_only=False,mmap=True)
 assert ck['step']==370688 and len(ck['rng'])==8 and ck['optimizer']['state']
 assert sha(parent/'model.pt')=='c3ee867ab22d5893a8a3fbba894657279301ce2add16e1e862966b2046118a41'
 assert all(torch.equal(t,model['model'][k]) and torch.isfinite(t).all() for k,t in ck['model'].items())
 assert all(torch.isfinite(v).all() for state in ck['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
 losses=[json.loads(l)['loss'] for l in (parent/'metrics_rank0.jsonl').read_text().splitlines() if '"loss"' in l][-10:]
 assert len(losses)==10 and all(math.isfinite(v) for v in losses)
 checkpoint_sha=sha(parent/'last.pt');del ck,model
 print('Verified epoch1 weights, optimizer and eight RNG states',flush=True)
 lengths=np.load(old/'train_lengths.npy',mmap_mode='r');assert len(lengths)==189792028
 order=bucket_order(np.arange(len(lengths),dtype=np.uint32),lengths,124,1024)
 seen=np.zeros(len(order),dtype=bool)
 for start in range(0,len(order),1000000):
  chunk=order[start:start+1000000];assert chunk.min()>=0 and chunk.max()<len(order);assert len(np.unique(chunk))==len(chunk);assert not seen[chunk].any();seen[chunk]=True
 assert seen.all();del seen
 assert not np.array_equal(order[:10000],np.load(old/'full_order.npy',mmap_mode='r')[:10000])
 np.save(dest/'order.npy',order)
 if not (dest/'validation.npy').exists():(dest/'validation.npy').symlink_to((old/'validation.npy').resolve())
 atomic(dest/'manifest.json',{'status':'complete','rows':len(order),'shuffle_seed':124,'all_rows_once':True,'differs_from_epoch1':True,'data_manifest_sha256':meta['data_manifest_sha256'],'order.npy_sha256':sha(dest/'order.npy'),'validation.npy_sha256':sha(dest/'validation.npy'),'parent_checkpoint_sha256':checkpoint_sha,'parent_model_sha256':sha(parent/'model.pt'),'parent_last_losses':losses,'parent_optimizer_rng_verified':True,'test_used':False})
 print('Epoch2 permutation verified:',len(order),'rows',flush=True)
if __name__=='__main__':main()
