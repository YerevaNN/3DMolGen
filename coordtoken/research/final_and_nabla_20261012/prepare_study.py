"""Verify immutable inputs and build training-only indices without test access."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from integrity import HERE,ROOT,OUT,sha,verify_sources
from prepare import atomic
from data import bucket_order

def mixed():
 data=ROOT/'logs/full_release_20261008/packed';old=ROOT/'logs/full_release_20261008/indices';dest=OUT/'mixed_indices';dest.mkdir(exist_ok=True)
 m=json.loads((old/'manifest.json').read_text())
 assert sha(data/'manifest.json')==m['data_manifest_sha256']=='8dd1d3a7e315a0439f9cba7a589a43285d9cf9efda77b700e1729895711e1cce'
 for s in json.loads((data/'manifest.json').read_text())['sources'].values():
  st=Path(s['path']).stat();assert (st.st_size,st.st_mtime_ns)==(s['bytes'],s['mtime_ns'])
 order=np.load(old/'full_order.npy',mmap_mode='r');assert len(order)==189792028
 assert sha(old/'full_order.npy')==m['full_order.npy_sha256']
 seen=np.zeros(len(order),dtype=bool)
 for start in range(0,len(order),1000000):
  x=np.asarray(order[start:start+1000000]);assert int(x.min())>=0 and int(x.max())<len(order);assert len(np.unique(x))==len(x);assert not seen[x].any();seen[x]=True
 assert seen.all();del seen
 prefix=np.load(ROOT/'logs/atom_lr_batch_20261011/indices/order.npy',mmap_mode='r');assert np.array_equal(order[:len(prefix)],prefix)
 for name,source in [('order.npy',old/'full_order.npy'),('validation.npy',old/'validation.npy')]:
  if not (dest/name).exists():(dest/name).symlink_to(source.resolve())
 assert sha(dest/'validation.npy')=='24600ba212630a878d668d16a2fb417b72488132810bd003e80bdf72a6c2e2d5'
 atomic(dest/'manifest.json',{'status':'complete','rows':len(order),'data_manifest_sha256':m['data_manifest_sha256'],'order.npy_sha256':sha(dest/'order.npy'),'validation.npy_sha256':sha(dest/'validation.npy'),'all_rows_once':True,'tuning_prefix_match':True,'source_checks':'Pinned packed manifest plus source size/mtime; verified permutation and published input checksums; no new whole-CSV rehash','test_used':False})

def nabla():
 data=ROOT/'research/coordtoken_v2/data/nabla_full_training_v2';dest=OUT/'nabla_indices';dest.mkdir(exist_ok=True)
 m=json.loads((data/'manifest.json').read_text());v=json.loads((data/'independent_verification.json').read_text())
 assert sha(data/'manifest.json')==v['dataset_manifest_sha256']=='59992a703daf84d18c56ac68e543585eede93a1f7f4a11c83109d61e8e28b2bb'
 assert v['status']=='verified' and m['rows']==8849983 and m['all_original_records_retained'] and m['provenance']['source_split']=='dataset_train_full'
 # Fresh hashes of original DB and all prepared arrays, not a claim from filenames.
 source=Path(m['provenance']['original_db']);assert sha(source)==m['provenance']['original_sha256']
 for name,entry in m['arrays'].items():
  assert sha(data/(name+'.npy'))==entry['sha256'],name
  print('Verified',name,flush=True)
 keys=np.load(data/'keys.npy',mmap_mode='r');tokens=np.load(data/'tokens.npy',mmap_mode='r');splits=np.load(data/'split.npy',mmap_mode='r');assert not np.any(splits)
 unique,inverse=np.unique(keys,return_inverse=True)
 hold=np.array([int.from_bytes(hashlib.sha256(b'coordtoken-nabla-hparam-20261012:'+bytes(k)).digest()[:8],'big')%100<5 for k in unique])
 membership=hold[inverse];train=np.flatnonzero(~membership).astype(np.uint32);val=np.flatnonzero(membership).astype(np.uint32)
 assert len(train)+len(val)==m['rows'] and len(np.intersect1d(keys[train],keys[val]))==0
 lengths=np.empty(m['rows'],dtype=np.uint16)
 for start in range(0,m['rows'],32768):lengths[start:start+32768]=(tokens[start:start+32768]>=0).sum(1)
 assert 0<int(lengths.min())<=int(lengths.max())<=tokens.shape[1]
 np.save(dest/'lengths.npy',lengths)
 order=bucket_order(train,lengths,123,1024);np.save(dest/'order.npy',order)
 # One reproducibly sampled conformer/key; equal weight for molecular identities.
 rng=np.random.default_rng(123);rng.shuffle(val);_,first=np.unique(keys[val],return_index=True);eligible=val[first];rng.shuffle(eligible);validation=eligible[:8192]
 np.save(dest/'validation.npy',validation);np.save(dest/'holdout.npy',np.flatnonzero(membership).astype(np.uint32));np.save(dest/'training_ids.npy',train)
 assert len(np.unique(order))==len(train) and not membership[order].any() and membership[validation].all()
 assert len(np.unique(keys[validation]))==len(validation)
 atomic(dest/'manifest.json',{'status':'complete','official_rows':m['rows'],'training_rows':len(train),'holdout_rows':int(membership.sum()),'training_keys':int((~hold).sum()),'holdout_keys':int(hold.sum()),'validation_rows':len(validation),'key_intersection':0,'all_training_rows_once':True,'max_length':int(lengths.max()),'data_manifest_sha256':sha(data/'manifest.json'),'original_db':str(source),'original_db_sha256':sha(source),'fresh_array_hashes_verified':True,'order.npy_sha256':sha(dest/'order.npy'),'validation.npy_sha256':sha(dest/'validation.npy'),'selection':'5%full keys held out; seeded one conformer per key; development only, not a fresh independent confirmation population','test_used':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['mixed','nabla']);a=p.parse_args();verify_sources();globals()[a.kind]()
 print('Preparation complete:',a.kind,flush=True)
