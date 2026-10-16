"""Read-only packed train/validation access and deterministic length-aware shuffling."""
import hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset
from prepare import META, atomic

class Packed:
    def __init__(self,root,split):
        if split not in {'train','val'}:raise ValueError('Only train/val are allowed')
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest['status']!='complete' or self.manifest['errors']:raise ValueError('Preparation has not passed')
        self.vocab=self.manifest['vocab'];self.atom_vocab=np.array([v.startswith('[') for v in self.vocab])
        self.shards=[s for s in self.manifest['shards'] if s['task']['split']==split]
        self.ends=np.cumsum([s['valid'] for s in self.shards]);self.starts=np.r_[0,self.ends[:-1]];self.rows=int(self.ends[-1]);self.maps=[]
        self.datasets=sorted({d for s in self.shards for d in s['datasets']});self.dsmap={d:i for i,d in enumerate(self.datasets)}
        for s in self.shards:
            p=Path(s['task']['output']);n=s['valid'];t=s['tokens']
            if not n:raise ValueError('Empty shard')
            local={i:self.dsmap[d] for d,i in s['datasets'].items()}
            remap=np.array([local[i] for i in range(len(local))],dtype=np.int16)
            self.maps.append((np.memmap(p/'tokens.bin',mode='r',dtype='<u2',shape=(t,)),np.memmap(p/'coords.bin',mode='r',dtype='<f4',shape=(t,3)),np.memmap(p/'offsets.bin',mode='r',dtype='<u8',shape=(n,)),np.memmap(p/'meta.bin',mode='r',dtype=META,shape=(n,)),remap))
    def metadata(self,indices,field):
        indices=np.asarray(indices,dtype=np.int64);shards=np.searchsorted(self.ends,indices,side='right');parts=[]
        dtype=np.int16 if field=='dataset' else META.fields[field][0]
        out=np.empty(len(indices),dtype=dtype)
        for shard in np.unique(shards):
            mask=shards==shard;local=indices[mask]-self.starts[shard];m=self.maps[shard];values=m[3][field][local]
            out[mask]=m[4][values] if field=='dataset' else values
        return out
    def batch(self,indices):
        indices=np.asarray(indices,dtype=np.int64)
        if (indices<0).any() or (indices>=self.rows).any():raise ValueError('Invalid row index')
        lengths=self.metadata(indices,'length');length=(int(lengths.max())+7)//8*8
        tokens=np.full((len(indices),length),-1,dtype=np.int64);coords=np.zeros((len(indices),length,3),np.float32)
        shards=np.searchsorted(self.ends,indices,side='right')
        for i,(idx,shard,n) in enumerate(zip(indices,shards,lengths)):
            tok,xyz,offsets,meta,_=self.maps[shard];row=int(idx-self.starts[shard]);start=int(offsets[row]);n=int(n)
            tokens[i,:n]=tok[start:start+n];coords[i,:n]=xyz[start:start+n]
        atoms=(tokens>=0)&self.atom_vocab[tokens.clip(0)]
        return {'tokens':torch.from_numpy(tokens),'coords':torch.from_numpy(coords),'atoms':torch.from_numpy(atoms)}

def bucket_order(indices,lengths,seed,global_batch,pool_batches=64):
    # Start with a global row permutation, bucket only within random pools, then
    # shuffle the batches within each pool to avoid a length curriculum.
    indices=np.asarray(indices);order=indices.copy();rng=np.random.default_rng(seed);rng.shuffle(order)
    pool=global_batch*pool_batches
    for start in range(0,len(order),pool):
        chunk=order[start:start+pool]
        chunk[:]=chunk[np.argsort(lengths[chunk],kind='stable')]
        full=len(chunk)//global_batch
        if full:
            block=chunk[:full*global_batch].reshape(full,global_batch)
            block[:]=block[rng.permutation(full)]
    return order

class Batches(IterableDataset):
    def __init__(self,data,order,batch_size,rank=0,world=1,start_step=0,accumulation=1):
        self.data=data;self.order=order;self.batch_size=batch_size;self.rank=rank;self.world=world;self.start_step=start_step;self.accumulation=accumulation
    def __iter__(self):
        info=torch.utils.data.get_worker_info();worker=info.id if info else 0;workers=info.num_workers if info else 1
        size=self.batch_size*self.world
        previous_group=None;group_atoms=None
        for step in range(self.start_step+worker,math.ceil(len(self.order)/size),workers):
            global_ids=self.order[step*size:(step+1)*size]
            ids=global_ids[self.rank*self.batch_size:(self.rank+1)*self.batch_size]
            active=len(ids)
            if not active:ids=global_ids[:1]
            batch=self.data.batch(ids)
            if not active:batch['atoms'].zero_()
            group_start=(step//self.accumulation)*size*self.accumulation
            if group_start!=previous_group:
                group_ids=self.order[group_start:group_start+size*self.accumulation]
                group_atoms=int(self.data.metadata(group_ids,'atoms').astype(np.int64).sum());previous_group=group_start
            batch['global_atoms']=group_atoms
            batch['global_examples']=len(global_ids);batch['step']=step
            yield batch
