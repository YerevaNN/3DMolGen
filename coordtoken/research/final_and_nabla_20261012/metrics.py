import numpy as np
import torch

def bond_indices(tokens,vocab):
    edges=[];last=None;stack=[];rings={}
    for i,t in enumerate(tokens):
        if t<0:break
        s=vocab[int(t)]
        if s.startswith('['):
            if last is not None:edges.append((last,i))
            last=i
        elif s=='(':stack.append(last)
        elif s==')':last=stack.pop()
        elif s=='.':last=None
        elif s.isdigit() or s.startswith('%'):
            if s in rings:edges.append((rings.pop(s),last))
            else:rings[s]=last
    if rings or stack:raise ValueError('Unbalanced graph syntax')
    return edges

@torch.no_grad()
def evaluate(model,data,indices,batch_size=64,device='cuda'):
    model.eval();rmsds=[];bonds=[];pairs=[];code_counts=torch.zeros(2**model.config.bits,device=device,dtype=torch.int64);bits_per_row=[]
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size];batch=data.batch(ids);cpu_tokens=batch['tokens'].numpy()
        edges=[bond_indices(t,data.vocab) for t in cpu_tokens]
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):pred,codes=model(**batch)
        target=batch['coords'];atoms=batch['atoms'];n=atoms.sum(-1).clamp_min(1)
        rmsds.extend((((pred-target).square().sum(-1)*atoms).sum(-1)/n).sqrt().cpu().tolist())
        code_mask=model.code_mask(batch['tokens'],atoms)
        code_counts+=torch.bincount(codes[code_mask].long(),minlength=2**model.config.bits)
        bits_per_row.extend((code_mask.sum(-1)*model.config.bits).cpu().tolist())
        max_edges=max(1,max(map(len,edges)));ii=np.zeros((len(edges),max_edges),np.int64);jj=ii.copy();valid=np.zeros_like(ii,dtype=bool)
        for b,e in enumerate(edges):
            if e:ii[b,:len(e)],jj[b,:len(e)]=np.array(e).T;valid[b,:len(e)]=True
        ii=torch.as_tensor(ii,device=device);jj=torch.as_tensor(jj,device=device);valid=torch.as_tensor(valid,device=device)
        bi=torch.arange(len(edges),device=device)[:,None]
        be=((pred[bi,ii]-pred[bi,jj]).norm(dim=-1)-(target[bi,ii]-target[bi,jj]).norm(dim=-1)).abs()
        bonds.extend(((be*valid).sum(-1)/valid.sum(-1).clamp_min(1)).cpu().tolist())
        mask=atoms[:,:,None]&atoms[:,None,:];mask=mask&torch.ones(mask.shape[-2:],device=device,dtype=torch.bool).triu(1)
        pair=(torch.cdist(pred,pred,compute_mode='donot_use_mm_for_euclid_dist')-torch.cdist(target,target,compute_mode='donot_use_mm_for_euclid_dist')).square()
        pairs.extend(((pair*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)).sqrt().cpu().tolist())
    rmsds=np.asarray(rmsds);bonds=np.asarray(bonds);pairs=np.asarray(pairs);ds=data.metadata(indices,'dataset')
    failures=int((~np.isfinite(rmsds)).sum());per_source={}
    for name,di in data.dsmap.items():
        use=ds==di
        if use.any():per_source[name]=dict(n=int(use.sum()),mean_rmsd=float(rmsds[use].mean()),bond_mae=float(bonds[use].mean()),pair_rmse=float(pairs[use].mean()))
    hist=code_counts.cpu().numpy();p=hist[hist>0]/max(1,hist.sum())
    return dict(rows=len(indices),failures=failures,source_macro_rmsd=float(np.mean([v['mean_rmsd'] for v in per_source.values()])),mean_rmsd=float(rmsds.mean()),bond_mae=float(bonds.mean()),pair_rmse=float(pairs.mean()),mean_geometry_bits=float(np.mean(bits_per_row)),occupied_codes=int((hist>0).sum()),entropy_bits=float(-(p*np.log2(p)).sum()),per_source=per_source),rmsds
