#!/usr/bin/env python3
"""Pack only train/validation from the frozen release; no test path is opened."""
import argparse, ast, collections, concurrent.futures, csv, hashlib, json, os, re, struct, time
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"train.py").exists() and (p/"utils.py").exists())
META=np.dtype([('length','<u2'),('atoms','<u2'),('dataset','<u2'),('key','S27'),('source_offset','<u8')])

def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def vocabulary():
    tree=ast.parse((ROOT/'utils.py').read_text());vocab=pattern=None
    for node in tree.body:
        if isinstance(node,ast.Assign):
            names=[getattr(x,'id','') for x in node.targets]
            if 'VOCAB' in names:vocab=ast.literal_eval(node.value)
            if '_ENRICHED_TOKEN_PATTERN' in names:pattern=ast.literal_eval(node.value.args[0])
    assert vocab and pattern
    return vocab,pattern

def encode(text,vocab,regex):
    ids=[];xyz=[];atoms=[];pos=0
    for m in regex.finditer(text):
        if m.start()!=pos:raise ValueError('unparsed_fragment')
        if m.group(1) is not None:
            token=m.group(1);coords=[float(c) for c in m.group(2).split(',')]
            if len(coords)!=3:raise ValueError('bad_coordinate_count')
            atoms.append(len(ids))
        else:token=next(g for g in m.groups()[2:] if g is not None);coords=[0.,0.,0.]
        if token not in vocab:raise ValueError('unsupported_token:'+token)
        ids.append(vocab[token]);xyz.append(coords);pos=m.end()
    if pos!=len(text) or not atoms:raise ValueError('unparsed_or_no_atoms')
    if len(ids)>65535:raise ValueError('length_over_uint16')
    xyz=np.asarray(xyz,dtype=np.float32)
    if not np.isfinite(xyz).all():raise ValueError('nonfinite_coordinate')
    xyz[atoms]-=xyz[atoms].mean(0)
    return np.asarray(ids,dtype='<u2'),xyz,len(atoms)

def worker(task):
    path=Path(task['path']);out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    receipt=out/'receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['task']!=task:raise RuntimeError('resume task mismatch')
        return r
    voc,pattern=vocabulary();mapping={v:i for i,v in enumerate(voc)};regex=re.compile(pattern)
    datasets={};counts=collections.Counter();errors=collections.Counter();hist=collections.Counter();examples=[]
    rows=valid=positions=0;bytehash=hashlib.sha256();started=time.time()
    streams={k:(out/(k+'.bin')).open('wb',buffering=4*1024*1024) for k in ['tokens','coords','offsets','meta']}
    try:
        with path.open('rb',buffering=4*1024*1024) as f:
            header=next(csv.reader([f.readline().decode('utf-8')],strict=True));first=f.tell()
            if task['start']>first:
                f.seek(task['start']-1)
                if f.read(1)!=b'\n':f.readline()
            else:f.seek(first)
            while f.tell()<task['end']:
                offset=f.tell();line=f.readline()
                if not line:break
                bytehash.update(line);rows+=1
                try:
                    row=next(csv.reader([line.decode('utf-8')],strict=True))
                    if len(row)!=len(header):raise ValueError('csv_column_count')
                    row=dict(zip(header,row));ids,xyz,na=encode(row['enriched_text'],mapping,regex)
                    ds=row['dataset'];di=datasets.setdefault(ds,len(datasets));key=row['inchikey'].encode('ascii')
                    if len(key)>27:raise ValueError('bad_inchikey_length')
                    meta=np.array([(len(ids),na,di,key,offset)],dtype=META)
                    streams['offsets'].write(struct.pack('<Q',positions));streams['meta'].write(meta.tobytes())
                    streams['tokens'].write(ids.tobytes());streams['coords'].write(xyz.astype('<f4',copy=False).tobytes())
                    positions+=len(ids);valid+=1;counts[ds]+=1;hist[len(ids)]+=1
                except Exception as error:
                    label=str(error);errors[label]+=1
                    if len(examples)<20:examples.append({'offset':offset,'error':label})
    finally:
        for f in streams.values():f.close()
    r=dict(status='complete' if not errors else 'invalid',task=task,rows=rows,valid=valid,tokens=positions,
           datasets=datasets,counts=dict(counts),length_histogram=dict(hist),errors=dict(errors),examples=examples,
           input_region_sha256=bytehash.hexdigest(),seconds=time.time()-started)
    atomic(receipt,r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--train-shards',type=int,default=64);a=p.parse_args()
    release=a.release.resolve();out=a.output;out.mkdir(parents=True,exist_ok=True)
    release_manifest=json.loads((release/'manifest.json').read_text());voc,pattern=vocabulary()
    tasks=[];sources={}
    for split,nshards in [('val',8),('train',a.train_shards)]:
        path=release/f'merged_{split}.csv';entry=next(e for e in release_manifest['files'] if Path(e['path']).name==path.name)
        stat=path.stat()
        if stat.st_size!=entry['bytes']:raise ValueError('source size mismatch')
        sources[split]=dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,rows=entry['rows'],release_sha256=entry['sha256'])
        for i in range(nshards):tasks.append(dict(split=split,shard=i,path=str(path),start=stat.st_size*i//nshards,end=stat.st_size*(i+1)//nshards,output=str((out/split/f'shard_{i:03d}').resolve()),source_mtime_ns=stat.st_mtime_ns))
    plan=dict(release=str(release),sources=sources,vocab=voc,pattern=pattern,tasks=tasks,seed=20261008,test_opened=False)
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:raise ValueError('existing preparation plan differs')
    atomic(out/'plan.json',plan);started=time.time();receipts=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();receipts.append(r)
            atomic(out/'progress.json',dict(status='running',completed=len(receipts),total=len(tasks),rows=sum(x['rows'] for x in receipts),errors=sum(sum(x['errors'].values()) for x in receipts),seconds=time.time()-started))
            print('COMPLETE',r['task']['split'],r['task']['shard'],r['rows'],r['errors'],flush=True)
    for split,source in sources.items():
        rows=sum(r['rows'] for r in receipts if r['task']['split']==split)
        if rows!=source['rows']:raise ValueError(f'{split} count mismatch: {rows} != {source["rows"]}')
        stat=Path(source['path']).stat()
        if stat.st_size!=source['bytes'] or stat.st_mtime_ns!=source['mtime_ns']:raise ValueError('source changed during preparation')
    errors=sum(sum(r['errors'].values()) for r in receipts)
    manifest=dict(status='complete' if not errors else 'invalid',sources=sources,vocab=voc,meta_dtype=META.descr,
                  shards=sorted(receipts,key=lambda r:(r['task']['split'],r['task']['shard'])),errors=errors,
                  total_seconds=time.time()-started,test_opened=False,shuffle='seeded global row permutation generated by training, then disjoint DDP slices; packed storage order is not training order',
                  source_checksum_policy='Release manifest hash retained; file size/mtime pinned and every parsed byte region hashed. No separate whole-file checksum reread.')
    atomic(out/'manifest.json',manifest);atomic(out/'progress.json',{'status':manifest['status'],'seconds':time.time()-started,'errors':errors})
    if errors:raise RuntimeError(f'{errors} invalid rows; refusing training')

if __name__=='__main__':main()
