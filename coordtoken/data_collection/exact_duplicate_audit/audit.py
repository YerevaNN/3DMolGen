"""Read-only exact enriched_text audit, with full-string collision verification."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import itertools
import json
import mmap
import os
from pathlib import Path
import struct
import time
import numpy as np

BLOCK=8*1024*1024
RECORD=struct.Struct('<32sHQ')
DTYPE=np.dtype([('digest','V32'),('category','<u2'),('offset','<u8')])


def read(p):return json.loads(Path(p).read_text())

def atomic(p,value):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(BLOCK),b''):h.update(block)
 return h.hexdigest()

def stat_check(f):
 s=Path(f['path']).stat()
 if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns']:raise ValueError('Input changed: '+f['path'])

def prepare(release,out,workers):
 manifest=read(release/'manifest.json')
 if manifest['status']!='complete_verified_merge':raise ValueError('Merge not verified')
 receipts_path=release/'chunk_receipts.json'
 if sha(receipts_path)!=manifest['chunk_receipts_sha256']:raise ValueError('Merge receipts changed')
 receipts=read(receipts_path);prior=read(manifest['prior_combined_manifest'])
 files=[];regions=[];categories=[];lookup={}
 for i,role in enumerate(('train','val','test')):
  f=next(f for f in manifest['files'] if Path(f['path']).name==f'merged_{role}.csv')
  item=dict(f,role=role,mtime_ns=Path(f['path']).stat().st_mtime_ns,text_col=1,merged=True)
  stat_check(item);files.append(item)
  with open(f['path'],'rb') as stream:header=stream.readline()
  if next(csv.reader([header.decode()]))!=manifest['schema']:raise ValueError('CSV schema changed')
  start=len(header);n=0
  for r in receipts:
   if role not in r['outputs']:continue
   part=r['outputs'][role]
   if part['bytes']:
    regions.append({'file_index':i,'start':start,'end':start+part['bytes'],
                    'rows':part['rows'],'sha256':part['sha256']})
   start+=part['bytes'];n+=part['rows']
  if start!=f['bytes'] or n!=f['rows']:raise ValueError('Merge region accounting mismatch')
 for f in prior['files']:
  name=f['dataset'];subset=''
  if name.startswith('nablaDFT_'):subset=name.removeprefix('nablaDFT_');name='nablaDFT'
  key=(f['role'],name,subset)
  if key in lookup:raise ValueError('Duplicate category')
  cid=len(categories);lookup[key]=cid
  categories.append({'id':cid,'dataset':name,'test_subset':subset,'split':f['role'],
                     'file_index':('train','val','test').index(f['role']),'rows':f['rows']})
 ref=prior['geom_revisited_reference'];cid=len(categories)
 categories.append({'id':cid,'dataset':'GEOM-Revisited','test_subset':'','split':'test','file_index':3,'rows':ref['rows']})
 ref=dict(ref,role='test',merged=False,text_col=1,category=cid)
 stat_check(ref);files.append(ref)
 # Small external benchmark: hash the actual file and derive a body region.
 with open(ref['path'],'rb') as f:header=f.readline();body=f.read()
 if hashlib.sha256(header+body).hexdigest()!=ref['sha256']:raise ValueError('GEOM reference changed')
 if next(csv.reader([header.decode()]))!=['name','enriched_text']:raise ValueError('Reference header')
 regions.append({'file_index':3,'start':len(header),'end':ref['bytes'],'rows':ref['rows'],
                 'sha256':hashlib.sha256(body).hexdigest()})
 # Distribute byte-balanced tasks, retaining source offsets for exact comparison.
 assignments=[[] for _ in range(workers)];loads=[0]*workers
 for region in sorted(regions,key=lambda r:r['end']-r['start'],reverse=True):
  worker=min(range(workers),key=lambda i:loads[i]);assignments[worker].append(region)
  loads[worker]+=region['end']-region['start']
 return {'files':files,'categories':categories,'assignments':assignments,
         'expected_rows':sum(f['rows'] for f in files),'release_manifest':str(release/'manifest.json'),
         'release_manifest_sha256':sha(release/'manifest.json'),'receipt_sha256':manifest['chunk_receipts_sha256'],
         'equality':'exact enriched_text field equality; no canonicalization, rounding, or name comparison',
         'inputs_modified':False}


def scan_worker(args):
 wid,regions,plan,out=args;out=Path(out)/'scan'/f'{wid:03d}';out.mkdir(parents=True)
 lookup={(c['split'],c['dataset'],c['test_subset']):c['id'] for c in plan['categories']}
 buffers=[bytearray() for _ in range(256)];counts=Counter();n=0;started=time.monotonic()
 for ri,region in enumerate(regions):
  item=plan['files'][region['file_index']];stat_check(item);h=hashlib.sha256();rows=0
  with open(item['path'],'rb',buffering=BLOCK) as f:
   f.seek(region['start'])
   while f.tell()<region['end']:
    offset=f.tell();raw=f.readline();h.update(raw);rows+=1
    row=next(csv.reader([raw.decode('utf-8')],strict=True))
    if item['merged']:
     if len(row)!=8:raise ValueError('Merged column count')
     cid=lookup[(item['role'],row[2],row[7])]
    else:
     if len(row)!=2:raise ValueError('Reference column count')
     cid=item['category']
    text=row[1]
    if not text:raise ValueError('Empty enriched_text')
    digest=hashlib.sha256(text.encode('utf-8')).digest()
    buffers[digest[0]].extend(RECORD.pack(digest,cid,offset));counts[cid]+=1;n+=1
   if f.tell()!=region['end']:raise ValueError('Region boundary mismatch')
  if rows!=region['rows'] or h.hexdigest()!=region['sha256']:raise ValueError('Merged region changed')
  stat_check(item)
  atomic(out/'progress.json',{'regions':ri+1,'total_regions':len(regions),'rows':n,'seconds':time.monotonic()-started})
 parts={}
 for b,buf in enumerate(buffers):
  p=out/f'{b:02x}.bin';p.write_bytes(buf)
  parts[b]={'path':str(p),'bytes':len(buf),'sha256':hashlib.sha256(buf).hexdigest()}
 result={'worker':wid,'rows':n,'category_counts':dict(counts),'parts':parts,'seconds':time.monotonic()-started}
 atomic(out/'report.json',result);return result


def analyze_bucket(args):
 bucket,parts,plan,out=args;out=Path(out)/'buckets';out.mkdir(parents=True,exist_ok=True)
 arrays=[]
 for p in parts:
  if sha(p['path'])!=p['sha256'] or Path(p['path']).stat().st_size!=p['bytes']:raise ValueError('Digest partition changed')
  arrays.append(np.fromfile(p['path'],dtype=DTYPE))
 records=np.concatenate(arrays);del arrays
 records.sort(order=['digest','category','offset'])
 repeated=np.flatnonzero(records['digest'][1:]==records['digest'][:-1])
 if len(repeated):
  cuts=np.flatnonzero(np.diff(repeated)>1)+1
  starts=np.r_[repeated[0],repeated[cuts]];ends=np.r_[repeated[cuts-1]+2,repeated[-1]+2]
 else:starts=ends=[]
 cats=plan['categories'];streams=[];maps=[]
 for f in plan['files']:
  stat_check(f);stream=open(f['path'],'rb');streams.append(stream);maps.append(mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ))
 def fetch(record):
  cid=int(record['category']);idx=cats[cid]['file_index'];mm=maps[idx];mm.seek(int(record['offset']))
  row=next(csv.reader([mm.readline().decode('utf-8')],strict=True))
  return row[1],row[0]
 summary={'bucket':bucket,'rows_scanned':len(records),'duplicate_text_groups':0,
          'rows_in_duplicate_groups':0,'redundant_rows_global':0,'hash_collision_groups':0}
 within_split=Counter();within_category=Counter();split_pairs=Counter();category_pairs=Counter()
 examples=[];pair_examples={};same_split_examples={}
 with open(out/f'{bucket:02x}_groups.jsonl','w') as ledger:
  for start,end in zip(starts,ends):
   candidate=records[start:end];exact={}
   for record in candidate:
    text,name=fetch(record)
    exact.setdefault(text,[]).append((int(record['category']),int(record['offset']),name))
   if len(exact)>1:summary['hash_collision_groups']+=1
   for text,members in exact.items():
    if len(members)<2:continue
    count=len(members);summary['duplicate_text_groups']+=1
    summary['rows_in_duplicate_groups']+=count;summary['redundant_rows_global']+=count-1
    cc=Counter(m[0] for m in members);sc=Counter()
    for cid,n in cc.items():
     sc[cats[cid]['split']]+=n;within_category[cid]+=n-1
    for split,n in sc.items():within_split[split]+=n-1
    for a,b in itertools.combinations(sorted(sc),2):split_pairs[f'{a}|{b}']+=1
    for a,b in itertools.combinations(sorted(cc),2):category_pairs[f'{a}|{b}']+=1
    digest=hashlib.sha256(text.encode()).hexdigest()
    entry={'sha256':digest,'rows':count,'categories':dict(cc),'occurrences':[
     {'category':cid,'byte_offset':offset,'name':name} for cid,offset,name in members]}
    ledger.write(json.dumps(entry)+'\n')
    example={'sha256':digest,'rows':count,'categories':dict(cc),'enriched_text':text,
             'occurrences':entry['occurrences'][:6]}
    if len(examples)<3:examples.append(example)
    for a,b in itertools.combinations(sorted(sc),2):pair_examples.setdefault(f'{a}|{b}',example)
    for split,n in sc.items():
     if n>1:same_split_examples.setdefault(split,example)
 for mm in maps:mm.close()
 for f in streams:f.close()
 summary.update(within_split_redundant_rows=dict(within_split),within_category_redundant_rows=dict(within_category),
                split_pair_shared_texts=dict(split_pairs),category_pair_shared_texts=dict(category_pairs),
                examples=examples,split_pair_examples=pair_examples,within_split_examples=same_split_examples)
 atomic(out/f'{bucket:02x}_report.json',summary);return summary


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--workers',type=int,default=64);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();state=out/'progress.json'
 try:
  plan=prepare(args.release,out,args.workers);atomic(out/'plan.json',plan)
  atomic(state,{'phase':'scan','workers':args.workers,'expected_rows':plan['expected_rows']})
  scans=[]
  with ProcessPoolExecutor(args.workers) as pool:
   futures=[pool.submit(scan_worker,(i,regions,plan,str(out))) for i,regions in enumerate(plan['assignments'])]
   for future in as_completed(futures):
    scans.append(future.result());atomic(state,{'phase':'scan','completed_workers':len(scans),
      'workers':args.workers,'completed_worker_rows':sum(s['rows'] for s in scans),'seconds':time.monotonic()-started})
  scans.sort(key=lambda s:s['worker']);atomic(out/'scan_receipts.json',scans)
  counts=Counter()
  for s in scans:counts.update({int(k):v for k,v in s['category_counts'].items()})
  if sum(counts.values())!=plan['expected_rows']:raise ValueError('Audit row count mismatch')
  for c in plan['categories']:
   if counts[c['id']]!=c['rows']:raise ValueError('Dataset count mismatch')
  atomic(state,{'phase':'confirm_exact_duplicates','completed_buckets':0,'total_buckets':256,'seconds':time.monotonic()-started})
  summaries=[]
  with ProcessPoolExecutor(min(args.workers,32)) as pool:
   futures=[pool.submit(analyze_bucket,(b,[s['parts'][b] for s in scans],plan,str(out))) for b in range(256)]
   for future in as_completed(futures):
    summaries.append(future.result());atomic(state,{'phase':'confirm_exact_duplicates','completed_buckets':len(summaries),
     'total_buckets':256,'confirmed_duplicate_text_groups':sum(s['duplicate_text_groups'] for s in summaries),'seconds':time.monotonic()-started})
  report={k:sum(s[k] for s in summaries) for k in ['rows_scanned','duplicate_text_groups','rows_in_duplicate_groups','redundant_rows_global','hash_collision_groups']}
  for key in ['within_split_redundant_rows','within_category_redundant_rows','split_pair_shared_texts','category_pair_shared_texts']:
   c=Counter()
   for s in summaries:c.update(s[key])
   report[key]=dict(c)
  report['examples']=[];report['split_pair_examples']={};report['within_split_examples']={}
  for s in sorted(summaries,key=lambda s:s['bucket']):
   if len(report['examples'])<10:report['examples'].extend(s['examples'][:10-len(report['examples'])])
   for k,v in s['split_pair_examples'].items():report['split_pair_examples'].setdefault(k,v)
   for k,v in s['within_split_examples'].items():report['within_split_examples'].setdefault(k,v)
  for f in plan['files']:stat_check(f)
  if sha(plan['release_manifest'])!=plan['release_manifest_sha256']:raise ValueError('Release manifest changed')
  report.update(status='complete_verified_exact_duplicate_audit',seconds=time.monotonic()-started,
   categories=plan['categories'],inputs=plan['files'],inputs_modified=False,equality=plan['equality'],
   all_hash_candidates_compared_as_full_strings=True,plan_sha256=sha(out/'plan.json'))
  atomic(out/'report.json',report);atomic(state,{'phase':'complete','seconds':report['seconds'],
    'duplicate_text_groups':report['duplicate_text_groups'],'redundant_rows_global':report['redundant_rows_global']})
  print(json.dumps({k:v for k,v in report.items() if k not in ('examples','inputs','categories','split_pair_examples','within_split_examples','category_pair_shared_texts')}),flush=True)
 except BaseException as e:
  atomic(state,{'phase':'failed','error':repr(e),'seconds':time.monotonic()-started});raise

if __name__=='__main__':main()
