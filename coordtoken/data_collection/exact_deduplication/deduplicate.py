"""Remove audited exact repeats within merged splits, preserving all provenance."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'merge_release'))
import merge

DROP=struct.Struct('<QQ32s')
DTYPE=np.dtype([('offset','<u8'),('keeper','<u8'),('digest','V32')])
ALIAS_HEADER=['removed_name','kept_name','dataset','source_name','source_file_id','source_row_index','inchikey','test_subset','split']
ALIAS_HEADER_BYTES=(','.join(ALIAS_HEADER)+'\n').encode()
read=merge.read;sha=merge.sha;atomic=merge.atomic;BLOCK=merge.BLOCK


def prepare(release,audit,out):
 manifest=read(release/'manifest.json');report=read(audit/'report.json')
 if manifest['status']!='complete_verified_merge' or report['status']!='complete_verified_exact_duplicate_audit':raise ValueError('Required prior checks incomplete')
 if report['split_pair_shared_texts']:raise ValueError('Cross-split duplicate policy requires review')
 if sha(release/'chunk_receipts.json')!=manifest['chunk_receipts_sha256']:raise ValueError('Merge receipts changed')
 if sha(release/'sources.json')!=manifest['sources_sha256']:raise ValueError('Source manifest changed')
 ap=read(audit/'plan.json')
 if sha(audit/'plan.json')!=report['plan_sha256'] or sha(release/'manifest.json')!=ap['release_manifest_sha256']:raise ValueError('Audit inputs changed')
 files=[]
 for role in ('train','val','test'):
  f=next(f for f in manifest['files'] if Path(f['path']).name==f'merged_{role}.csv')
  audited=next(i for i in report['inputs'] if i['path']==f['path'])
  merge.check_source(audited);files.append(dict(f,role=role,mtime_ns=audited['mtime_ns']))
 cats=report['categories'];buffers=[bytearray() for _ in files];ledger_hashes={}
 group_count=group_rows=0
 for bucket in range(256):
  path=audit/'buckets'/f'{bucket:02x}_groups.jsonl';h=hashlib.sha256()
  with path.open('rb') as f:
   for raw in f:
    h.update(raw);g=json.loads(raw);group_count+=1;group_rows+=g['rows'];members={}
    if g['rows']!=len(g['occurrences']):raise ValueError('Malformed duplicate ledger')
    for occurrence in g['occurrences']:
     fi=cats[occurrence['category']]['file_index']
     if fi<3:members.setdefault(fi,[]).append(occurrence['byte_offset'])
    for fi,offsets in members.items():
     offsets.sort()
     for offset in offsets[1:]:buffers[fi].extend(DROP.pack(offset,offsets[0],bytes.fromhex(g['sha256'])))
  ledger_hashes[str(path)]=h.hexdigest()
 if group_count!=report['duplicate_text_groups'] or group_rows!=report['rows_in_duplicate_groups']:raise ValueError('Duplicate ledger accounting mismatch')
 deletions=[]
 for fi,buf in enumerate(buffers):
  a=np.frombuffer(buf,dtype=DTYPE).copy();a.sort(order='offset')
  if len(a)>1 and np.any(a['offset'][1:]<=a['offset'][:-1]):raise ValueError('Repeated deletion offset')
  if np.any(np.isin(a['keeper'],a['offset'])):raise ValueError('Retained owner marked for removal')
  path=out/f"drop_{files[fi]['role']}.npy";np.save(path,a)
  deletions.append({'path':str(path),'rows':len(a),'sha256':sha(path)})
 for fi,role in enumerate(('train','val')):
  if deletions[fi]['rows']!=report['within_split_redundant_rows'].get(role,0):raise ValueError('Expected duplicate count mismatch')
 sources=read(release/'sources.json');by_id={f['source_file_id']:f for f in sources['files']}
 receipts=read(release/'chunk_receipts.json');tasks=[]
 for fi,f in enumerate(files):
  with open(f['path'],'rb') as stream:header=stream.readline()
  if header!=merge.HEADER_BYTES:raise ValueError('Merged header changed')
  offset=len(header);a=np.load(deletions[fi]['path'],mmap_mode='r')
  for r in receipts:
   if f['role'] not in r['outputs']:continue
   part=r['outputs'][f['role']];start=offset;offset+=part['bytes']
   if not part['bytes']:continue
   source=by_id[r['source_file_id']]
   lo=int(np.searchsorted(a['offset'],start));hi=int(np.searchsorted(a['offset'],offset))
   tasks.append({'id':len(tasks),'file':f,'file_index':fi,'start':start,'end':offset,'rows':part['rows'],
    'sha256':part['sha256'],'drops':deletions[fi]['path'],'drop_start':lo,'drop_end':hi,
    'source_file_id':r['source_file_id'],'dataset':source['output_dataset'],'test_subset':source['test_subset'],
    'output':str(out/'parts'/f'{len(tasks):05d}')})
  if offset!=f['bytes']:raise ValueError('Region boundary total mismatch')
 expected={f['role']:f['rows']-deletions[i]['rows'] for i,f in enumerate(files)}
 shutil.copyfile(release/'sources.json',out/'sources.json')
 prior=read(manifest['prior_combined_manifest'])
 return {'files':files,'tasks':tasks,'deletions':deletions,'expected_rows':expected,'categories':cats,
  'reference':prior['geom_revisited_reference'],'source_manifest_sha256':manifest['sources_sha256'],
  'parent_release':str(release),'parent_manifest_sha256':sha(release/'manifest.json'),
  'audit':str(audit),'audit_report_sha256':sha(audit/'report.json'),'ledger_hashes':ledger_hashes,
  'policy':'Exact enriched_text equality within each merged split; earliest input byte offset retained; GEOM-Revisited preserved separately.',
  'input_files_modified':False}


def filter_region(t):
 f=t['file'];merge.check_source(f);out=Path(t['output']);out.mkdir(parents=True,exist_ok=False)
 drops=np.load(t['drops'],mmap_mode='r')[t['drop_start']:t['drop_end']]
 di=0;rows=kept=removed=0;h=hashlib.sha256();kh=hashlib.sha256();alias_buffer=io.StringIO()
 path=out/(f['role']+'.csv');aliases=out/'aliases.csv'
 with open(f['path'],'rb',buffering=BLOCK) as src,open(f['path'],'rb') as owner,\
      path.open('wb',buffering=BLOCK) as dst,aliases.open('w',newline='',buffering=BLOCK) as af:
  mm=mmap.mmap(owner.fileno(),0,access=mmap.ACCESS_READ)
  writer=csv.writer(af,lineterminator='\n');src.seek(t['start'])
  try:
   while src.tell()<t['end']:
    offset=src.tell();raw=src.readline();rows+=1;h.update(raw)
    if di<len(drops) and int(drops[di]['offset'])==offset:
     deletion=drops[di];di+=1;row=next(csv.reader([raw.decode('utf-8')],strict=True))
     mm.seek(int(deletion['keeper']));keeper=next(csv.reader([mm.readline().decode('utf-8')],strict=True))
     if len(row)!=8 or len(keeper)!=8 or row[1]!=keeper[1] or row[6]!=keeper[6]:raise ValueError('Full-string duplicate verification failed')
     if hashlib.sha256(row[1].encode()).digest()!=deletion['digest'].tobytes():raise ValueError('Duplicate digest mismatch')
     if row[0]==keeper[0]:raise ValueError('Same source record cannot alias itself')
     writer.writerow([row[0],keeper[0],*row[2:],f['role']]);removed+=1
    else:
     if di<len(drops) and int(drops[di]['offset'])<offset:raise ValueError('Deletion does not address a record boundary')
     dst.write(raw);kh.update(raw);kept+=1
   if src.tell()!=t['end']:raise ValueError('Region overrun')
  finally:mm.close()
 if rows!=t['rows'] or di!=len(drops) or h.hexdigest()!=t['sha256']:raise ValueError('Source region verification failed')
 merge.check_source(f)
 r={'id':t['id'],'source_file_id':t['source_file_id'],'input_rows':rows,'removed_rows':removed,
    'dataset':t['dataset'],'test_subset':t['test_subset'],'role':f['role'],
    'outputs':{f['role']:{'path':str(path),'bytes':path.stat().st_size,'rows':kept,'sha256':kh.hexdigest()}},
    'aliases':{'path':str(aliases),'bytes':aliases.stat().st_size,'rows':removed,'sha256':sha(aliases)}}
 atomic(out/'report.json',r);return r


def verify_region(job):
 target,start,part=job;h=hashlib.sha256();remaining=part['bytes'];rows=0
 with open(target,'rb',buffering=BLOCK) as f:
  f.seek(start)
  while remaining:
   block=f.read(min(BLOCK,remaining))
   if not block:raise ValueError('Truncated assembled file')
   h.update(block);rows+=block.count(b'\n');remaining-=len(block)
 if h.hexdigest()!=part['sha256'] or rows!=part['rows']:raise ValueError('Output readback mismatch')
 return rows


def file_receipt(job):
 path,rows=job
 return {'path':str(path),'rows':rows,'bytes':Path(path).stat().st_size,'sha256':sha(path)}


def assemble(out,results,workers,state):
 copy_jobs=[];verify_jobs=[];totals=Counter()
 for role in ('train','val','test','aliases'):
  target=out/('duplicate_aliases.csv' if role=='aliases' else 'merged_'+role+'.csv')
  header=ALIAS_HEADER_BYTES if role=='aliases' else merge.HEADER_BYTES;offset=len(header)
  for r in results:
   p=r['aliases'] if role=='aliases' else r['outputs'].get(role)
   if p is None:continue
   totals[role]+=p['rows']
   if p['bytes']:
    copy_jobs.append((p['path'],str(target),offset,p['bytes'],p['sha256']))
    verify_jobs.append((str(target),offset,p))
   offset+=p['bytes']
  with target.open('xb') as f:f.write(header);f.truncate(offset)
 with ProcessPoolExecutor(min(workers,32)) as pool:
  for i,n in enumerate(pool.map(merge.copy_region,copy_jobs),1):
   atomic(state,{'phase':'assemble','completed_parts':i,'total_parts':len(copy_jobs)})
 atomic(state,{'phase':'verify_output_regions'})
 with ProcessPoolExecutor(workers) as pool:
  for i,n in enumerate(pool.map(verify_region,verify_jobs),1):
   if i%100==0:atomic(state,{'phase':'verify_output_regions','completed_regions':i,'total_regions':len(verify_jobs)})
 atomic(state,{'phase':'whole_file_checksums'})
 with ProcessPoolExecutor(4) as pool:
  receipts=list(pool.map(file_receipt,[(out/('duplicate_aliases.csv' if r=='aliases' else 'merged_'+r+'.csv'),totals[r]) for r in ('train','val','test','aliases')]))
 return receipts


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=64)
 args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);state=out/'progress.json';started=time.monotonic()
 try:
  atomic(state,{'phase':'prepare_drop_plan'});plan=prepare(args.release,args.audit,out);atomic(out/'plan.json',plan)
  results=[]
  with ProcessPoolExecutor(args.workers) as pool:
   futures=[pool.submit(filter_region,t) for t in plan['tasks']]
   for future in as_completed(futures):
    results.append(future.result());atomic(state,{'phase':'remove_exact_repeats','completed_tasks':len(results),'total_tasks':len(futures),'seconds':time.monotonic()-started})
  results.sort(key=lambda r:r['id']);atomic(out/'chunk_receipts.json',results)
  removed=Counter();retained=Counter();category_counts=Counter()
  for r in results:
   removed[r['role']]+=r['removed_rows'];n=r['outputs'][r['role']]['rows'];retained[r['role']]+=n
   category_counts[(r['role'],r['dataset'],r['test_subset'])]+=n
  if dict(retained)!=plan['expected_rows']:raise ValueError('Retained row count mismatch')
  receipts=assemble(out,results,args.workers,state)
  for f in plan['files']:merge.check_source(f)
  categories=[]
  for c in plan['categories']:
   if c['file_index']==3:continue
   dataset=c['dataset']+('_'+c['test_subset'] if c['test_subset'] else '')
   categories.append({'dataset':dataset,'role':c['split'],'rows':category_counts[(c['split'],c['dataset'],c['test_subset'])]})
  atomic(out/'category_manifest.json',{'status':'molecular_split_verified','kind':'row categories within merged CSV files',
    'files':categories,'geom_revisited_reference':plan['reference'],'exact_sample_deduplication_performed':True,
    'conformation_test_molecular_overlap_exception':True})
  manifest={'status':'complete_verified_merge','exact_sample_deduplication_performed':True,
    'deduplication_policy':plan['policy'],'files':receipts[:3],'duplicate_aliases':receipts[3],
    'removed_rows':dict(removed),'seconds':time.monotonic()-started,'workers':args.workers,'schema':merge.HEADER,
    'sources_manifest':str(out/'sources.json'),'sources_sha256':sha(out/'sources.json'),
    'chunk_receipts_sha256':sha(out/'chunk_receipts.json'),'plan_sha256':sha(out/'plan.json'),
    'prior_combined_manifest':str(out/'category_manifest.json'),'parent_release':str(args.release),
    'checks':{'removed_records_equal_retained_full_text':True,'removed_and_retained_inchikeys_equal':True,
      'source_region_hashes':True,'row_counts':True,'output_region_readback':True,'all_removed_provenance_retained':True},
    'shuffle_performed':False,'conformation_test_molecular_overlap_exception':True,
    'post_deduplication_exact_audit':'pending'}
  atomic(out/'manifest.json',manifest);atomic(state,{'phase':'complete','seconds':manifest['seconds'],'removed_rows':dict(removed)})
  print(json.dumps(manifest),flush=True)
 except BaseException as e:
  atomic(state,{'phase':'failed','error':repr(e),'seconds':time.monotonic()-started});raise

if __name__=='__main__':main()
