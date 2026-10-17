"""Materialize approved split CSVs with exact prepared-input row provenance.

No resplitting, shuffling or deduplication. Parallel chunks align existing
kept/excluded records against immutable sources and row-aligned key sidecars.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time

HEADER = ['name', 'enriched_text', 'dataset', 'source_name', 'source_file_id',
          'source_row_index', 'inchikey', 'test_subset']
HEADER_BYTES = (','.join(HEADER) + '\n').encode()
ORIGINAL_HEADER = b'name,enriched_text\n'
BLOCK = 8 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(BLOCK), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def check_source(item):
    s = Path(item['path']).stat()
    if (s.st_size, s.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
        raise ValueError('Source changed: ' + item['path'])


def source_rows(item, task):
    with open(item['path'], 'rb', buffering=BLOCK) as f:
        if next(csv.reader([f.readline().decode('utf-8')])) != ['name', 'enriched_text']:
            raise ValueError('Unexpected source header')
        if task['start']:
            f.seek(task['start'] - 1)
            f.readline()
        while f.tell() < task['end']:
            line = f.readline()
            if not line:
                break
            yield line


def run_chunk(task):
    item = task['source']
    check_source(item)
    out = Path(task['output'])
    out.mkdir(parents=True, exist_ok=False)
    streams = {}; writers = {}; heads = {}; parts = {}; part_hashes = {}
    counts = Counter(); source_hash = hashlib.sha256(); key_hash = hashlib.sha256()
    key_stream = None
    inspect = None
    started = time.monotonic()
    try:
        if task.get('keys'):
            key_stream = open(task['keys'], 'rb', buffering=BLOCK)
        else:
            from rdkit import RDLogger, rdBase
            RDLogger.DisableLog('rdApp.*')
            if rdBase.rdkitVersion != task['rdkit_version']:
                raise ValueError('RDKit version changed')
            spec = importlib.util.spec_from_file_location('merge_frozen_parser', task['parser_path'])
            mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            vocab = set(task['vocab'])
            inspect = lambda text: mod.inspect(text, vocab)[2]['key']
        for role, info in task.get('parts', {}).items():
            parts[role] = open(info['path'], 'rb', buffering=BLOCK)
            heads[role] = parts[role].readline()
            part_hashes[role] = hashlib.sha256()
        roles = [r for r in ('train', 'val', 'test') if r in parts or r == task.get('fixed_role')]
        for role in roles:
            streams[role] = open(out / (role + '.csv'), 'w', newline='', buffering=BLOCK)
            writers[role] = csv.writer(streams[role], lineterminator='\n')
        total = 0
        for raw in source_rows(item, task):
            source_hash.update(raw)
            row_index = task['row_start'] + total
            total += 1
            key_line = key_stream.readline() if key_stream else None
            if key_stream:
                if not key_line:
                    raise ValueError('Source has more rows than key index')
                key_hash.update(key_line)
                key = key_line.rstrip(b'\n').decode('ascii')
            if parts:
                matching = [role for role, head in heads.items() if head == raw]
                if len(matching) != 1:
                    raise ValueError(f'Nonunique source-to-split alignment at {row_index}: {matching}')
                role = matching[0]
                part_hashes[role].update(raw)
                heads[role] = parts[role].readline()
            else:
                role = task['fixed_role']
            counts[role] += 1
            if role == 'excluded':
                continue
            row = next(csv.reader([raw.decode('utf-8')], strict=True))
            if len(row) != 2:
                raise ValueError('Unexpected source column count')
            original_name, enriched_text = row
            if not key_stream:
                key = inspect(enriched_text)
            if len(key) != 27:
                raise ValueError('Missing full InChIKey on retained row')
            record_id = f"{item['source_file_id']}:{row_index}"
            writers[role].writerow([record_id, enriched_text, item['output_dataset'], original_name,
                                   item['source_file_id'], row_index, key, item['test_subset']])
        if key_stream and key_stream.readline():
            raise ValueError('Key index has more rows than source')
        if any(heads.values()):
            raise ValueError('Unconsumed split records')
        if total != task['rows'] or source_hash.hexdigest() != task['source_record_sha256']:
            raise ValueError('Source chunk hash or row count mismatch')
        if key_stream and key_hash.hexdigest() != task['keys_sha256']:
            raise ValueError('Key sidecar changed')
        if dict(counts) != {k:v for k,v in task['expected_counts'].items() if v}:
            raise ValueError(f'Incorrect split row counts: {counts} vs {task["expected_counts"]}')
        for role, h in part_hashes.items():
            if h.hexdigest() != task['parts'][role]['sha256']:
                raise ValueError('Existing split part hash mismatch')
    finally:
        for f in list(streams.values()) + list(parts.values()) + ([key_stream] if key_stream else []):
            f.close()
    check_source(item)
    result = {'id': task['id'], 'source_file_id': item['source_file_id'],
              'row_start': task['row_start'], 'input_rows': total, 'counts': dict(counts),
              'source_record_sha256': source_hash.hexdigest(), 'outputs': {},
              'seconds': time.monotonic()-started}
    for role in roles:
        p = out / (role + '.csv')
        result['outputs'][role] = {'path': str(p), 'bytes': p.stat().st_size,
                                   'rows': counts[role], 'sha256': sha(p)}
    atomic(out / 'report.json', result)
    return result


def copy_region(job):
    src, target, offset, size, digest = job
    h = hashlib.sha256();written = 0
    fd = os.open(target, os.O_WRONLY)
    try:
        with open(src, 'rb', buffering=BLOCK) as f:
            for block in iter(lambda: f.read(BLOCK), b''):
                h.update(block)
                view = memoryview(block)
                while view:
                    n = os.pwrite(fd, view, offset+written)
                    if n <= 0:
                        raise OSError('Short output write')
                    written += n;view = view[n:]
    finally:
        os.close(fd)
    if written != size or h.hexdigest() != digest:
        raise ValueError('Merge part changed during assembly')
    return written


def verify_output(job):
    path, regions, expected_rows = job
    h = hashlib.sha256();counts = 0
    with open(path, 'rb', buffering=BLOCK) as f:
        header = f.read(len(HEADER_BYTES))
        if header != HEADER_BYTES:
            raise ValueError('Merged header changed')
        h.update(header)
        for region in regions:
            rh = hashlib.sha256();remaining = region['bytes'];lines = 0
            while remaining:
                block = f.read(min(BLOCK, remaining))
                if not block:
                    raise ValueError('Truncated merged file')
                remaining -= len(block);h.update(block);rh.update(block);lines += block.count(b'\n')
            if rh.hexdigest() != region['sha256'] or lines != region['rows']:
                raise ValueError('Merged region readback mismatch')
            counts += lines
        if f.read(1) or counts != expected_rows:
            raise ValueError('Merged row accounting mismatch')
    return {'path':str(path), 'rows':counts, 'bytes':Path(path).stat().st_size, 'sha256':h.hexdigest()}


def prepare(root, output, code):
    arun=root/'group_a_20261006/final/run';brun=root/'group_b_20261006/split_run'
    irun=root/'group_b_20261006/index_run';crun=root/'group_c_20261006/run'
    am=read(arun/'manifest.json');im=read(irun/'manifest.json')
    combined=read(crun/'combined_split_manifest.json')
    if combined['status'] != 'molecular_split_verified':raise ValueError('Splits not complete')
    sources=[];tasks=[];evidence={};used=set()
    def pin(p):
        p=Path(p);evidence[str(p)]=sha(p)
    for p in [arun/'manifest.json',arun/'report.json',irun/'manifest.json',irun/'report.json',
              irun/'receipts.json',brun/'report.json',crun/'report.json',crun/'combined_split_manifest.json',code]:pin(p)
    receipts={r['task_id']:r for r in read(irun/'receipts.json')}
    def add_source(item, group):
        s=dict(item)
        sid='s'+hashlib.sha256((item['path']+'\0'+item['sha256']).encode()).hexdigest()[:16]
        if sid in used:raise ValueError('Duplicate source identity')
        used.add(sid)
        dataset=item['dataset'];subset=''
        if dataset.startswith('nablaDFT_'):
            subset=dataset.removeprefix('nablaDFT_');dataset='nablaDFT'
        s.update(source_file_id=sid,group=group,output_dataset=dataset,test_subset=subset)
        check_source(s);sources.append(s);return s
    def add_indexed(item, manifest, fi, index_run, part_run, kind, group):
        s=add_source(item,group);row_start=0
        for t in sorted((t for t in manifest['tasks'] if t['file_index']==fi),key=lambda t:t['start']):
            index=index_run/'index'/f"{t['id']:04d}"
            ip=index/'report.json';r=read(ip)
            if index_run==irun:
                receipt=receipts[t['id']]
                # Receipt's report hash binds the row-aligned index to completed extraction.
                if sha(ip)!=receipt['report_sha256']:raise ValueError('Index receipt mismatch')
            pin(ip)
            job=dict(t,id=len(tasks),source=s,row_start=row_start,rows=r['rows'],
                     source_record_sha256=r['source_record_sha256'],keys=str(index/'keys.txt'),
                     keys_sha256=r['keys_sha256'],parts={})
            if kind=='fixed':
                job.update(fixed_role='test',expected_counts={'test':r['rows']})
            else:
                folder=part_run/'parts'/f"{t['id']:04d}";rp=folder/'report.json';pr=read(rp);pin(rp)
                if kind=='a':
                    role=item['role'];kept=pr['counts'].get('keep',0);excluded=r['rows']-kept
                    job['parts']={role:{'path':str(folder/'kept.csv'),'sha256':pr['kept_sha256']},
                                  'excluded':{'path':str(folder/'excluded.csv'),'sha256':pr['excluded_sha256']}}
                    job['expected_counts']={role:kept,'excluded':excluded}
                else:
                    job['parts']={role:{'path':str(folder/(role+'.csv')),'sha256':pr['sha256'][role]}
                                  for role in ('train','val','test','excluded')}
                    c={role:pr['counts'].get(role,0) for role in ('train','val','test')}
                    c['excluded']=r['rows']-sum(c.values());job['expected_counts']=c
            job['output']=str(output/'parts'/f"{job['id']:05d}")
            tasks.append(job);row_start+=r['rows']
        if row_start!=item['rows']:raise ValueError('Source index row total mismatch')
    for fi,item in enumerate(am['files']):
        if item['group_a']:add_indexed(item,am,fi,arun,arun,'a','A')
    for fi,item in enumerate(im['files']):
        group='B' if item['kind']=='group_b_pool' else 'C'
        add_indexed(item,im,fi,irun,brun if group=='B' else crun,'bc',group)
    for fi,item in enumerate(am['files']):
        if item['dataset'] in ('nablaDFT_scaffolds','nablaDFT_structures'):
            add_indexed(item,am,fi,arun,None,'fixed','C')
    contract=read(root/'group_c_20261006/contract.json')
    item=next(i for i in contract['nabla_tests'] if i['dataset']=='nablaDFT_conformations')
    s=add_source(dict(item,role='test'),'C');pin(am['parser_path'])
    # This exempt test had no molecular exclusion index. Generate its keys once,
    # in parallel, while adding provenance; no split decision depends on them.
    row_start=0
    with open(item['path'],'rb',buffering=BLOCK) as f:
        header=f.readline()
        if next(csv.reader([header.decode('utf-8')]))!=['name','enriched_text']:raise ValueError('Unexpected conformation header')
        whole=hashlib.sha256(header)
        while True:
            start=f.tell();block=f.read(8*1024*1024)
            if not block:break
            if not block.endswith(b'\n'):block+=f.readline()
            whole.update(block);n=block.count(b'\n')
            job={'id':len(tasks),'source':s,'start':start,'end':f.tell(),'row_start':row_start,'rows':n,
                 'source_record_sha256':hashlib.sha256(block).hexdigest(),'fixed_role':'test',
                 'expected_counts':{'test':n},'rdkit_version':am['rdkit_version'],
                 'parser_path':am['parser_path'],'vocab':am['vocab'],
                 'output':str(output/'parts'/f'{len(tasks):05d}')}
            tasks.append(job);row_start+=n
    if whole.hexdigest()!=item['sha256'] or row_start!=item['rows']:raise ValueError('Conformation source changed')
    expected=Counter()
    for f in combined['files']:expected[f['role']]+=f['rows']
    selected=Counter()
    for t in tasks:
        for role,n in t['expected_counts'].items():
            if role!='excluded':selected[role]+=n
    if selected!=expected:raise ValueError('Plan does not match verified final splits')
    return {'sources':sources,'tasks':tasks,'evidence_hashes':evidence,'expected_rows':dict(expected),
            'prior_combined_manifest':str(crun/'combined_split_manifest.json'),
            'schema':HEADER,'source_row_index_base':0,'source_row_index_excludes_header':True,
            'source_scope':'prepared input CSVs before this splitting/filtering run; not upstream raw database indices',
            'ordering':'source order in sources.json, then ascending source_row_index within each output split',
            'exact_sample_deduplication_performed':False,'shuffle_performed':False,
            'conformation_test_molecular_overlap_exception':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=64)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();state=out/'progress.json'
    try:
        atomic(state,{'phase':'prepare_provenance_plan'})
        plan=prepare(args.root,out,Path(__file__).resolve());atomic(out/'plan.json',plan)
        atomic(out/'sources.json',{'row_index_base':0,'row_index_excludes_header':True,
                                  'source_scope':plan['source_scope'],'files':plan['sources']})
        for p,digest in plan['evidence_hashes'].items():
            if sha(p)!=digest:raise ValueError('Plan evidence changed: '+p)
        results=[];completed_rows=0
        with ProcessPoolExecutor(args.workers) as pool:
            futures={pool.submit(run_chunk,t):t['id'] for t in plan['tasks']}
            for future in as_completed(futures):
                r=future.result();results.append(r);completed_rows+=r['input_rows']
                elapsed=time.monotonic()-started
                progress={'phase':'write_provenance_chunks','completed_tasks':len(results),'total_tasks':len(futures),
                          'processed_input_rows':completed_rows,'seconds':elapsed,'input_rows_per_second':completed_rows/elapsed}
                atomic(state,progress)
                if len(results)%25==0:print(json.dumps(progress),flush=True)
        results.sort(key=lambda r:r['id']);atomic(out/'chunk_receipts.json',results)
        jobs=[];regions={};expected=plan['expected_rows']
        for role in ('train','val','test'):
            target=out/('merged_'+role+'.csv');offset=len(HEADER_BYTES);regions[role]=[]
            for r in results:
                if role not in r['outputs']:continue
                part=r['outputs'][role];regions[role].append(part)
                jobs.append((part['path'],str(target),offset,part['bytes'],part['sha256']));offset+=part['bytes']
            with target.open('xb') as f:f.write(HEADER_BYTES);f.truncate(offset)
            if sum(r['rows'] for r in regions[role])!=expected[role]:raise ValueError('Output row total mismatch')
        with ProcessPoolExecutor(min(args.workers,32)) as pool:
            completed=0;done_bytes=0
            for size in pool.map(copy_region,jobs,chunksize=1):
                completed+=1;done_bytes+=size
                atomic(state,{'phase':'assemble_merged_files','completed_parts':completed,'total_parts':len(jobs),
                              'copied_bytes':done_bytes,'seconds':time.monotonic()-started})
        atomic(state,{'phase':'verify_merged_files','seconds':time.monotonic()-started})
        with ProcessPoolExecutor(3) as pool:
            outputs=list(pool.map(verify_output,[(out/('merged_'+r+'.csv'),regions[r],expected[r]) for r in ('train','val','test')]))
        for source in plan['sources']:check_source(source)
        report={'status':'complete_verified_merge','files':outputs,'seconds':time.monotonic()-started,
                'workers':args.workers,'schema':HEADER,'sources_manifest':str(out/'sources.json'),
                'plan_sha256':sha(out/'plan.json'),'sources_sha256':sha(out/'sources.json'),
                'chunk_receipts_sha256':sha(out/'chunk_receipts.json'),
                'checks':{'source_chunk_hashes':True,'source_to_split_exact_record_alignment':True,
                          'retained_and_excluded_part_hashes':True,'indexed_key_hashes':True,
                          'source_row_indices_before_filtering':True,'merged_region_readback':True,
                          'row_counts_match_verified_splits':True},
                'prior_combined_manifest':plan['prior_combined_manifest'],
                'exact_sample_deduplication_performed':False,'shuffle_performed':False,
                'conformation_test_molecular_overlap_exception':True}
        atomic(out/'manifest.json',report);atomic(state,{'phase':'complete',**report})
        print(json.dumps(report),flush=True)
    except BaseException as e:
        atomic(state,{'phase':'failed','error':repr(e),'seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
