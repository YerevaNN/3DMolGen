"""Finalize nabla splits without removing any Group A/B training records."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'group_b_split'))
import split_pools as b
import verify_holdout_precedence as verify
shared=b.shared


def blocked_by_b(args):
    path,candidates=args
    db=b.database(path,readonly=True)
    db.execute('ATTACH DATABASE ? AS nabla',(str(candidates),))
    keys=[x[0] for x in db.execute("SELECT c.key FROM nabla.counts c CROSS JOIN main.counts prior ON prior.key=c.key WHERE c.owner=1 AND prior.action='train'")]
    db.close()
    return keys


def select_validation(db,target,seed=42):
    db.create_function('sample_rank',2,lambda key,split:hashlib.sha256(
        f'{seed}\0nablaDFT\0{split}\0{key}'.encode()).digest(),deterministic=True)
    db.execute("UPDATE counts SET action=CASE WHEN owner=3 THEN 'excluded_test_overlap' WHEN owner=2 THEN 'excluded_val_quota' WHEN other_train=0 THEN 'eligible_val_fill' ELSE 'train' END")
    matched=db.execute('SELECT coalesce(sum(n),0) FROM counts WHERE owner=2').fetchone()[0]
    eligible=db.execute("SELECT coalesce(sum(n),0) FROM counts WHERE action='eligible_val_fill'").fetchone()[0]
    if matched<=target:
        db.execute("UPDATE counts SET action='val' WHERE owner=2");selected=matched
    else:
        selected=b.select_ranked(db,2,'excluded_val_quota','val',target)
    filled=b.select_ranked(db,1,'eligible_val_fill','val',max(0,target-selected))
    db.execute("UPDATE counts SET action='train' WHERE action='eligible_val_fill'")
    db.commit()
    return {'target_rows':target,'existing_validation_candidate_rows':matched,
            'existing_validation_selected_rows':selected,'eligible_fill_rows_absent_from_other_training':eligible,
            'new_validation_rows':filled,'selected_validation_rows':selected+filled,
            'target_shortfall':max(0,target-selected-filled)}


def copy_test(args):
    item,output=args
    shared.check_stat(item)
    dest=Path(output)/'candidate/test'/Path(item['path']).name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(item['path'],dest)
    digest=shared.sha(dest)
    if digest!=item['sha256']:raise ValueError('Fixed nabla test hash mismatch')
    shared.check_stat(item)
    return {'dataset':item['dataset'],'role':'test','path':str(dest),'rows':item['rows'],'sha256':digest}


def check_b_validation(args):
    source,counts_path=args
    db=b.database(source,readonly=True);db.execute('ATTACH DATABASE ? AS nabla',(str(counts_path),))
    keys,rows=db.execute("SELECT count(*),coalesce(sum(prior.n),0) FROM nabla.counts c CROSS JOIN main.counts prior ON prior.key=c.key WHERE c.action='val' AND prior.action='train'").fetchone()
    db.close();return {'dataset':Path(source).stem,'matching_training_keys':keys,'matching_training_rows':rows}


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);state=out/'progress.json';started=time.monotonic()
    atomic=shared.atomic
    try:
        for path,digest in args.contract['code_hashes'].items():
            if shared.sha(path)!=digest:raise ValueError('Code changed after preparation')
        arun=args.group_a;index=args.index_run;brun=args.group_b
        am=json.loads((arun/'manifest.json').read_text());im=json.loads((index/'manifest.json').read_text())
        ir=json.loads((index/'report.json').read_text());br=json.loads((brun/'report.json').read_text())
        if ir['status']!='complete_verified_index' or br['status']!='group_b_iterative_split_complete':raise ValueError('Previous stage incomplete')
        if shared.sha(index/'manifest.json')!=ir['manifest_sha256'] or shared.sha(index/'receipts.json')!=ir['receipts_sha256']:raise ValueError('Index provenance changed')
        for path,digest in args.contract['evidence_hashes'].items():
            if shared.sha(path)!=digest:raise ValueError('Previous stage evidence changed')
        fi,item=next((i,f) for i,f in enumerate(im['files']) if f['kind']=='nabla_training_reference')
        shared.verify_input(item)
        tasks=[t for t in im['tasks'] if t['file_index']==fi]
        receipts={r['task_id']:r for r in json.loads((index/'receipts.json').read_text())}
        cpath=out/'nablaDFT.sqlite';db=b.database(cpath)
        db.execute('CREATE TABLE counts(key TEXT PRIMARY KEY,n INTEGER NOT NULL,owner INTEGER NOT NULL DEFAULT 0,other_train INTEGER NOT NULL DEFAULT 0,action TEXT) WITHOUT ROWID')
        atomic(state,{'phase':'aggregate_nabla_keys'})
        for task in tasks:
            counts,r=b.counts_for(index,task,receipts[task['id']])
            db.executemany('INSERT INTO counts(key,n) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET n=n+excluded.n',counts.items());db.commit()
        db.execute('ATTACH DATABASE ? AS previous',(str(brun/'ownership.sqlite'),))
        db.execute('PRAGMA previous.cache_size=-1048576')
        db.execute('UPDATE counts SET owner=coalesce((SELECT role FROM previous.registry WHERE registry.key=counts.key),0)');db.commit()
        db.execute('DETACH DATABASE previous')
        if db.execute('SELECT count(*) FROM counts WHERE owner=0').fetchone()[0]:raise ValueError('Nabla key missing from prior registry')
        candidate_keys={r[0] for r in db.execute('SELECT key FROM counts WHERE owner=1')}
        old_test=set(json.loads((arun/'test_keys.json').read_text()));old_val=set(json.loads((arun/'validation_keys.json').read_text()))
        blocked=set();atomic(state,{'phase':'find_keys_in_other_training'})
        for task in am['tasks']:
            f=am['files'][task['file_index']]
            if f['group_a'] and f['role']=='train':
                counts,_=b.counts_for(arun,task)
                blocked.update((set(counts)&candidate_keys)-old_test-old_val)
        b_dbs=[brun/f"{d['dataset']}.sqlite" for d in br['datasets']]
        db.close()
        with ProcessPoolExecutor(4) as pool:
            for keys in pool.map(blocked_by_b,[(p,cpath) for p in b_dbs]):blocked.update(keys)
        db=b.database(cpath);db.executemany('UPDATE counts SET other_train=1 WHERE key=?',((k,) for k in blocked));db.commit()
        tests=args.contract['nabla_tests'];target=round((item['rows']+sum(f['rows'] for f in tests))*.005)
        selection=select_validation(db,target)
        if db.execute("SELECT count(*) FROM counts WHERE (action='val' AND (owner=3 OR other_train=1)) OR (action='train' AND owner>1)").fetchone()[0]:raise ValueError('Nabla ownership invariant failed')
        expected=Counter(dict(db.execute('SELECT action,sum(n) FROM counts GROUP BY action')))
        selected={r[0] for r in db.execute("SELECT key FROM counts WHERE action='val'")};db.close()
        atomic(out/'selection.json',selection);atomic(out/'validation_keys.json',sorted(selected))
        atomic(state,{'phase':'write_nabla','selection':selection})
        with ProcessPoolExecutor(args.workers) as pool:
            parts=shared.parallel(pool,b.write_chunk,[(im,t,index,out,cpath,receipts[t['id']]) for t in tasks],'write_nabla',state)
        actual=Counter()
        for r in parts:actual.update(r['counts'])
        expected['quarantine_identity_failure']=next(f['identity_failures'] for f in ir['files'] if f['dataset']=='nablaDFT')
        if actual!=expected or sum(actual.values())!=item['rows']:raise ValueError('Nabla row accounting mismatch')
        with ProcessPoolExecutor(5) as pool:
            outputs=list(pool.map(b.assemble,[('nablaDFT',role,tasks,out) for role in ('train','val')]))
            outputs+=list(pool.map(copy_test,[(f,out) for f in tests]))
        # Reparse written validation independently; use actual A/B holdout keys
        # from the completed independent audit, not only the ownership registry.
        atomic(state,{'phase':'verify_final_global_separation'})
        vfile=next(f for f in outputs if f['role']=='val');vitem={'path':vfile['path']}
        chunks=out/'validation_recheck';chunks.mkdir()
        task={'id':0,'file_index':0,'start':0,'end':Path(vfile['path']).stat().st_size}
        verify.scan((vitem,task,chunks));observed=json.loads((chunks/'0000.json').read_text())['counts']
        if set(observed)!=selected or sum(observed.values())!=vfile['rows']:raise ValueError('Actual validation differs from assigned keys')
        fresh_holdouts=sqlite3.connect(f'file:{args.audit}/actual_holdouts.sqlite?mode=ro',uri=True)
        db=b.database(cpath,readonly=True);db.execute('ATTACH DATABASE ? AS prior_holdout',(str(args.audit/'actual_holdouts.sqlite'),))
        leaks=db.execute("SELECT count(*),coalesce(sum(c.n),0) FROM prior_holdout.held h CROSS JOIN counts c ON c.key=h.key WHERE c.action='train'").fetchone()
        db.close();fresh_holdouts.close()
        if leaks!=(0,0):raise ValueError('Nabla training overlaps actual A/B holdouts')
        # Test-only references directly rebuilt in the previous audit.
        previous_files=[]
        for label,report_path in [('A',arun/'report.json'),('B',brun/'report.json')]:
            previous_files.extend(f for f in json.loads(report_path.read_text())['files'] if f['role'] in ('test','val'))
        previous_files.extend(f for f in am['files'] if not f['group_a'] and f['role']=='test')
        actual_tests=set()
        for p in (args.audit/'holdout_chunks').glob('*.json'):
            r=json.loads(p.read_text())
            if previous_files[r['file_index']]['role']=='test':actual_tests.update(r['counts'])
        if selected&actual_tests:raise ValueError('Nabla validation overlaps actual protected tests')
        a_overlap=0
        for task in am['tasks']:
            f=am['files'][task['file_index']]
            if f['group_a'] and f['role']=='train':
                counts,_=b.counts_for(arun,task)
                a_overlap+=sum(n for k,n in counts.items() if k in selected and k not in old_test and k not in old_val)
        with ProcessPoolExecutor(4) as pool:
            b_checks=list(pool.map(check_b_validation,[(p,cpath) for p in b_dbs]))
        if a_overlap or any(r['matching_training_keys'] for r in b_checks):raise ValueError('New nabla validation overlaps other training')
        # Recheck hashes for the unchanged A/B files. This ties prior independent
        # verification and current membership checks to the same materialized data.
        previous_outputs=[]
        for label,report_path in [('A',arun/'report.json'),('B',brun/'report.json')]:
            previous_outputs.extend({**f,'group':label} for f in json.loads(report_path.read_text())['files'])
        with ProcessPoolExecutor(8) as pool:list(pool.map(verify.verify_hash,previous_outputs))
        shared.check_stat(item)
        combined=[]
        for f in previous_outputs+[{**f,'group':'C'} for f in outputs]:
            combined.append({'group':f['group'],'dataset':f['dataset'],'role':f['role'],'rows':f.get('output_rows',f.get('rows')),'path':f['path'],'sha256':f['sha256']})
        atomic(out/'combined_split_manifest.json',{'status':'molecular_split_verified','files':combined,
               'geom_revisited_reference':next(f for f in am['files'] if f['dataset']=='geom_revisited'),
               'conformation_test_molecular_overlap_exception':True,'scaffold_wide_exclusion':False,
               'exact_sample_deduplication_performed':False})
        result={'status':'group_c_and_global_molecular_split_complete','selection':selection,
                'counts':dict(actual),'files':outputs,'other_training_rows_removed':0,
                'checks':{'nabla_train_vs_actual_holdouts_keys':0,'nabla_validation_vs_actual_tests_keys':0,
                          'new_nabla_validation_vs_group_a_train_rows':a_overlap,'group_b_training':b_checks,
                          'unchanged_ab_files_hash_verified':len(previous_outputs),'actual_validation_reparsed':True},
                'seconds':time.monotonic()-started,'prior_independent_audit':str(args.audit/'report.json'),
                'combined_manifest':str(out/'combined_split_manifest.json'),
                'exact_sample_deduplication_performed':False,'policy':args.contract['policy']}
        atomic(out/'report.json',result);atomic(state,{'phase':'complete','status':result['status'],'seconds':result['seconds']})
    except Exception as exc:
        atomic(state,{'phase':'failed','error':f'{type(exc).__name__}: {exc}'});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=32)
    a=p.parse_args();a.contract=json.loads(a.contract.read_text())
    for k in ('group_a','group_b','index_run','audit'):setattr(a,k,Path(a.contract[k]))
    run(a)
