"""Reparse actual A/B holdouts and verify training membership without the ownership registry."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import inchi

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'group_a_split'))
from process_group_a import atomic, sha, chunk_rows

COORD=re.compile(r'<[^>]+>')
BRACKET=re.compile(r'\[([^\]]+)\]')
ORGANIC=set('B C N O P S F Cl Br I b c n o p s'.split())


@lru_cache(maxsize=20000)
def direct_key(smiles):
    p=Chem.SmilesParserParams();p.removeHs=False
    mol=Chem.MolFromSmiles(smiles,p)
    if mol is None:raise ValueError('Written holdout cannot be parsed')
    for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
    key=inchi.MolToInchiKey(mol)
    if not re.fullmatch(r'[A-Z]{14}-[A-Z]{8}SA-[A-Z]',key):raise ValueError('Written holdout key unavailable')
    return key


def scan(args):
    item,task,output=args
    RDLogger.DisableLog('rdApp.*')
    counts=Counter();n=0
    for offset,raw in chunk_rows(item,task):
        row=next(csv.reader([raw.decode()],strict=True))
        if len(row)!=2:raise ValueError('Malformed CSV')
        smiles=BRACKET.sub(lambda m:m[1] if m[1] in ORGANIC else m[0],COORD.sub('',row[1]))
        counts[direct_key(smiles)]+=1;n+=1
    dest=Path(output)/f"{task['id']:04d}.json"
    atomic(dest,{'file_index':task['file_index'],'rows':n,'counts':counts})
    return {'task':task['id'],'rows':n}


def verify_hash(item):
    if sha(item['path'])!=item['sha256']:raise ValueError('Written file hash mismatch: '+item['path'])
    return item['path']


def check_b(args):
    dataset,root,refs_path=args
    db=sqlite3.connect(f'file:{root}/{dataset}.sqlite?mode=ro',uri=True)
    db.execute('PRAGMA cache_size=-1048576')
    db.execute('ATTACH DATABASE ? AS refs',(str(refs_path),))
    # Iterate the smaller reference set, using primary-key lookups in counts.
    n,rows=db.execute("SELECT count(*),coalesce(sum(c.n),0) FROM refs.held r CROSS JOIN counts c ON c.key=r.key WHERE c.action='train'").fetchone()
    db.close()
    return {'dataset':dataset,'overlap_keys':n,'overlap_rows':rows}


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);chunks=a.output/'holdout_chunks';chunks.mkdir()
    start=time.monotonic();progress=a.output/'progress.json'
    ar=json.loads((a.group_a/'report.json').read_text());br=json.loads((a.group_b/'report.json').read_text())
    am=json.loads((a.group_a/'manifest.json').read_text())
    files=[];all_outputs=[]
    for group,r in [('A',ar),('B',br)]:
        for x in r['files']:
            item={**x,'label':f"{group}/{x['role']}/{x['dataset']}",
                  'rows':x.get('output_rows',x.get('rows'))}
            all_outputs.append(item)
            if x['role'] in ('test','val'):files.append(item)
    for x in am['files']:
        if not x['group_a'] and x['role']=='test':files.append(x)
    tasks=[]
    for i,f in enumerate(files):
        for offset in range(0,Path(f['path']).stat().st_size,64*1024*1024):
            tasks.append({'id':len(tasks),'file_index':i,'start':offset,
                          'end':min(offset+64*1024*1024,Path(f['path']).stat().st_size)})
    with ProcessPoolExecutor(a.workers) as pool:
        futures=[pool.submit(scan,(files[t['file_index']],t,chunks)) for t in tasks]
        completed=0;rows=0
        for f in as_completed(futures):
            r=f.result();completed+=1;rows+=r['rows']
            atomic(progress,{'phase':'reparse_actual_holdouts','completed_chunks':completed,'total_chunks':len(tasks),'rows':rows})
    counts=[Counter() for _ in files]
    for t in tasks:
        r=json.loads((chunks/f"{t['id']:04d}.json").read_text());counts[r['file_index']].update(r['counts'])
    tests=set();vals=set();val_reports=[]
    for f,c in zip(files,counts):
        if sum(c.values())!=f['rows']:raise ValueError('Written holdout row count mismatch')
        (tests if f['role']=='test' else vals).update(c)
    for f,c in zip(files,counts):
        if f['role']=='val':
            keys=set(c)&tests
            val_reports.append({'label':f['label'],'overlap_keys':len(keys),'overlap_rows':sum(c[k] for k in keys)})
    refs_path=a.output/'actual_holdouts.sqlite';db=sqlite3.connect(refs_path)
    db.execute('CREATE TABLE held(key TEXT PRIMARY KEY) WITHOUT ROWID')
    db.executemany('INSERT INTO held VALUES (?)',((k,) for k in sorted(tests|vals)));db.commit();db.close()
    atomic(progress,{'phase':'check_indexed_training_against_actual_holdouts'})
    old_test=set(json.loads((a.group_a/'test_keys.json').read_text()))
    old_val=set(json.loads((a.group_a/'validation_keys.json').read_text()))
    a_overlaps=Counter();held=tests|vals
    for t in am['tasks']:
        f=am['files'][t['file_index']]
        if not f['group_a'] or f['role']!='train':continue
        c=json.loads((a.group_a/'index'/f"{t['id']:04d}"/'counts.json').read_text())
        for k,n in c.items():
            if k not in old_test and k not in old_val and k in held:a_overlaps[f['dataset']]+=n
    with ProcessPoolExecutor(4) as pool:
        b_checks=list(pool.map(check_b,[(d['dataset'],a.group_b,refs_path) for d in br['datasets']]))
    # Explicitly trace the original test/validation conflicts into PubChem.
    db=sqlite3.connect(f'file:{a.group_b}/pubchem3d.sqlite?mode=ro',uri=True)
    db.execute('CREATE TEMP TABLE conflicts(key TEXT PRIMARY KEY) WITHOUT ROWID')
    db.executemany('INSERT INTO conflicts VALUES (?)',((k,) for k in old_test&old_val))
    conflict_rows=db.execute('SELECT c.action,count(*),sum(c.n) FROM conflicts x CROSS JOIN counts c ON c.key=x.key GROUP BY c.action').fetchall();db.close()
    atomic(progress,{'phase':'fresh_hash_verification','files':len(all_outputs)+3})
    with ProcessPoolExecutor(8) as pool:
        verified=list(pool.map(verify_hash,all_outputs+[x for x in files if x not in all_outputs]))
    result={'status':'passed','seconds':time.monotonic()-start,'rdkit':rdBase.rdkitVersion,
            'directly_reparsed_holdout_rows':sum(sum(c.values()) for c in counts),'holdout_files':len(files),
            'actual_test_unique_keys':len(tests),'actual_validation_unique_keys':len(vals),
            'actual_test_validation_shared_keys':len(tests&vals),'validation_checks':val_reports,
            'group_a_training_overlapping_rows':dict(a_overlaps),'group_b_training_checks':b_checks,
            'original_reference_test_validation_shared_keys':len(old_test&old_val),
            'pubchem_original_conflicts_by_destination':[{'destination':r[0],'keys':r[1],'rows':r[2]} for r in conflict_rows],
            'freshly_hash_verified_files':len(verified),
            'scope':'Every written A/B holdout and fixed nabla structure/scaffold plus GEOM-revisited reparsed directly. Training membership checked through saved key assignments, with fresh hashes of every written A/B CSV. No ownership.sqlite used; no full fresh training reparse.'}
    if tests&vals or a_overlaps or any(r['overlap_keys'] for r in b_checks):result['status']='failed_overlap'
    atomic(a.output/'report.json',result);atomic(progress,{'phase':'complete','status':result['status'],'seconds':result['seconds']})
    print(json.dumps({k:v for k,v in result.items() if k not in ['validation_checks']},indent=2),flush=True)
    if result['status']!='passed':raise ValueError('Independent overlap verification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group-a',type=Path,required=True);p.add_argument('--group-b',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=32)
    a=p.parse_args()
    try:main(a)
    except Exception as exc:
        if a.output.exists():atomic(a.output/'progress.json',{'phase':'failed','error':str(exc)})
        raise
