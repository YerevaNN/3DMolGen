"""Iterative whole-InChIKey Group B splitting with disk-backed ownership."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'group_a_split'))
import process_group_a as shared


def database(path, readonly=False):
    if readonly:
        db = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
        db.execute('PRAGMA query_only=ON')
    else:
        db = sqlite3.connect(path)
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=NORMAL')
    # Keep large controller indexes cached; workers only need a small read cache.
    db.execute('PRAGMA cache_size=' + ('-65536' if readonly else '-8388608'))
    db.execute('PRAGMA temp_store=FILE')
    return db


def counts_for(run, task, receipt=None):
    folder = Path(run) / 'index' / f"{task['id']:04d}"
    if receipt:
        if shared.sha(folder / 'report.json') != receipt['report_sha256'] or shared.sha(folder / 'counts.json') != receipt['counts_sha256']:
            raise ValueError('Indexed evidence changed')
    report = json.loads((folder / 'report.json').read_text())
    counts = json.loads((folder / 'counts.json').read_text())
    if report['task'] != task or sum(counts.values()) + report['failures'] != report['rows']:
        raise ValueError('Index count accounting mismatch')
    return counts, report


def insert_registry(db, keys, role):
    db.executemany('INSERT INTO registry VALUES (?,?) ON CONFLICT(key) DO UPDATE SET role=max(role,excluded.role)',
                   ((key, role) for key in keys))
    db.commit()


def choose(db, target, dataset, seed):
    """Prefer matching holdouts, then fill only from unassigned molecular keys."""
    db.create_function('sample_rank', 2, lambda key, split: hashlib.sha256(
        f'{seed}\0{dataset}\0{split}\0{key}'.encode()).digest(), deterministic=True)
    db.execute("UPDATE counts SET action=CASE owner WHEN 3 THEN 'excluded_test_quota' WHEN 2 THEN 'excluded_val_quota' ELSE 'train' END")
    selection = {}
    for split, role in [('test', 3), ('val', 2)]:
        fixed_rows = db.execute('SELECT coalesce(sum(n),0) FROM counts WHERE owner=?', (role,)).fetchone()[0]
        if fixed_rows <= target:
            db.execute('UPDATE counts SET action=? WHERE owner=?', (split, role))
            selected_rows = fixed_rows
        else:
            selected_rows = select_ranked(db, role, None, split, target)
        filled = 0
        if selected_rows < target:
            filled = select_ranked(db, 0, 'train', split, target - selected_rows)
        selection[split] = {'target_rows': target, 'available_matching_holdout_rows': fixed_rows,
                            'matching_holdout_rows_selected': selected_rows, 'new_holdout_rows': filled,
                            'selected_rows': selected_rows + filled,
                            'target_shortfall': max(0, target - selected_rows - filled)}
    db.commit()
    return selection


def select_ranked(db, owner, previous_action, split, needed):
    if needed <= 0:
        return 0
    sql = 'SELECT key,n FROM counts WHERE owner=?'
    params = [owner]
    if previous_action is not None:
        sql += ' AND action=?'
        params.append(previous_action)
    # Each group has at least one row, so this LIMIT contains enough groups if
    # enough eligible rows exist. Hash ranking makes selection reproducible.
    sql += ' ORDER BY sample_rank(key,?),key LIMIT ?'
    params.extend([split, needed])
    rows = db.execute(sql, params).fetchall()
    picked, total = [], 0
    for key, n in rows:
        if total >= needed:
            break
        picked.append((split, key))
        total += n
    db.executemany('UPDATE counts SET action=? WHERE key=?', picked)
    return total


def allocate(count_path, registry_path, run, tasks, receipts, item, seed, fraction):
    db = database(count_path)
    db.execute('CREATE TABLE counts(key TEXT PRIMARY KEY,n INTEGER NOT NULL,owner INTEGER NOT NULL DEFAULT 0,action TEXT) WITHOUT ROWID')
    for t in tasks:
        counts, _ = counts_for(run, t, receipts[t['id']])
        db.executemany('INSERT INTO counts(key,n) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET n=n+excluded.n', counts.items())
        db.commit()
    db.execute('ATTACH DATABASE ? AS ownership', (str(registry_path),))
    db.execute('UPDATE counts SET owner=coalesce((SELECT role FROM ownership.registry WHERE registry.key=counts.key),0)')
    db.commit()
    db.execute('DETACH DATABASE ownership')
    db.execute('CREATE INDEX candidate_owner ON counts(owner)')
    target = max(1, round(item['rows'] * fraction))
    selection = choose(db, target, item['dataset'], seed)
    tally = dict(db.execute('SELECT action,sum(n) FROM counts GROUP BY action'))
    if db.execute("SELECT count(*) FROM counts WHERE (action='train' AND owner>1) OR (action='val' AND owner=3) OR (action IN ('test','val') AND owner=1)").fetchone()[0]:
        raise ValueError('Allocation violates existing ownership')
    db.commit()
    db.close()
    registry = database(registry_path)
    registry.execute('ATTACH DATABASE ? AS current', (str(count_path),))
    registry.execute("INSERT INTO registry SELECT key,CASE action WHEN 'test' THEN 3 WHEN 'val' THEN 2 ELSE 1 END FROM current.counts WHERE action IN ('test','val','train') ON CONFLICT(key) DO UPDATE SET role=max(role,excluded.role)")
    registry.commit()
    registry.close()
    return {'dataset': item['dataset'], 'selection': selection, 'valid_rows_by_action': tally}


def write_chunk(args):
    manifest, task, index_run, dest, counts_path, receipt = args
    item = manifest['files'][task['file_index']]
    shared.check_stat(item)
    counts, report = counts_for(index_run, task, receipt)
    index = Path(index_run) / 'index' / f"{task['id']:04d}"
    if shared.sha(index / 'keys.txt') != report['keys_sha256']:
        raise ValueError('Key sidecar changed')
    db = database(counts_path, readonly=True)
    routes = {}
    keys = list(counts)
    for start in range(0, len(keys), 500):
        batch = keys[start:start + 500]
        routes.update(db.execute('SELECT key,action FROM counts WHERE key IN (' + ','.join('?' for _ in batch) + ')', batch))
    db.close()
    if len(routes) != len(keys):
        raise ValueError('Unassigned molecular key')
    dest = Path(dest) / 'parts' / f"{task['id']:04d}"
    dest.mkdir(parents=True, exist_ok=False)
    outputs = {role: (dest / f'{role}.csv').open('wb') for role in ('train','val','test','excluded')}
    reasons = (dest / 'excluded_reasons.tsv').open('w')
    digest = hashlib.sha256()
    tally = Counter()
    quarantined = {(r['name'], r['row_sha256']) for r in manifest['quarantine']['rows'] if r['source'] == item['path']}
    names = {name for name, _ in quarantined}
    try:
        with (index / 'keys.txt').open() as keyfile:
            for entry, line in itertools.zip_longest(shared.chunk_rows(item, task), keyfile):
                if entry is None or line is None:
                    raise ValueError('Row/key alignment failure')
                offset, raw = entry
                digest.update(raw)
                key = line.rstrip('\n')
                action = routes[key] if key else 'quarantine_identity_failure'
                if names:
                    name = raw.split(b',',1)[0].strip(b'"').decode()
                    if name in names and (name, hashlib.sha256(raw).hexdigest()) in quarantined:
                        raise ValueError('Original quarantined record reappeared')
                tally[action] += 1
                if action in ('train','val','test'):
                    outputs[action].write(raw)
                else:
                    outputs['excluded'].write(raw)
                    reasons.write(f'{offset}\t{key}\t{action}\n')
    finally:
        for stream in outputs.values():
            stream.close()
        reasons.close()
    if digest.hexdigest() != report['source_record_sha256'] or sum(tally.values()) != report['rows']:
        raise ValueError('Writer source/index mismatch')
    shared.check_stat(item)
    result = {'task': task, 'counts': dict(tally),
              'sha256': {role: shared.sha(dest / f'{role}.csv') for role in outputs}}
    shared.atomic(dest / 'report.json', result)
    return result


def assemble(args):
    dataset, role, tasks, dest = args
    dest = Path(dest)
    path = dest / 'candidate' / role / f'{dataset}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    rows = 0
    with path.open('xb') as out:
        out.write(shared.HEADER)
        digest.update(shared.HEADER)
        for t in tasks:
            folder = dest / 'parts' / f"{t['id']:04d}"
            report = json.loads((folder / 'report.json').read_text())
            rows += report['counts'].get(role,0)
            h = hashlib.sha256()
            with (folder / f'{role}.csv').open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):
                    h.update(block)
                    digest.update(block)
                    out.write(block)
            if h.hexdigest() != report['sha256'][role]:
                raise ValueError('Output part changed')
        out.flush()
        os.fsync(out.fileno())
    if shared.sha(path) != digest.hexdigest():
        raise ValueError('Output readback hash mismatch')
    with path.open('rb') as f:
        if sum(1 for _ in f) - 1 != rows:
            raise ValueError('Output row count mismatch')
    return {'dataset': dataset, 'role': role, 'rows': rows, 'path': str(path), 'sha256': digest.hexdigest()}


def seed_registry(manifest, index_run, receipts, registry_path, state):
    group_a = Path(manifest['group_a_run'])
    for name, field in [('report.json','group_a_report_sha256'),('manifest.json','group_a_manifest_sha256'),
                        ('test_keys.json','group_a_test_keys_sha256'),('validation_keys.json','group_a_validation_keys_sha256')]:
        if shared.sha(group_a / name) != manifest[field]:
            raise ValueError('Group A evidence changed: ' + name)
    am = json.loads((group_a / 'manifest.json').read_text())
    db = database(registry_path)
    db.execute('CREATE TABLE registry(key TEXT PRIMARY KEY,role INTEGER NOT NULL) WITHOUT ROWID')
    insert_registry(db, json.loads((group_a / 'test_keys.json').read_text()), 3)
    insert_registry(db, json.loads((group_a / 'validation_keys.json').read_text()), 2)
    for task in am['tasks']:
        f = am['files'][task['file_index']]
        if not f['group_a'] or f['role'] != 'train':
            continue
        counts, r = counts_for(group_a, task)
        folder = group_a / 'index' / f"{task['id']:04d}"
        # Verify the previous index counts against its pinned row-aligned keys.
        if shared.sha(folder / 'keys.txt') != r['keys_sha256']:
            raise ValueError('Group A key sidecar changed')
        with (folder / 'keys.txt').open() as stream:
            observed = Counter(line.rstrip('\n') for line in stream)
        observed.pop('',None)
        if dict(observed) != counts:
            raise ValueError('Group A key counts disagree with sidecar')
        insert_registry(db, counts, 1)
    for task in manifest['tasks']:
        f = manifest['files'][task['file_index']]
        if f['kind'] == 'nabla_training_reference':
            counts, _ = counts_for(index_run, task, receipts[task['id']])
            insert_registry(db, counts, 1)
    totals = dict(db.execute('SELECT role,count(*) FROM registry GROUP BY role'))
    db.close()
    shared.atomic(state, {'phase': 'reference_registry_ready', 'unique_keys_by_role': totals})
    return totals


def run(index_run, output, workers):
    index_run, output = Path(index_run), Path(output)
    report = json.loads((index_run / 'report.json').read_text())
    if report['status'] != 'complete_verified_index' or shared.sha(index_run / 'manifest.json') != report['manifest_sha256']:
        raise ValueError('Index incomplete or changed')
    if shared.sha(index_run / 'receipts.json') != report['receipts_sha256']:
        raise ValueError('Index receipts changed')
    receipts = {r['task_id']: r for r in json.loads((index_run / 'receipts.json').read_text())}
    manifest = json.loads((index_run / 'manifest.json').read_text())
    for name, digest in manifest['code_hashes'].items():
        if shared.sha(name) != digest:
            raise ValueError('Indexed parser/code changed: ' + name)
    for f in manifest['files']:
        shared.check_stat(f)
    output.mkdir(parents=True, exist_ok=False)
    state = output / 'progress.json'
    registry = output / 'ownership.sqlite'
    shared.atomic(output / 'input_manifest.json', manifest)
    started = time.monotonic()
    try:
        shared.atomic(state, {'phase': 'seed_reference_registry'})
        initial = seed_registry(manifest, index_run, receipts, registry, state)
        reports, products, count_dbs = [], [], []
        policy = manifest['policy']['group_b']
        if policy['test_target_fraction_of_input_rows'] != policy['validation_target_fraction_of_input_rows']:
            raise ValueError('This runner requires equal test/validation target fractions')
        for dataset in policy['datasets_in_order']:
            fi, item = next((i,f) for i,f in enumerate(manifest['files']) if f['dataset'] == dataset and f['kind'] == 'group_b_pool')
            tasks = [t for t in manifest['tasks'] if t['file_index'] == fi]
            count_db = output / f'{dataset}.sqlite'
            shared.atomic(state, {'phase': 'allocate', 'dataset': dataset})
            allocation = allocate(count_db, registry, index_run, tasks, receipts, item,
                                  policy['seed'], policy['test_target_fraction_of_input_rows'])
            shared.atomic(output / f'{dataset}.allocation.json', allocation)
            with ProcessPoolExecutor(workers) as pool:
                part_reports = shared.parallel(pool, write_chunk,
                    [(manifest,t,index_run,output,count_db,receipts[t['id']]) for t in tasks],
                    f'write_{dataset}', state)
            actual = Counter()
            for r in part_reports:
                actual.update(r['counts'])
            expected = Counter(allocation['valid_rows_by_action'])
            expected['quarantine_identity_failure'] = next(f['identity_failures'] for f in report['files'] if f['dataset']==dataset)
            if actual != expected or sum(actual.values()) != item['rows']:
                raise ValueError('Dataset routing/accounting mismatch')
            with ProcessPoolExecutor(3) as pool:
                out = shared.parallel(pool, assemble, [(dataset,role,tasks,output) for role in ('train','val','test')],
                                      f'assemble_{dataset}', state)
            allocation.update(input_rows=item['rows'], counts=dict(actual), outputs=out)
            shared.atomic(output / f'{dataset}.report.json', allocation)
            reports.append(allocation)
            products.extend(out)
            count_dbs.append(count_db)
        # Recheck every assignment against final global ownership, including
        # holdouts introduced by later datasets.
        final_checks = []
        for count_db in count_dbs:
            db = database(count_db, readonly=True)
            db.execute('ATTACH DATABASE ? AS global_owner',(str(registry),))
            violations = db.execute("SELECT count(*),coalesce(sum(c.n),0) FROM counts c JOIN global_owner.registry r USING(key) WHERE (c.action='train' AND r.role>1) OR (c.action='val' AND r.role>2)").fetchone()
            db.close()
            if violations != (0,0):
                raise ValueError('Final cross-dataset overlap')
            final_checks.append({'dataset': count_db.stem, 'overlapping_keys': 0, 'overlapping_rows': 0})
        for f in manifest['files']:
            shared.check_stat(f)
        final = {'status':'group_b_iterative_split_complete','final_global_split':False,
                 'pending':'Construct Group C validation and apply its keys globally; final corpus verification and exact-sample deduplication remain separate.',
                 'initial_registry':initial,'datasets':reports,'files':products,'final_overlap_checks':final_checks,
                 'seconds':time.monotonic()-started,'manifest_sha256':shared.sha(index_run/'manifest.json'),
                 'runner_sha256':shared.sha(__file__),'policy':manifest['policy'],'index_run':str(index_run)}
        shared.atomic(output/'report.json',final)
        shared.atomic(state,{'phase':'complete','status':final['status'],'seconds':final['seconds']})
    except Exception as exc:
        shared.atomic(state, {'phase':'failed','error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--runner-sha256',required=True)
    a=p.parse_args()
    if shared.sha(__file__) != a.runner_sha256:
        raise ValueError('Runner changed after submission')
    run(a.index_run,a.output,a.workers)
