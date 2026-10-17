"""Preserve Group A partitions; exclude higher-priority full Standard InChIKeys.

Inputs are immutable. Candidate outputs remain provisional until Group B/C
holdouts are selected and applied globally. No scaffold-wide or stereo-free
matching, conformer collapsing, or chemistry repair is performed.
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
import time

HEADER = b'name,enriched_text\n'
KEY_TEST = set()
KEY_VAL = set()
FORBIDDEN_RECORDS = set()


def atomic(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_stat(item):
    s = Path(item['path']).stat()
    if (s.st_size, s.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
        raise ValueError('Input changed: ' + item['path'])


def verify_input(item):
    check_stat(item)
    if sha(item['path']) != item['sha256']:
        raise ValueError('Input hash mismatch: ' + item['path'])
    check_stat(item)
    return item['label']


def chunk_rows(item, task):
    with open(item['path'], 'rb') as stream:
        header = stream.readline()
        if next(csv.reader([header.decode()])) != ['name', 'enriched_text']:
            raise ValueError('Unexpected input header')
        if task['start']:
            stream.seek(task['start'] - 1)
            stream.readline()
        while stream.tell() < task['end']:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            yield offset, raw


def extract(args):
    manifest, task, run = args
    from rdkit import RDLogger, rdBase
    if rdBase.rdkitVersion != manifest['rdkit_version']:
        raise ValueError('RDKit version changed')
    RDLogger.DisableLog('rdApp.*')
    dependency = Path(manifest['parser_path'])
    if sha(dependency) != manifest['parser_sha256']:
        raise ValueError('Pinned parser changed')
    spec = importlib.util.spec_from_file_location('frozen_audit_parser', dependency)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    item = manifest['files'][task['file_index']]
    check_stat(item)
    dest = Path(run) / 'index' / f"{task['id']:04d}"
    dest.mkdir(parents=True, exist_ok=False)
    vocab = set(manifest['vocab'])
    counts = Counter()
    digest = hashlib.sha256()
    n = failures = 0
    started = time.monotonic()
    with (dest / 'keys.txt').open('w') as keys, (dest / 'failures.jsonl').open('w') as bad:
        for offset, raw in chunk_rows(item, task):
            digest.update(raw)
            n += 1
            name = ''
            try:
                row = next(csv.reader([raw.decode()], strict=True))
                if len(row) != 2:
                    raise ValueError('CSV column count')
                name, text = row
                _, _, detail = parser.inspect(text, vocab)
                key = detail['key']
                if not key or len(key) != 27:
                    raise ValueError('Standard InChIKey unavailable')
            except (ValueError, RuntimeError, UnicodeError, csv.Error) as exc:
                failures += 1
                key = ''
                bad.write(json.dumps({'name': name, 'offset': offset, 'reason': str(exc),
                                      'raw_csv_row': raw.decode(errors='replace')}) + '\n')
            keys.write(key + '\n')
            if key:
                counts[key] += 1
            if n % 10000 == 0:
                atomic(dest / 'progress.json', {'rows': n, 'failures': failures,
                       'seconds': round(time.monotonic() - started, 1)})
    check_stat(item)
    if failures and item['role'] != 'train':
        raise ValueError('Reference identity failure: ' + item['label'])
    atomic(dest / 'counts.json', counts)
    result = {'task': task, 'rows': n, 'failures': failures,
              'source_record_sha256': digest.hexdigest(), 'keys_sha256': sha(dest / 'keys.txt'),
              'seconds': round(time.monotonic() - started, 1)}
    atomic(dest / 'report.json', result)
    return result


def route(key, role, test, val):
    if not key:
        return 'quarantine_identity_failure'
    if role == 'test':
        return 'keep'
    if key in test:
        return 'excluded_test_identity'
    if role == 'train' and key in val:
        return 'excluded_validation_identity'
    return 'keep'


def init_writer(test_path, val_path, forbidden):
    global KEY_TEST, KEY_VAL, FORBIDDEN_RECORDS
    KEY_TEST = set(json.loads(Path(test_path).read_text()))
    KEY_VAL = set(json.loads(Path(val_path).read_text()))
    FORBIDDEN_RECORDS = {(r['source'], r['name'], r['row_sha256']) for r in forbidden}


def write_chunk(args):
    manifest, task, run = args
    item = manifest['files'][task['file_index']]
    check_stat(item)
    index = Path(run) / 'index' / f"{task['id']:04d}"
    report = json.loads((index / 'report.json').read_text())
    if sha(index / 'keys.txt') != report['keys_sha256']:
        raise ValueError('Key sidecar changed')
    dest = Path(run) / 'parts' / f"{task['id']:04d}"
    dest.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    digest = hashlib.sha256()
    retained_keys = set()
    with (index / 'keys.txt').open() as keys, (dest / 'kept.csv').open('wb') as kept, \
         (dest / 'excluded.csv').open('wb') as excluded, (dest / 'excluded_reasons.tsv').open('w') as reasons:
        for entry, line in itertools.zip_longest(chunk_rows(item, task), keys):
            if entry is None or line is None:
                raise ValueError('Row/key alignment failure')
            offset, raw = entry
            digest.update(raw)
            key = line.rstrip('\n')
            action = route(key, item['role'], KEY_TEST, KEY_VAL)
            if item['dataset'] == 'chembl3d':
                name = next(csv.reader([raw.decode()]))[0]
                if (item['path'], name, hashlib.sha256(raw).hexdigest()) in FORBIDDEN_RECORDS:
                    raise ValueError('Previously quarantined record reappeared: ' + name)
            counts[action] += 1
            if action == 'keep':
                kept.write(raw)
                retained_keys.add(key)
            else:
                excluded.write(raw)
                reasons.write(f'{offset}\t{key}\t{action}\n')
    if digest.hexdigest() != report['source_record_sha256'] or sum(counts.values()) != report['rows']:
        raise ValueError('Source/index alignment mismatch')
    check_stat(item)
    # Validate retained identities independently of routing decisions.
    if item['role'] in ('train', 'val') and retained_keys & KEY_TEST:
        raise ValueError('Retained test leakage')
    if item['role'] == 'train' and retained_keys & KEY_VAL:
        raise ValueError('Retained validation leakage')
    result = {'task': task, 'counts': dict(counts), 'retained_unique_keys': len(retained_keys),
              'kept_sha256': sha(dest / 'kept.csv'), 'excluded_sha256': sha(dest / 'excluded.csv')}
    atomic(dest / 'report.json', result)
    return result


def assemble(args):
    manifest, fi, run = args
    item = manifest['files'][fi]
    tasks = [t for t in manifest['tasks'] if t['file_index'] == fi]
    dest = Path(run) / 'candidate' / item['role']
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / (item['dataset'] + '.csv')
    digest = hashlib.sha256()
    counts = Counter()
    with output.open('xb') as out:
        out.write(HEADER)
        digest.update(HEADER)
        for task in tasks:
            part = Path(run) / 'parts' / f"{task['id']:04d}"
            report = json.loads((part / 'report.json').read_text())
            counts.update(report['counts'])
            h = hashlib.sha256()
            with (part / 'kept.csv').open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    h.update(block)
                    digest.update(block)
                    out.write(block)
            if h.hexdigest() != report['kept_sha256']:
                raise ValueError('Output part changed')
        out.flush()
        os.fsync(out.fileno())
    # Fresh output read verifies both materialized bytes and physical record count.
    if sha(output) != digest.hexdigest():
        raise ValueError('Output readback mismatch')
    with output.open('rb') as f:
        rows = sum(1 for _ in f) - 1
    if rows != counts['keep'] or sum(counts.values()) != item['rows']:
        raise ValueError('Output row accounting mismatch')
    return {'dataset': item['dataset'], 'role': item['role'], 'path': str(output),
            'sha256': digest.hexdigest(), 'input_rows': item['rows'], 'counts': dict(counts),
            'output_rows': rows, 'bytes': output.stat().st_size}


def parallel(pool, fn, args, phase, state_path):
    futures = [pool.submit(fn, x) for x in args]
    results = []
    for f in as_completed(futures):
        results.append(f.result())
        state = {'phase': phase, 'completed_tasks': len(results), 'total_tasks': len(futures),
                 'updated_unix': time.time()}
        atomic(state_path, state)
        print(json.dumps(state), flush=True)
    return results


def run_pipeline(manifest, run, workers):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    atomic(run / 'manifest.json', manifest)
    state = run / 'progress.json'
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(workers) as pool:
            parallel(pool, verify_input, manifest['files'], 'verify_input_hashes', state)
            parallel(pool, extract, [(manifest, t, run) for t in manifest['tasks']], 'extract_keys', state)
        for fi, item in enumerate(manifest['files']):
            n = sum(json.loads((run / 'index' / f"{t['id']:04d}" / 'report.json').read_text())['rows']
                    for t in manifest['tasks'] if t['file_index'] == fi)
            if n != item['rows']:
                raise ValueError('Indexed row count mismatch: ' + item['label'])
        test, val = set(), set(manifest['reserved_validation_keys'])
        ref_sets = {}
        for t in manifest['tasks']:
            item = manifest['files'][t['file_index']]
            if item['role'] not in ('test', 'val'):
                continue
            keys = set(json.loads((run / 'index' / f"{t['id']:04d}" / 'counts.json').read_text()))
            (test if item['role'] == 'test' else val).update(keys)
            if item['role'] == 'test':
                ref_sets.setdefault(item['label'], set()).update(keys)
        atomic(run / 'test_keys.json', sorted(test))
        atomic(run / 'validation_keys.json', sorted(val))
        intersections = []
        for a, b in itertools.combinations(sorted(ref_sets), 2):
            n = len(ref_sets[a] & ref_sets[b])
            if n:
                intersections.append({'first': a, 'second': b, 'shared_keys': n})
        atomic(run / 'reference_summary.json', {'test_unique_keys': len(test),
               'validation_unique_keys': len(val), 'test_validation_shared_keys': len(test & val),
               'test_domain_overlaps': intersections,
               'conformation_test_used_as_molecular_blacklist': False,
               'scaffold_wide_exclusion': False})
        group_tasks = [t for t in manifest['tasks'] if manifest['files'][t['file_index']]['group_a']]
        with ProcessPoolExecutor(workers, initializer=init_writer,
                                 initargs=(run / 'test_keys.json', run / 'validation_keys.json',
                                           manifest['quarantined_group_a_records'])) as pool:
            parallel(pool, write_chunk, [(manifest, t, run) for t in group_tasks], 'write_group_a', state)
        with ProcessPoolExecutor(min(workers, 8)) as pool:
            reports = parallel(pool, assemble,
                     [(manifest, i, run) for i, x in enumerate(manifest['files']) if x['group_a']],
                     'assemble_and_verify', state)
        for item in manifest['files']:
            check_stat(item)
        report = {'status': 'group_a_fixed_reference_pass_complete',
                  'final_global_split': False, 'pending': 'Apply newly selected Group B/C holdout keys to Group A training/validation before final release.',
                  'files': sorted(reports, key=lambda x: (x['role'], x['dataset'])),
                  'seconds': round(time.monotonic() - started, 1),
                  'policy': manifest['policy'], 'checks': ['pinned full input hashes',
                  'complete row/key alignment', 'zero retained higher-priority key overlaps',
                  'quarantined ChEMBL IDs absent', 'output hash readback and row accounting']}
        atomic(run / 'report.json', report)
        atomic(state, {'phase': 'complete', 'status': report['status'], 'seconds': report['seconds']})
    except Exception as exc:
        atomic(state, {'phase': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if sha(__file__) != manifest['runner_sha256']:
        raise ValueError('Runner changed since manifest preparation')
    run_pipeline(manifest, args.output, args.workers)


if __name__ == '__main__':
    main()
