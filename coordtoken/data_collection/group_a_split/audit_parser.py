"""Read-only parallel-by-byte-range audit of existing grouped training CSVs.

No molecule removal, source writes, conformer generation or test scoring.
Completed chunks are reusable only with the same manifest/code/RDKit version.
"""
import argparse
import ast
from collections import Counter
import csv
import fcntl
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdinchi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORGANIC = set('B C N O P S F Cl Br I b c n o p s'.split())
TOKEN = re.compile(r'(\[[^\]]+\])<([^>]+)>|(%\d{2})|(=|#|:|/|\\|-)|(\()|(\))|(\d)|(\.)')
NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')


def atomic(path, data):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def fingerprint(path):
    s = path.stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


@lru_cache(maxsize=20000)
def graph(smiles):
    params = Chem.SmilesParserParams()
    params.sanitize = True
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('graph_parse_failure')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        inchi, code, message, _, _ = rdinchi.MolToInchi(mol)
        key = rdinchi.InchiToInchiKey(inchi) if code in (0, 1) and inchi.startswith('InChI=1S/') else None
    except Exception as exc:
        key, code, message = None, 2, f'{type(exc).__name__}: {exc}'
    return {'n': mol.GetNumAtoms(), 'bonds': tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()),
            'key': key, 'inchi_warning': message if code == 1 else None,
            'isotope': any(a.GetIsotope() for a in mol.GetAtoms()),
            'radical': any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
            'stereo': any(b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY) for b in mol.GetBonds())}


def inspect(text, vocab):
    smiles, coords, names = [], [], []
    pos = 0
    for match in TOKEN.finditer(text):
        if match.start() != pos:
            raise ValueError('unrecognized_enriched_fragment')
        pos = match.end()
        if match.group(1):
            desc = match.group(1)
            parts = match.group(2).split(',')
            if len(parts) != 3:
                raise ValueError('coordinate_triplet_length')
            try:
                xyz = tuple(float(x.strip()) for x in parts)
            except ValueError:
                raise ValueError('invalid_coordinate_number') from None
            if not all(math.isfinite(x) for x in xyz):
                raise ValueError('nonfinite_coordinates')
            if not all(NUMBER.fullmatch(x.strip()) for x in parts):
                raise ValueError('nonstandard_coordinate_number')
            if max(map(abs, xyz)) > 3.4028234663852886e38:
                raise ValueError('coordinate_float32_overflow')
            coords.append(xyz)
            inner = desc[1:-1]
            smiles.append(inner if inner in ORGANIC else desc)
            names.append(desc)
        else:
            smiles.append(match.group())
            names.append(match.group())
    if pos != len(text) or not coords:
        raise ValueError('unparsed_or_empty_enriched_text')
    g = graph(''.join(smiles))
    if g['n'] != len(coords):
        raise ValueError('atom_coordinate_count_mismatch')
    flags = []
    if len(set(coords)) != len(coords):
        flags.append('coincident_atoms_review')
    distances = [math.dist(coords[i], coords[j]) for i, j in g['bonds']]
    if distances and min(distances) < .7:
        flags.append('bond_below_0.7A_review')
    if distances and max(distances) > 2.5:
        flags.append('bond_above_2.5A_review')
    unknown = sorted(set(names) - vocab)
    if unknown:
        flags.append('outside_current_vocab_review')
    if not g['key']:
        flags.append('standard_inchikey_unavailable_review')
    properties = {'isotope': g['isotope'], 'radical': g['radical'], 'bond_stereo': g['stereo'],
                  'inchi_warning': bool(g['inchi_warning'])}
    details = {'atoms': len(coords), 'min_bond_A': min(distances) if distances else None,
               'max_bond_A': max(distances) if distances else None, 'unknown_tokens': unknown,
               'key': g['key'], 'inchi_warning': g['inchi_warning']}
    return flags, properties, details


def audit(manifest_path, task_id, output):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    task = manifest['tasks'][task_id]
    item = manifest['files'][task['file_index']]
    source = Path(item['path'])
    expected = {'size': item['size'], 'mtime_ns': item['mtime_ns']}
    if fingerprint(source) != expected:
        raise ValueError('Source size/mtime changed before audit')
    out = output / f'task_{task_id:04d}'
    out.mkdir(parents=True, exist_ok=True)
    contract = {'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'rdkit': rdBase.rdkitVersion, 'task': task, 'source': item}
    with (out / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'report.json').exists():
            done = json.loads((out / 'report.json').read_text())
            if done.get('contract') != contract or done.get('status') != 'complete':
                raise ValueError('Existing report contract mismatch')
            print('Verified existing completed chunk', task_id, flush=True)
            return
        started = last = time.monotonic()
        counts = Counter()
        examples = []
        attempt = f'{time.time_ns()}_{os.getpid()}'
        flags_path = out / f'flagged_{attempt}.jsonl.gz'
        digest = hashlib.sha256()
        vocab = set(manifest['vocab'])
        RDLogger.DisableLog('rdApp.*')
        state = {'status': 'running', 'contract': contract, 'counts': {}, 'flagged_records': str(flags_path)}
        atomic(out / 'progress.json', state)
        try:
            with source.open('rb', buffering=4*1024*1024) as f, gzip.open(flags_path, 'wt') as flagged:
                header = f.readline()
                if next(csv.reader([header.decode().strip()])) != ['name', 'enriched_text']:
                    raise ValueError('Unexpected CSV header')
                if task['start']:
                    f.seek(task['start'] - 1)
                    f.readline()
                while True:
                    offset = f.tell()
                    if offset >= task['end']:
                        break
                    raw = f.readline()
                    if not raw:
                        break
                    counts['rows'] += 1
                    digest.update(raw)
                    name = ''
                    details = {}
                    errors = []
                    flags = []
                    try:
                        row = next(csv.reader([raw.decode('utf-8')], strict=True))
                        if len(row) != 2:
                            raise ValueError('csv_column_count')
                        name, text = row
                        flags, properties, details = inspect(text, vocab)
                        counts['format_pass_rows'] += 1
                        for prop, value in properties.items():
                            counts[f'rows_with_{prop}'] += bool(value)
                    except (ValueError, UnicodeError, csv.Error, RuntimeError) as exc:
                        errors = [str(exc)[:300]]
                        counts['format_failure_rows'] += 1
                    counts['review_flag_rows'] += bool(flags)
                    counts['any_flagged_rows'] += bool(errors or flags)
                    for flag in errors + flags:
                        counts[flag] += 1
                    if errors or flags:
                        record = {'byte_offset': offset, 'name': name, 'format_failures': errors,
                                  'review_flags': flags, 'details': details}
                        flagged.write(json.dumps(record) + '\n')
                        if len(examples) < 10:
                            examples.append(record)
                    if time.monotonic() - last >= 30:
                        state.update(counts=dict(counts), next_byte=f.tell(), elapsed_seconds=time.monotonic()-started)
                        atomic(out / 'progress.json', state)
                        print(json.dumps({'task': task_id, 'counts': dict(counts), 'elapsed': state['elapsed_seconds']}), flush=True)
                        last = time.monotonic()
                state['next_byte'] = f.tell()
            if fingerprint(source) != expected:
                raise ValueError('Source changed during audit')
            assert counts['format_pass_rows'] + counts['format_failure_rows'] == counts['rows']
            state.update(status='complete', counts=dict(counts), examples=examples,
                         elapsed_seconds=time.monotonic()-started, scanned_record_bytes_sha256=digest.hexdigest(),
                         physical_validity_proven=False, source_graph_correspondence_proven=False,
                         full_cross_split_overlap_checked=False, publication_ready=False)
            atomic(out / 'report.json', state)
            atomic(out / 'progress.json', state)
            print(json.dumps({'task': task_id, 'status': 'complete', 'counts': dict(counts)}), flush=True)
        except Exception as exc:
            state.update(status='failed', counts=dict(counts), error=f'{type(exc).__name__}: {exc}')
            atomic(out / 'progress.json', state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.task, args.output)
