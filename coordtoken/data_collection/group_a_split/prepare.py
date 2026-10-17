"""Pin the approved prepared Group A files and fixed molecular references."""
import argparse
import json
from pathlib import Path
from rdkit import rdBase
from process_group_a import atomic, sha, check_stat

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preflight', type=Path, required=True, help='Directory containing prepared_input_manifest.json and verified references')
parser.add_argument('--output', type=Path, required=True, help='New manifest output path')
args = parser.parse_args()
BASE = args.preflight.resolve()
if args.output.exists():
    raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True, exist_ok=True)
prepared = json.loads((BASE / 'prepared_input_manifest.json').read_text())
policy = json.loads((BASE / 'split_policy.json').read_text())
quarantine = json.loads((BASE / 'quarantine_resolution/current_quarantine.json').read_text())
reserved = json.loads((BASE / 'reserved_quarantine_reference_keys.json').read_text())
refs = json.loads((BASE / 'references/manifest.json').read_text())
verified_refs = json.loads((BASE / 'references/final_report.json').read_text())
assert verified_refs['status'] == 'complete_input_hash_and_row_verified_audit'
assert all(f['rdkit'] == rdBase.rdkitVersion for f in verified_refs['files'])
assert policy['nabla_scaffold_wide_exclusion'] is False
files = []
for f in prepared['files']:
    rel = Path(f['relative_path'])
    group_a = rel.parts[0] == 'grp_a'
    if not group_a and str(rel) not in ('grp_c/test/nablaDFT_structures.csv', 'grp_c/test/nablaDFT_scaffolds.csv'):
        continue
    files.append({**f, 'dataset': rel.stem, 'role': rel.parts[1],
                  'group_a': group_a, 'label': str(rel)})
assert sum(f['group_a'] for f in files) == 42
for role in ('train', 'val', 'test'):
    assert sorted(f['dataset'] for f in files if f['group_a'] and f['role'] == role) == sorted(policy['group_a']['datasets'])
geom = next(f for f in refs['files'] if f['dataset'] == 'geom_revisited')
files.append({'path': geom['path'], 'bytes': geom['size'], 'mtime_ns': geom['mtime_ns'],
              'sha256': geom['expected_sha256'], 'rows': geom['expected_data_lines'],
              'dataset': 'geom_revisited', 'label': 'geom_revisited/test', 'role': 'test', 'group_a': False})
for f in files:
    check_stat(f)
tasks = []
for i, f in enumerate(files):
    for start in range(0, f['bytes'], 64 * 1024 * 1024):
        tasks.append({'id': len(tasks), 'file_index': i, 'start': start, 'end': min(start + 64 * 1024 * 1024, f['bytes'])})
manifest = {'files': files, 'tasks': tasks, 'rdkit_version': rdBase.rdkitVersion,
            'parser_path': str(HERE / 'audit_parser.py'), 'parser_sha256': sha(HERE / 'audit_parser.py'),
            'runner_sha256': sha(HERE / 'process_group_a.py'), 'vocab': refs['vocab'],
            'reserved_validation_keys': [r['full_standard_inchikey'] for r in reserved['rows']],
            'quarantined_group_a_records': [r for r in quarantine['rows'] if '/grp_a/' in r['source']],
            'policy': policy, 'source_artifacts': [{'path': str(BASE / p), 'sha256': sha(BASE / p)} for p in
                ('prepared_input_manifest.json', 'split_policy.json', 'quarantine_resolution/current_quarantine.json',
                 'reserved_quarantine_reference_keys.json', 'references/manifest.json', 'references/final_report.json')],
            'scope': 'Group A fixed-reference pass only; Group B/C newly selected holdouts must be applied before final release.',
            'group_a_input_rows': sum(f['rows'] for f in files if f['group_a'])}
atomic(args.output, manifest)
print(json.dumps({'files': len(files), 'chunks': len(tasks), 'group_a_rows': manifest['group_a_input_rows'],
                  'total_rows_including_references': sum(f['rows'] for f in files)}))
