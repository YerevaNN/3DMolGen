"""Prepare a pinned index manifest for Group B and the existing nabla training pool."""
import argparse
import json
from pathlib import Path
import sys
from rdkit import rdBase

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'group_a_split'))
from process_group_a import sha, atomic, check_stat


def prepare(preflight, group_a, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((preflight / 'prepared_input_manifest.json').read_text())
    policy = json.loads((preflight / 'split_policy.json').read_text())
    a_report = json.loads((group_a / 'report.json').read_text())
    if a_report['status'] != 'group_a_fixed_reference_pass_complete':
        raise ValueError('Group A is incomplete')
    a_manifest = json.loads((group_a / 'manifest.json').read_text())
    if rdBase.rdkitVersion != a_manifest['rdkit_version']:
        raise ValueError('RDKit differs from verified Group A')
    order = policy['group_b']['datasets_in_order']
    inputs = {f['relative_path']: f for f in prepared['files']}
    files = []
    for rel in [f'grp_b/{name}.csv' for name in order] + ['grp_c/train/nablaDFT.csv']:
        f = inputs[rel]
        check_stat(f)
        files.append({**f, 'dataset': Path(rel).stem, 'role': 'train', 'label': rel,
                      'kind': 'group_b_pool' if rel.startswith('grp_b/') else 'nabla_training_reference'})
    tasks = []
    for i, f in enumerate(files):
        for start in range(0, f['bytes'], 64 * 1024 * 1024):
            tasks.append({'id': len(tasks), 'file_index': i, 'start': start,
                          'end': min(start + 64 * 1024 * 1024, f['bytes'])})
    parser = HERE.parent / 'group_a_split/audit_parser.py'
    shared = HERE.parent / 'group_a_split/process_group_a.py'
    manifest = {'files': files, 'tasks': tasks, 'rdkit_version': rdBase.rdkitVersion,
                'parser_path': str(parser), 'parser_sha256': sha(parser), 'vocab': a_manifest['vocab'],
                'code_hashes': {str(p): sha(p) for p in (parser, shared, HERE / 'index_inputs.py')},
                'group_a_run': str(group_a), 'group_a_report_sha256': sha(group_a / 'report.json'),
                'group_a_manifest_sha256': sha(group_a / 'manifest.json'),
                'group_a_test_keys_sha256': sha(group_a / 'test_keys.json'),
                'group_a_validation_keys_sha256': sha(group_a / 'validation_keys.json'),
                'policy': policy, 'quarantine': json.loads((preflight / 'quarantine_resolution/current_quarantine.json').read_text()),
                'selection_scope': 'Existing Group A and nabla training keys and earlier Group B training keys block new holdout filling. Future Group B pools have no assigned training membership yet.'}
    atomic(output, manifest)
    print(json.dumps({'files': len(files), 'tasks': len(tasks), 'rows': sum(f['rows'] for f in files),
                      'group_b_rows': sum(f['rows'] for f in files if f['kind'] == 'group_b_pool')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', required=True, type=Path)
    p.add_argument('--group-a', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    prepare(a.preflight.resolve(), a.group_a.resolve(), a.output.resolve())
