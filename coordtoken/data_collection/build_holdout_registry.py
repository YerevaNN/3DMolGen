#!/usr/bin/env python3
"""Freeze local holdout paths and extract downstream GEOM identity references."""
import argparse
import csv
import hashlib
import json
from inchikey_filter import standard_inchikey
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--geom-manifests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    registry = {'status': 'local_references_frozen', 'identity_policy': 'Full Standard InChIKey regenerated from decoded text; no 3D stereo inference', 'metadata_key_disagreements': [], 'training_csv': str((args.source_root/'merged_train.csv').resolve()), 'references': [], 'pending_external_domains': ['PoseBench experimental ligands', 'Platinum Diverse experimental ligands', 'GEOM-XL exact evaluation identities']}
    for split in ('val', 'test'):
        for path in sorted((args.source_root/split).glob('*.csv')):
            exclusion = path.stem != 'nablaDFT_conformations'
            registry['references'].append({**fingerprint(path), 'label': f'{split}/{path.stem}', 'role': 'validation' if split=='val' else 'test', 'molecule_exclusion': exclusion, 'note': '' if exclusion else 'Known-molecule conformation evaluation; do not exclude identities wholesale. Conformer overlap audit remains required.'})
    generated = args.output/'geom_downstream'
    generated.mkdir()
    graph_index = args.geom_manifests/'graph_index.jsonl'
    registry['downstream_manifest_provenance'] = [fingerprint(graph_index), fingerprint(args.geom_manifests/'hashes.json'), fingerprint(args.geom_manifests/'overlap_report.json'), fingerprint(args.geom_manifests/'molecules.csv')]
    outputs = {split: (generated/f'{split}.csv').open('w', newline='') for split in ('valid', 'benchmark_test')}
    writers = {split: csv.writer(handle) for split, handle in outputs.items()}
    counts = {split: 0 for split in outputs}
    for writer in writers.values(): writer.writerow(['name', 'enriched_text'])
    try:
        with graph_index.open() as handle:
            for line in handle:
                row = json.loads(line)
                split = row['split']
                if split in writers:
                    # Preserve the original CoordToken graph string, including stereo.
                    writers[split].writerow([row['molecule_id'], row['graph']])
                    counts[split] += 1
        # Benchmark identities live in molecules.csv, not graph_index.jsonl.
        if counts['benchmark_test'] == 0:
            with (args.geom_manifests/'molecules.csv').open(newline='') as handle:
                for row in csv.DictReader(handle):
                    if row['split'] != 'benchmark_test':
                        continue
                    smiles = row['standardized_isomeric_smiles']
                    decoded_key = standard_inchikey(smiles)
                    if decoded_key != row['inchikey']:
                        registry['metadata_key_disagreements'].append({
                            'molecule_id': row['source_molecule_id'],
                            'smiles': smiles, 'text_key': decoded_key,
                            'metadata_key': row['inchikey'], 'metadata_inchi': row['inchi'],
                            'resolution': 'Protect text-derived identity under the declared policy; retain discrepancy for 3D stereo audit.'})
                    writers['benchmark_test'].writerow([row['source_molecule_id'], smiles])
                    counts['benchmark_test'] += 1
    finally:
        for handle in outputs.values(): handle.close()
    for split in outputs:
        if not counts[split]: raise ValueError(f'Missing downstream reference: {split}')
        path = generated/f'{split}.csv'
        registry['references'].append({**fingerprint(path), 'label': f'geom_downstream/{split}', 'role': 'validation' if split=='valid' else 'test', 'molecule_exclusion': True, 'rows': counts[split], 'note': 'Includes all benchmark identities, including identities excluded from a particular evaluation score.'})
    (args.output/'registry.json').write_text(json.dumps(registry, indent=2)+'\n')
    print(json.dumps({'references': len(registry['references']), 'exclusion_references': sum(x['molecule_exclusion'] for x in registry['references']), 'downstream_counts': counts, 'metadata_key_disagreements': len(registry['metadata_key_disagreements'])}, indent=2))


if __name__ == '__main__': main()
