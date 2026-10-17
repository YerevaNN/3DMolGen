#!/usr/bin/env python3
"""Launch the resumable molecular-identity audit from an explicit holdout registry."""
import argparse
import json
import os
from pathlib import Path
import sys
from build_holdout_registry import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    references = [r for r in registry['references'] if r['molecule_exclusion']]
    if not references or len(references) > 62:
        raise ValueError('Audit supports 1 to 62 explicitly selected references')
    for reference in references:
        path = Path(reference['path'])
        if fingerprint(path)['sha256'] != reference['sha256']:
            raise ValueError(f'Frozen reference changed: {path}')
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-u', str(root/'scripts/audit_inchikey_overlap.py'), '--train', registry['training_csv'], '--output', str(args.output), '--workers', str(args.workers), '--benchmark', '--resume']
    for reference in references:
        command += ['--reference', reference['path']]
    os.execv(sys.executable, command)


if __name__ == '__main__': main()
