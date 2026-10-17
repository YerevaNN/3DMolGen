"""Index pinned Group B pools and the existing nabla training pool in parallel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'group_a_split'))
import process_group_a as shared


def run(manifest, output, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shared.atomic(output / 'manifest.json', manifest)
    state = output / 'progress.json'
    started = time.monotonic()
    try:
        for name, digest in manifest['code_hashes'].items():
            if shared.sha(name) != digest:
                raise ValueError('Pinned code changed: ' + name)
        with ProcessPoolExecutor(workers) as pool:
            shared.parallel(pool, shared.verify_input, manifest['files'], 'verify_input_hashes', state)
            reports = shared.parallel(pool, shared.extract,
                        [(manifest, t, output) for t in manifest['tasks']], 'extract_keys', state)
        summaries = []
        receipts = []
        for t in manifest['tasks']:
            folder = output / 'index' / f"{t['id']:04d}"
            receipts.append({'task_id': t['id'], 'report_sha256': shared.sha(folder / 'report.json'),
                             'counts_sha256': shared.sha(folder / 'counts.json')})
        for i, item in enumerate(manifest['files']):
            parts = [r for r in reports if r['task']['file_index'] == i]
            rows = sum(r['rows'] for r in parts)
            if rows != item['rows']:
                raise ValueError('Indexed row count mismatch: ' + item['label'])
            shared.check_stat(item)
            summaries.append({'dataset': item['dataset'], 'rows': rows,
                              'identity_failures': sum(r['failures'] for r in parts)})
        shared.atomic(output / 'receipts.json', receipts)
        shared.atomic(output / 'report.json', {'status': 'complete_verified_index',
            'manifest_sha256': shared.sha(output / 'manifest.json'),
            'receipts_sha256': shared.sha(output / 'receipts.json'),
            'files': summaries, 'seconds': time.monotonic() - started})
        shared.atomic(state, {'phase': 'complete', 'files': summaries})
    except Exception as exc:
        shared.atomic(state, {'phase': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=64)
    a = p.parse_args()
    run(json.loads(a.manifest.read_text()), a.output, a.workers)
