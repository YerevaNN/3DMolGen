#!/usr/bin/env python3
"""Independently verify a published filtered CSV's hash, size and record count."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def inspect(path):
    before = path.stat()
    digest = hashlib.sha256()
    newlines = 0
    last_byte = b''
    with path.open('rb') as handle:
        header = handle.readline()
        if header.rstrip(b'\r\n') != b'name,enriched_text':
            raise ValueError(f'Unexpected CSV header: {path}')
        digest.update(header)
        last_byte = header[-1:]
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
            newlines += block.count(b'\n')
            last_byte = block[-1:]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed during verification: {path}')
    # Filter rejects multiline CSV records, so physical lines count records here.
    if after.st_size > len(header) and last_byte != b'\n':
        newlines += 1
    return {'path': str(path.resolve()), 'rows': newlines, 'bytes': after.st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.output/'report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Filtering is not complete')
    started = time.monotonic()
    result = inspect(args.output/'merged_train.csv')
    if (result['rows'], result['bytes'], result['sha256']) != (report['remaining_rows'], report['output_bytes'], report['sha256']):
        raise ValueError('Independent output verification failed')
    quarantine = inspect(args.output/'quarantined.csv')
    if quarantine['rows'] != report['quarantined_rows']:
        raise ValueError('Quarantine count differs from report')
    if report['input_rows'] != report['remaining_rows'] + report['overlap_removed_rows'] + report['quarantined_rows']:
        raise ValueError('Input/output conservation failed')
    result = {'status': 'verified', 'output': result, 'quarantine': quarantine, 'elapsed_seconds': time.monotonic()-started}
    temporary = args.output/'independent_verification.tmp'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(args.output/'independent_verification.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
