"""Exercise molecular ownership, conformer retention, chunking, and fail-closed input checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from rdkit import rdBase
import process_group_a as pipeline

PARSER = Path(__file__).parent / 'audit_parser.py'


def text(smiles_atom, x=0):
    return f'[{smiles_atom}]<{x},0,0>'


class PipelineTests(unittest.TestCase):
    def test_real_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # E/Z, isotopes, and distinct molecules must not become interchangeable.
            spec = pipeline.importlib.util.spec_from_file_location('audit_fixture', PARSER)
            parser = pipeline.importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser)
            self.assertNotEqual(parser.graph('F/C=C/F')['key'], parser.graph('F/C=C\\F')['key'])
            self.assertNotEqual(parser.graph('C')['key'], parser.graph('[13CH4]')['key'])
            reserved = parser.graph('F')['key']
            definitions = [
                ('chembl3d', 'train', True, [('test_hit', text('C')), ('val_hit', text('N')),
                    ('nabla_hit', text('O')), ('reserved_hit', text('F')),
                    ('chembl3d_110806', text('Cl')), ('conf_only_2', text('Cl', 1)),
                    ('isotope', text('13CH4')), ('bad', '[C]<nan,0,0>')]),
                ('chembl3d', 'val', True, [('test_hit', text('C')), ('val_keep', text('N'))]),
                ('chembl3d', 'test', True, [('fixed_test', text('C'))]),
                ('nabla_structures', 'test', False, [('nabla_fixed', text('O'))]),
                ('nabla_scaffolds', 'test', False, [('nabla_same_key', text('O', 1))]),
            ]
            # A conformation-test molecule (Cl) is intentionally not a reference input.
            files, tasks = [], []
            for i, (dataset, role, group_a, rows) in enumerate(definitions):
                path = root / f'{i}.csv'
                with path.open('w') as out:
                    writer = csv.writer(out, lineterminator='\n')
                    writer.writerow(['name', 'enriched_text'])
                    writer.writerows(rows)
                stat = path.stat()
                files.append({'dataset': dataset, 'role': role, 'group_a': group_a,
                              'label': f'{dataset}/{role}', 'path': str(path),
                              'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                              'rows': len(rows), 'sha256': pipeline.sha(path)})
                # Deliberately cut through CSV rows to test byte-boundary ownership.
                for start in range(0, stat.st_size, 37):
                    tasks.append({'id': len(tasks), 'file_index': i, 'start': start,
                                  'end': min(start + 37, stat.st_size)})
            manifest = {'files': files, 'tasks': tasks, 'rdkit_version': rdBase.rdkitVersion,
                        'parser_path': str(PARSER), 'parser_sha256': pipeline.sha(PARSER),
                        'vocab': [], 'reserved_validation_keys': [reserved],
                        'quarantined_group_a_records': [], 'policy': 'fixture'}
            # A name reused in train is not the quarantined validation record.
            manifest['quarantined_group_a_records'] = [{
                'source': files[1]['path'], 'name': 'chembl3d_110806',
                'row_sha256': pipeline.hashlib.sha256(b'quarantined original validation row').hexdigest()}]
            run = root / 'run'
            pipeline.run_pipeline(manifest, run, 2)
            report = json.loads((run / 'report.json').read_text())
            self.assertFalse(report['final_global_split'])
            def names(role):
                with (run / 'candidate' / role / 'chembl3d.csv').open() as f:
                    return [r['name'] for r in csv.DictReader(f)]
            self.assertEqual(names('train'), ['chembl3d_110806', 'conf_only_2', 'isotope'])
            self.assertEqual(names('val'), ['val_keep'])
            self.assertEqual(names('test'), ['fixed_test'])
            counts = next(x for x in report['files'] if x['role'] == 'train')['counts']
            self.assertEqual(counts['quarantine_identity_failure'], 1)
            self.assertEqual(counts['excluded_test_identity'], 2)
            self.assertEqual(counts['excluded_validation_identity'], 2)
            self.assertEqual(len(json.loads((run / 'reference_summary.json').read_text())['test_domain_overlaps']), 1)
            for f in files:
                self.assertEqual(pipeline.sha(f['path']), f['sha256'])
            # The exact original in its source must still trigger the guard.
            with open(files[0]['path'], 'rb') as f:
                next(f)
                first_raw = next(f)
            pipeline.init_writer(run / 'test_keys.json', run / 'validation_keys.json', [{
                'source': files[0]['path'], 'name': 'test_hit',
                'row_sha256': pipeline.hashlib.sha256(first_raw).hexdigest()}])
            retry = root / 'guard_retry'
            (retry / 'index').mkdir(parents=True)
            (retry / 'index' / '0000').symlink_to(run / 'index' / '0000', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'quarantined record reappeared'):
                pipeline.write_chunk((manifest, tasks[0], retry))
            files[0]['sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                pipeline.verify_input(files[0])


if __name__ == '__main__':
    unittest.main()
