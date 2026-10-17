import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / 'data_collection/inchikey_filter.py'
AUDIT = ROOT / 'scripts/audit_inchikey_overlap.py'
spec = importlib.util.spec_from_file_location('inchikey_filter', FILTER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FilterTests(unittest.TestCase):
    def test_identity_stereo_and_tautomer_policy(self):
        key = module.standard_inchikey
        self.assertEqual(key('[C]<0,0,0>[C][O]'), key('[O][C][C]'))
        self.assertEqual(key('O=c1cccc[nH]1'), key('Oc1ccccn1'))
        self.assertNotEqual(key('C[C@H](O)F'), key('C[C@@H](O)F'))
        self.assertNotEqual(key('C[C@H](O)F'), key('CC(O)F'))
        self.assertNotEqual(key('[13CH3]CO'), key('CCO'))
        with self.assertRaises(ValueError):
            key('C1CC')

    def test_filter_scope_quarantine_bytes_resume_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'val').mkdir()
            (root / 'test').mkdir()

            def write(path, rows):
                with path.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['name', 'enriched_text'])
                    writer.writerows(rows)
                return path

            refs = [
                write(root / 'val/nablaDFT.csv', [('v', '[C][C][O]')]),
                write(root / 'test/nablaDFT_conformations.csv', [('c', '[C][C][C]')]),
                write(root / 'test/nablaDFT_scaffolds.csv', [('s', 'O=c1cccc[nH]1')]),
                write(root / 'test/nablaDFT_structures.csv', [('t', 'C[C@H](O)F')]),
            ]
            train = write(root / 'merged_train.csv', [
                ('Alpha_same', '[C]<0,0,0>[C][O]'),
                ('Alpha_order', '[O][C][C]'),
                ('Alpha_tautomer', 'Oc1ccccn1'),
                ('nablaDFT_test', 'C[C@H](O)F'),
                ('nablaDFT_conformation', '[C][C][C]'),
                ('Alpha_enantiomer', 'C[C@@H](O)F'),
                ('Alpha_unassigned', 'CC(O)F'),
                ('Alpha_invalid', 'C1CC'),
                ('Alpha_other', 'CCCCC'),
            ])
            original = train.read_bytes()
            audit = root / 'audit'
            command = [sys.executable, str(AUDIT), '--train', str(train), '--output', str(audit), '--workers', '2', '--batch-size', '2']
            for ref in refs:
                command += ['--reference', str(ref)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            out = root / 'filtered'
            command = [sys.executable, str(FILTER), '--audit', str(audit), '--output', str(out), '--workers', '2', '--chunks', '19', '--failure-policy', 'quarantine']
            for label in ['val/nablaDFT', 'test/nablaDFT_scaffolds', 'test/nablaDFT_structures']:
                command += ['--reference-label', label]
            subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual((report['input_rows'], report['overlap_removed_rows'], report['quarantined_rows'], report['remaining_rows']), (9, 4, 1, 4))
            lines = original.splitlines(keepends=True)
            expected = b''.join(lines[i] for i in [0, 5, 6, 7, 9])
            self.assertEqual((out / 'merged_train.csv').read_bytes(), expected)
            self.assertEqual((out / 'quarantined.csv').read_bytes(), lines[0] + lines[8])
            self.assertEqual(train.read_bytes(), original)
            self.assertEqual(report['sha256'], module.sha256_file(out / 'merged_train.csv'))
            subprocess.run(command + ['--resume'], check=True, capture_output=True, text=True)
            self.assertEqual((out / 'merged_train.csv').read_bytes(), expected)
            # Existing data cannot be overwritten, and changed inputs invalidate resume.
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            with train.open('ab') as handle:
                handle.write(b'Alpha_new,CCCC\r\n')
            self.assertNotEqual(subprocess.run(command + ['--resume'], capture_output=True).returncode, 0)

    def test_chunk_resume_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            module.TRAIN = root / 'train.csv'
            module.TRAIN.write_bytes(b'name,enriched_text\nAlpha_a,CCO\nAlpha_b,CCC\n')
            module.OUTPUT = root / 'output'
            (module.OUTPUT / 'chunks').mkdir(parents=True)
            module.SOURCES = ['Alpha']
            module.PRESERVE = set()
            module.REMOVE = {('Alpha', 'CCO')}
            module.QUARANTINE = set()
            module.matching_source.cache_clear()
            task = (0, 0, module.TRAIN.stat().st_size)
            first = module.filter_chunk(task)
            self.assertEqual(module.filter_chunk(task), first)
            part = module.OUTPUT / 'chunks/0000.csv'
            part.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                module.filter_chunk(task)


if __name__ == '__main__':
    unittest.main()
