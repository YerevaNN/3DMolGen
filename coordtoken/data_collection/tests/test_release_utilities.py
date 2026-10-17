"""Exercise reference provenance and independent file verification."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ReleaseUtilitiesTests(unittest.TestCase):
    def test_registry_preserves_inputs_and_reports_metadata_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'val').mkdir(parents=True)
            (source / 'test').mkdir()
            rows = b'name,enriched_text\na,CCO\n'
            references = [source / 'val/nablaDFT.csv',
                          source / 'test/nablaDFT_conformations.csv',
                          source / 'test/nablaDFT_structures.csv']
            for path in references:
                path.write_bytes(rows)
            manifests = root / 'manifests'
            manifests.mkdir()
            (manifests / 'graph_index.jsonl').write_text(json.dumps({
                'split': 'valid', 'molecule_id': 'v1', 'graph': '[C][C][O]'
            }) + '\n')
            for name in ('hashes.json', 'overlap_report.json'):
                (manifests / name).write_text('{}\n')
            with (manifests / 'molecules.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['split', 'source_molecule_id',
                                 'standardized_isomeric_smiles', 'inchikey', 'inchi'])
                writer.writerow(['benchmark_test', 'b1', 'CCO',
                                 'different-metadata-key', 'metadata-inchi'])
            output = root / 'registry'
            command = [sys.executable, str(ROOT / 'data_collection/build_holdout_registry.py'),
                       '--source-root', str(source), '--geom-manifests', str(manifests),
                       '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            registry = json.loads((output / 'registry.json').read_text())
            self.assertEqual(len(registry['references']), 5)
            exclusions = {r['label']: r['molecule_exclusion'] for r in registry['references']}
            self.assertFalse(exclusions['test/nablaDFT_conformations'])
            self.assertTrue(exclusions['test/nablaDFT_structures'])
            self.assertTrue(exclusions['geom_downstream/benchmark_test'])
            self.assertEqual(len(registry['metadata_key_disagreements']), 1)
            for reference in registry['references']:
                self.assertEqual(reference['sha256'], hashlib.sha256(
                    Path(reference['path']).read_bytes()).hexdigest())
            for path in references:
                self.assertEqual(path.read_bytes(), rows)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_independent_verifier_counts_unterminated_row_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            data = b'name,enriched_text\nAlpha_1,CCC'
            (output / 'merged_train.csv').write_bytes(data)
            (output / 'quarantined.csv').write_bytes(b'name,enriched_text\nAlpha_2,C1CC\n')
            report = {'status': 'complete', 'remaining_rows': 1,
                      'output_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                      'input_rows': 3, 'overlap_removed_rows': 1, 'quarantined_rows': 1}
            (output / 'report.json').write_text(json.dumps(report))
            command = [sys.executable, str(ROOT / 'data_collection/verify_filtered_csv.py'),
                       '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            verified = json.loads((output / 'independent_verification.json').read_text())
            self.assertEqual(verified['output']['rows'], 1)
            self.assertEqual(verified['quarantine']['rows'], 1)
            (output / 'merged_train.csv').write_bytes(data.replace(b'CCC', b'CCO'))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
