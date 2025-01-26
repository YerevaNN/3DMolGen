import os

from icmpy.ConfGenerator import ConfGenerator

ICM_PATH = os.environ.get("ICM_PATH", "/auto/home/menuab/icm-3.9-4/icm")


def test_basic_icmpy_conformer_generation():
    mol_identifiers = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CCCCC",
        # "Cc1ccc([C@H]2[CH]c3cnccc3[N]C2=O)cc1"
    ]  # This can also include paths to sdf files

    config_file = "icmpy/default_config.yaml"

    generator = ConfGenerator(config_file=config_file, icm_path=ICM_PATH)
    results = generator.process_mol_list(mol_identifiers, n_jobs=1, show_progress=True)

    assert len(results) == 2
    assert all([r.success for r in results])
    assert all([r.mol.GetNumConformers() for r in results])
    assert results[0].smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert results[1].smiles == "CCCCC"
