"""
Save 5 reference and 5 generated conformers for 5 randomly sampled common SMILES into SDF files.

Reference: data/revisited_smi.pickle  (dict: SMILES -> {'confs': [Mol, ...]})
Generated: generation_results.pickle  (dict: SMILES -> [Mol, ...])

Output files go to data/tmp/conformers_sdf/
  <sanitized_smiles>.sdf          — reference conformers
  <sanitized_smiles>_generated.sdf — generated conformers
"""

import pickle
import random
import re
from pathlib import Path
from rdkit.Chem import SDWriter

N_SMILES = 5
N_CONFS = 5
RANDOM_SEED = 42

REF_PATH = "data/revisited_smi.pickle"
GEN_PATH = (
    "/mnt/weka/vtarasov/outputs/outputs/gen_results/"
    "20260601_143503_step-29600-hf_revisited/generation_results.pickle"
)
OUT_DIR = Path("data/tmp/conformers_sdf")


def sanitize_filename(smiles: str) -> str:
    return re.sub(r"[^\w\-]", "_", smiles)


def write_sdf(mols: list, path: Path, smiles: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = SDWriter(str(path))
    for mol in mols:
        mol.SetProp("_Name", smiles)
        writer.write(mol)
    writer.close()
    print(f"  Wrote {len(mols)} conformers -> {path}")


def main():
    print("Loading reference data...")
    with open(REF_PATH, "rb") as f:
        ref = pickle.load(f)

    print("Loading generated data...")
    with open(GEN_PATH, "rb") as f:
        gen = pickle.load(f)

    common = list(set(ref.keys()) & set(gen.keys()))
    print(f"Common SMILES: {len(common)}, randomly sampling {N_SMILES}")
    random.seed(RANDOM_SEED)
    selected = random.sample(common, N_SMILES)

    for smi in selected:
        print(f"\nSMILES: {smi}")
        safe = sanitize_filename(smi)

        ref_mols = ref[smi]["confs"][:N_CONFS]
        write_sdf(ref_mols, OUT_DIR / f"{safe}.sdf", smi)

        gen_mols = gen[smi][:N_CONFS]
        write_sdf(gen_mols, OUT_DIR / f"{safe}_generated.sdf", smi)

    print("\nDone.")


if __name__ == "__main__":
    main()
