

## Expected input from `posebustes_fixture.pkl`
```markdown
{'CC(=O)O': [Mol1, Mol2, Mol3, Mol4], 'CCO': [Mol5, Mol6, Mol7] }
```
The keys are the canonical SMILES of the molecules (also RDKit Mol Objects), and the values are lists of RDKit Mol Objects.

### Regenerating the fixture

If PoseBusters or RDKit versions change, recreate the fixture by running:

```python
from pathlib import Path
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

def build_conformers(smiles: str, num_confs: int) -> list[Chem.Mol]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    for conf_id in range(mol.GetNumConformers()):
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
        single = Chem.Mol(mol)
        single.RemoveAllConformers()
        single.AddConformer(mol.GetConformer(conf_id), assignId=True)
        yield Chem.RemoveHs(single)

fixture = {
    "CCO": list(build_conformers("CCO", 3)),
    "CC(=O)O": list(build_conformers("CC(=O)O", 4)),
}

fixture_dir = Path("molgen3D/evaluation/tests/fixtures")
fixture_dir.mkdir(parents=True, exist_ok=True)
with (fixture_dir / "posebusters_fixture.pkl").open("wb") as handle:
    pickle.dump(fixture, handle)
```

Keep the total molecule count modest (≤10) so that test runs stay quick.



## Expected Call from `bust_full_gens`:
```python
summary, overall = bust_full_gens(fixture, num_workers=2, config='mol', full_report=False)
```
## Expected Output from `bust_full_gens`:
```python
summary = pd.DataFrame(
    {
        'smiles': ['CC(=O)O', 'CCO'],
        'num_conformers': [4, 3],
        'num_passed': [4.0, 3.0],
        'pass_pct': [1.0, 1.0]
    },
)
overall = 1.0
```

The `summary` DataFrame can be saved directly as a CSV in the evaluation pipeline.

## Test coverage

Run the new regression tests with:

```bash
pytest molgen3D/evaluation/tests/test_posebusters.py
```

The suite exercises:

- `bust_cpu` with both `avg` and `bool` aggregation.
- Worker-count invariance when splitting conformers.
- `bust_full_gens` per-SMILES summaries and overall pass-rate consistency.

Tests are skipped automatically if the `posebusters` dependency is unavailable.