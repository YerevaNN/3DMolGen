from posebusters import run_posebusters
from rdkit import Chem
import pickle

results = [run_posebusters(mol) for mol in molecules]
for i, result in enumerate(results):
    print(f"Molecule {i+1} results:")
    for test, outcome in result.items():
        print(f"  {test}: {outcome}")

if __name__ == "__main__":
    
    with open("/auto/home/menuab/code/3DMolGen/gen_results/1e_1x_greedy/generation_resutls.pickle", 'rb') as f:
        model_preds = pickle.load(f)

    mols = [for values in model_preds.values()]