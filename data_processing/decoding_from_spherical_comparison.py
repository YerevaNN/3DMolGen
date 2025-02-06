from rdkit import Chem
from rdkit.Chem import rdMolAlign
import sys
import numpy as np
import copy

writer1 = Chem.SDWriter("mols1.sdf")
writer2 = Chem.SDWriter("mols2.sdf")

exclude_h = False

excluded_mols_array = np.load("/auto/home/filya/3DMolGen/data_processing/excluded_mols.npy")
val_indices_file = "/auto/home/menuab/code/3DMolGen/data/pcqm/pcqm4v2_valid_indice.txt"
val_indices = []
with open(val_indices_file, 'r') as f:
    for line in f:
        index = int(line.strip())
        val_indices.append(index)

def compare_mols_sdf(i, suppl1, suppl2):
    try:
        mol1 = next(suppl1)

        if i in excluded_mols_array:
            print("Excluded", i)
            return True
        if i in val_indices:
            print("Valid", i)
            return True

        if i != 9496:
            return True
        
        mol2 = next(suppl2)
        if not mol1 or not mol2:
            print(f"Error: Could not read molecule from one or both SDF files.")
            return None

        if exclude_h:
            mol1 = Chem.RemoveHs(mol1)
            mol2 = Chem.RemoveHs(mol2)

        if mol1.GetNumAtoms() != mol2.GetNumAtoms():
            writer1.write(mol1)
            writer2.write(mol2)
            print(f"NumAtoms not equal: MOL {i}")
            return False
        
        if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
            print(f"Error: One or both molecules lack 3D coordinates in SDF files.")
            return None

        # Align mol2 to mol1 based on coordinates
        try:
            rdMolAlign.AlignMol(mol2, mol1)
            mol1_copy = copy.deepcopy(mol1) #so that mol1 didnt accidentally get changed even slightly
            mol2_copy = copy.deepcopy(mol2)
            rmsd = rdMolAlign.GetBestRMS(mol1_copy, mol2_copy)
            # print(f"RMSD after alignment: {rmsd:.3f}")
        except Exception as err:
            print(f"Error during molecule alignment or rmsd calculation: {err}")
            return None
        if rmsd > 0.1:
            writer1.write(mol1)
            writer2.write(mol2)
            print(f"RMSD after alignment: {rmsd:.3f}")

        return rmsd <= 0.1
    except StopIteration:
        print(f"Error: SDF file is empty: MOL {i}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
        

out_file = "output_comp_with_H.txt"
if exclude_h:
    out_file = "output_comp.txt"
sys.stdout = open(out_file, "w") 

sdf1_path = "/auto/home/menuab/pcqm4m-v2-train.sdf"
sdf2_path = "/auto/home/filya/3DMolGen/mol_9496-rec.sdf"
# sdf2_path = "/auto/home/filya/3DMolGen/reconstructed_mols/reconstructed_molecules_1_with_H.sdf"
# if exclude_h:
#     sdf2_path = "/auto/home/filya/3DMolGen/reconstructed_mols/reconstructed_molecules_0.sdf"
try:
    suppl1 = Chem.SDMolSupplier(sdf1_path, removeHs=exclude_h, sanitize=exclude_h)
    suppl2 = Chem.SDMolSupplier(sdf2_path, removeHs=exclude_h, sanitize=exclude_h)
    if not suppl1 or not suppl2:
        print(f"Error: Could not open one or both SDF files: {sdf1_path}, {sdf2_path}")
except FileNotFoundError:
    print(f"Error: SDF file not found")

for i in range(9600): #3378606):
    if i % 100 == 0:
        print(i)
        sys.stdout.flush()

    are_same = compare_mols_sdf(i, suppl1, suppl2)
    if are_same is None:
        print("Comparison failed. Check error messages above.")
    elif are_same:
        pass
        # print("Molecules are the same (ignoring hydrogens).")
    else:
        print(i)
        print("Molecules are different.")

writer1.close()
writer2.close()
