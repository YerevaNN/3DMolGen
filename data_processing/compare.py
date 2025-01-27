from rdkit import Chem
from rdkit.Chem import AllChem

def create_atom_mapping_by_properties(mol1, mol2):
    atoms1 = [(atom.GetAtomicNum(), atom.GetDegree(), atom.GetIdx()) for atom in mol1.GetAtoms()]
    atoms2 = [(atom.GetAtomicNum(), atom.GetDegree(), atom.GetIdx()) for atom in mol2.GetAtoms()]

    # Sort atoms based on (atomic number, degree)
    sorted_atoms1 = sorted(atoms1, key=lambda x: (x[0], x[1]))
    sorted_atoms2 = sorted(atoms2, key=lambda x: (x[0], x[1]))

    atom_map = []
    for i in range(mol1.GetNumAtoms()):
        atom_map.append((sorted_atoms1[i][2], sorted_atoms2[i][2])) # map original indices

    return atom_map


def compare_mols_sdf(sdf_file1, sdf_file2):
    try:
        suppl1 = Chem.SDMolSupplier(sdf_file1)
        suppl2 = Chem.SDMolSupplier(sdf_file2)

        if not suppl1 or not suppl2:
            print(f"Error: Could not open one or both SDF files: {sdf_file1}, {sdf_file2}")
            return None

        mol1 = next(iter(suppl1))
        mol2 = next(iter(suppl2))

        if not mol1 or not mol2:
            print(f"Error: Could not read molecule from one or both SDF files.")
            return None

        mol1 = Chem.RemoveHs(mol1)

        if mol1.GetNumAtoms() != mol2.GetNumAtoms():
            print(f"NumAtoms not equal: {sdf_file1} or {sdf_file2}")
            return False

        smiles1 = Chem.MolToSmiles(mol1, canonical=True) # Canonical SMILES
        smiles2 = Chem.MolToSmiles(mol2, canonical=True) # Canonical SMILES

        if smiles1 != smiles2:
            print(f"Smiles not equal: {sdf_file1} or {sdf_file2}")
            return False

        if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
            print(f"Error: One or both molecules lack 3D coordinates in SDF files.")
            return None

        # Align mol2 to mol1 based on coordinates
        try:
            mp = create_atom_mapping_by_properties(mol1, mol2)
            AllChem.AlignMol(mol2, mol1, atomMap=mp)
        except Exception as align_err:
            print(f"Error during molecule alignment: {align_err}")
            return None


        # Calculate RMSD (Root Mean Square Deviation) after alignment
        rmsd = AllChem.GetBestRMS(mol1, mol2)

        print(f"RMSD after alignment: {rmsd:.3f}")

        return rmsd <= 0.1

    except FileNotFoundError:
        print(f"Error: SDF file not found: {sdf_file1} or {sdf_file2}")
        return None
    except StopIteration:
        print(f"Error: SDF file is empty: {sdf_file1} or {sdf_file2}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Example Usage (replace with your file paths)
sdf1_path = "/auto/home/menuab/pcqm4m-v2-train.sdf"  # Replace with path to SDF file WITH hydrogens
sdf2_path = "/auto/home/filya/3DMolGen/reconstructed_molecules.sdf"   # Replace with path to SDF file WITHOUT hydrogens

are_same = compare_mols_sdf(sdf_file1=sdf1_path, sdf_file2=sdf2_path)

if are_same is None:
    print("Comparison failed. Check error messages above.")
elif are_same:
    print("Molecules are the same (ignoring hydrogens and 3D pose).")
else:
    print("Molecules are different.")