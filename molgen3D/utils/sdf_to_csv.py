import os
import rdkit
import pandas as pd
from tqdm import tqdm

from rdkit.Chem import AllChem, Draw, SDMolSupplier, MolToSmiles, SDWriter, Mol


class SDFPropEnricher:
    def __init__(self, sdf_file_path: str, output_sdf_path: str = None, N: int = None):
        self.sdf_file_path = sdf_file_path
        self.N = N

        if output_sdf_path is None:
            output_sdf_path = sdf_file_path.replace(".sdf", "_enriched.sdf")

        self.output_sdf_path = output_sdf_path
    
    def enrich(self, drop_conformers: bool = False):
        sdf_supplier = SDMolSupplier(self.sdf_file_path)
        writer = SDWriter(self.output_sdf_path)

        if self.N is not None:
            sdf_supplier = tqdm(sdf_supplier, total=self.N)

        for i, mol in enumerate(sdf_supplier):
            if mol is None:
                continue

            mol = self.enrich_mol(mol, i, drop_conformers)
            writer.write(mol)

            if self.N is not None and i >= self.N:
                break

        writer.close()

    def enrich_mol(self, mol: Mol, i: int, drop_conformers: bool = False) -> Mol:
        if drop_conformers:
            mol.RemoveAllConformers()

        name = mol.GetProp("_Name") or str(i)
        mol.SetIntProp("mol_index", i)
        mol.SetProp("mol_id", name)

        return mol



class SDFToCSVParser:
    def __init__(self, sdf_file_path: str, output_path: str, N: int = None):
        self.sdf_file_path = sdf_file_path
        self.output_path = output_path.strip("/")
        self.N = N

    def parse(self, index_col: str = "_index"):
        sdf_supplier = SDMolSupplier(self.sdf_file_path)
        sdf_name = self.sdf_file_path.split("/")[-1].split(".")[0]
        csv_file = f"{self.output_path}/{sdf_name}_parsed.csv"

        if self.N is not None:
            sdf_supplier = tqdm(sdf_supplier, total=self.N)

        data = []
        i = 0
        for mol in sdf_supplier:
            row = {"_index": i}
            i += 1

            if mol is None:
                data.append(row)
                continue
            
            row["smi"] = MolToSmiles(mol, canonical=True)
            row["mol_id"] = mol.GetProp("_Name")
            row.update(mol.GetPropsAsDict())

            # Keep the SDF with conformers
            output_sdf = f"{self.output_path}/_{sdf_name}_confs/{i}.sdf"
            # Create the output directory
            output_dir = "/".join(output_sdf.split("/")[:-1])
            os.makedirs(output_dir, exist_ok=True)
            writer = SDWriter(output_sdf)
            writer.write(mol)
            writer.close()
            row["conf_path"] = output_sdf
            data.append(row)

            if self.N is not None and i >= self.N:
                break

        df = pd.DataFrame(data)
        df.set_index(index_col, inplace=True)
        df.to_csv(csv_file)
        return df


class CSVToRDKitParser:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def parse(self):
        df = pd.read_csv(self.csv_file_path)
        mols = []
        for i, row in df[["_index", "smi", "mol_id", "conf_path"]].iterrows():
            mol = SDMolSupplier(row["conf_path"])[0]
            assert MolToSmiles(mol, canonical=True) == row["smi"]
            mols.append(mol)
            
        return mols
    

if __name__ == "__main__":
    df1 = SDFToCSVParser("/auto/home/davit/3DMolGen/data/eval/output_censo.sdf", "data/join", N=10000).parse(index_col="pickle_path")
    df2 = SDFToCSVParser("/auto/home/davit/3DMolGen/data/eval/output_censo_ginger.sdf", "data/join", N=10000).parse(index_col="pickle_path")
    # df3 = SDFToCSVParser("data/join/enr_ging_skipped.sdf", "data/join", N=10000).parse()


    print()
    print("Parsed", "molecules")

