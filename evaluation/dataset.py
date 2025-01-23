import os
import rdkit
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Callable
from copy import deepcopy

from posebusters.modules.identity import check_identity
from posebusters.modules.rmsd import check_rmsd
from evaluation.conformer import get_conformer_statistics

import logging

logger = logging.getLogger(__name__)


class EvalDataset(ABC):
    def __init__(self, process_batch: Callable[[list[Chem.Mol]], list[Chem.Mol]]):
        self.process_batch = process_batch

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def batch_iterate(self, size: int = 1000) -> list[Chem.Mol]:
        pass

    def _flatten_on_conformer_axis(
        self, pred_mols: list[Chem.Mol], gt_mols: list[Chem.Mol]
    ) -> tuple[list[Chem.Mol], list[Chem.Mol], list[int]]:
        """
        Flattens the list of predicted and ground-truth molecules on the conformer axis.
        Args:
        - pred_mols (list[Chem.Mol]): List of predicted molecules, with potentially multiple conformers
        - gt_mols (list[Chem.Mol]): List of ground-truth molecules, with exactly one conformer

        Returns:
        - pred_mols_flat (list[Chem.Mol]): List of predicted molecules, with exactly one conformer
        - gt_mols_flat (list[Chem.Mol]): List of ground-truth molecules, with exactly one conformer
        - original_indices (list[int]): List of indices, mapping the flattened predicted molecules to the original predicted molecules
        """
        pred_mols_flat = []
        gt_mols_flat = []
        original_indices = []

        for (
            i,
            pred_mol,
        ) in enumerate(pred_mols):
            if isinstance(pred_mol, Chem.Mol) and pred_mol.GetNumConformers() > 1:
                for c in pred_mol.GetConformers():
                    _mol = deepcopy(pred_mol)
                    _mol.RemoveAllConformers()
                    _mol.AddConformer(c, assignId=True)
                    pred_mols_flat.append(_mol)
                gt_mols_flat.extend(
                    [deepcopy(gt_mols[i])] * pred_mol.GetNumConformers()
                )
                original_indices.extend([i] * pred_mol.GetNumConformers())
                continue

            pred_mols_flat.append(pred_mol)
            gt_mols_flat.append(gt_mols[i])
            original_indices.append(i)

        return pred_mols_flat, gt_mols_flat, original_indices

    def _construct_identity_df(
        self,
        batch_num,
        pred_mols: list[Chem.Mol],
        gt_mols: list[Chem.Mol],
        original_indices: list[int],
    ) -> pd.DataFrame:
        """
        Constructs a pandas DataFrame, containing the identity metrics for each conformer in the predicted molecules.
        Args:
        - pred_mols (list[Chem.Mol]): List of predicted molecules, with exactly one conformer
        - gt_mols (list[Chem.Mol]): List of ground-truth molecules, with exactly one conformer
        - original_indices (list[int]): List of indices, mapping the flattened predicted molecules to the original predicted molecules

        Returns:
        - eval_df (pd.DataFrame): DataFrame containing the identity metrics for each conformer in the predicted molecules
        """
        eval_data = []
        for i, (pred_mol, gt_mol) in enumerate(zip(pred_mols, gt_mols)):
            row = dict()
            row["success"] = pred_mol is not None and isinstance(pred_mol, Chem.Mol)
            if not row["success"]:
                row["error"] = f"Molecule could not be generated: {pred_mol}"
                continue

            # Canonicalize the SMILES strings and compare them
            pred_smiles = Chem.MolToSmiles(pred_mol)
            gt_smiles = Chem.MolToSmiles(gt_mol)
            pred_canonical = Chem.CanonSmiles(pred_smiles)
            gt_canonical = Chem.CanonSmiles(gt_smiles)
            row["original_index"] = f"{batch_num}_{original_indices[i]}"
            row["smiles_match"] = pred_canonical == gt_canonical
            row["pred_smi"] = pred_smiles
            row["gt_smi"] = gt_smiles
            row["pred_canonical_smi"] = pred_canonical
            row["gt_canonical_smi"] = gt_canonical

            # Calculate the identity metrics relevant to docking
            try:
                results = check_identity(pred_mol, gt_mol)["results"]
                for key, value in results.items():
                    row[f"posebusters_identity_{key}"] = value
            except Exception as e:
                logger.error(
                    f"Error while calculating identity metrics: {e}", exc_info=e
                )

            rmsd_dict = check_rmsd(pred_mol, gt_mol)
            row.update(rmsd_dict["results"])

            eval_data.append(row)

        return pd.DataFrame(eval_data)

    def evaluate(
        self, batch_size: int = 1000, max_batches: int = None, output_prefix: str = ""
    ) -> None:
        results = []
        for i, batch in tqdm(
            enumerate(self.batch_iterate(batch_size)), desc="Processing batches.."
        ):
            if max_batches is not None and i >= max_batches:
                break

            pred_mols = self.process_batch(batch)
            pred_mols, gt_mols, original_indices = self._flatten_on_conformer_axis(
                pred_mols, batch
            )

            eval_df = self._construct_identity_df(
                i, pred_mols, gt_mols, original_indices
            )
            posebusters_df = get_conformer_statistics(pred_mols)["results"]
            eval_df = pd.concat(
                [eval_df.reset_index(drop=True), posebusters_df.reset_index(drop=True)],
                axis=1,
            )

            results.append(eval_df)

        final_results = pd.concat(results)
        final_results.to_csv(f"{output_prefix}final_results.csv")

        return final_results


class SDFListDataset(EvalDataset):
    def __init__(
        self, data_path: str, process_batch: Callable[[list[Chem.Mol]], list[Chem.Mol]]
    ):
        self.data_path = data_path
        super().__init__(process_batch)

    def __len__(self):
        return len(os.listdir(self.data_path))

    def batch_iterate(self, size: int = 1000) -> Generator[list[Chem.Mol]]:
        sdfs = os.listdir(self.data_path)
        batch = []

        for sdf in sdfs:
            suppl = Chem.SDMolSupplier(os.path.join(self.data_path, sdf))
            for mol in suppl:
                if mol is None:
                    continue
                batch.append(mol)
                if len(batch) == size:
                    yield batch
                    batch = []

        if len(batch) > 0:
            yield batch


class SDFileDataset(EvalDataset):
    def __init__(
        self, data_path: str, process_batch: Callable[[list[Chem.Mol]], list[Chem.Mol]]
    ):
        super().__init__(process_batch)
        self.data_path = data_path

    def __len__(self):
        suppl = Chem.SDMolSupplier(self.data_path)
        return len([mol for mol in suppl if mol is not None])

    def batch_iterate(self, size: int = 1000) -> Generator[list[Chem.Mol]]:
        suppl = Chem.SDMolSupplier(self.data_path)
        batch = []

        for mol in suppl:
            if mol is None:
                continue
            batch.append(mol)

            if len(batch) == size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch


if __name__ == "__main__":
    import os
    import sys

    sys.path.append("/auto/home/davit/3DMolGen")

    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    import numpy as np
    import pandas as pd

    from icmpy.ConfGenerator import ConfGenerator
    from evaluation.conformer import calculate_rmsd, get_conformer_statistics

    N_JOBS = 2
    PROGRESS = True

    CONFIG = "/auto/home/davit/3DMolGen/icmpy/default_config.yaml"
    ICM_PATH = os.environ.get("ICM_PATH", "/auto/home/menuab/icm-3.9-4/icm")
    DATA = "/auto/home/davit/3DMolGen/data/pcqm4m-v2-train.sdf"

    class ProcessBatch:
        def __call__(self, mols):
            smis = [Chem.MolToSmiles(mol) for mol in mols]

            generator = ConfGenerator(CONFIG, ICM_PATH)
            results = generator.process_mol_list(smis, N_JOBS, PROGRESS)

            return [result.mol or result.error for result in results]

    class ProcessBatchRDkit:
        def __call__(self, mols):
            # Generate 1 conformer for each molecule and optimize it
            mols = [deepcopy(mol) for mol in mols]
            for mol in mols:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)

                # Duplicate the conformer
                mol.AddConformer(mol.GetConformer(0), assignId=True)
            print(f"Generated {len(mols)} conformers")
            return mols

    from evaluation.dataset import SDFileDataset

    dataset = SDFileDataset(DATA, process_batch=ProcessBatchRDkit())

    res = dataset.evaluate(batch_size=10, max_batches=3)
    res.head()
