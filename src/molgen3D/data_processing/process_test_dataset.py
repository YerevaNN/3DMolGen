from rdkit import Chem
import csv
import os
import cloudpickle  # type: ignore
import argparse
from pathlib import Path
from collections import Counter, OrderedDict
import random
import numpy as np
from molgen3D.evaluation.rdkit_utils import correct_smiles, clean_confs
from molgen3D.config.paths import get_base_path, load_paths_yaml


random.seed(43)
np.random.seed(43)

_CONFIG_DATA = load_paths_yaml()
_GEOM_MAPPING = {
    key.lower(): value
    for key, value in (_CONFIG_DATA.get("data", {}).get("GEOM") or {}).items()
}
if not _GEOM_MAPPING:
    raise KeyError("paths.yaml missing data.GEOM mapping for dataset folders.")

DATASET_ORDER = tuple(_GEOM_MAPPING.keys())

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return cloudpickle.load(f)

    
def load_corrected_smiles_map(csv_path: str):
    corrected_smiles_map = {}
    if not os.path.exists(csv_path):
        print(f"Corrected SMILES CSV not found at {csv_path}. Continuing without corrections.")
        return corrected_smiles_map

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            print(f"Reading CSV header: {','.join(header)}")
        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            geom_smiles = row[0].strip()
            corrected = row[2].strip() if len(row) >= 3 else None
            if not geom_smiles:
                print(f"Skipping row {row_idx}: missing geom SMILES")
                continue
            corrected_smiles_map[geom_smiles] = corrected or None

    print(f"Loaded corrected SMILES for {len(corrected_smiles_map)} molecules from CSV.")
    return corrected_smiles_map


def _get_dataset_folder(dataset: str) -> str:
    folder = _GEOM_MAPPING.get(dataset.lower())
    if folder is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available datasets: {sorted(_GEOM_MAPPING.keys())}"
        )
    return folder


def _resolve_base_path(base_path: str | None) -> Path:
    if base_path:
        candidate = Path(base_path).expanduser()
        return candidate if candidate.is_absolute() else candidate.resolve()
    return get_base_path("geom_dataset_root")


def process_dataset(dataset: str, process_type: str, base_path: Path) -> dict:
    dataset_key = dataset.upper()
    folder_name = _get_dataset_folder(dataset)

    base_path = Path(base_path)
    dataset_path = base_path / folder_name
    test_mols_path = dataset_path / "test_smiles.csv"
    test_pkl_path = dataset_path / "test_mols.pkl"

    if dataset_key == "DRUGS":
        output_name = f"{process_type}_smi.pickle"
    elif dataset_key == "QM9":
        output_name = "qm9_smi.pickle"
    elif dataset_key == "XL":
        output_name = "xl_smi.pickle"
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    destination_path = get_base_path("data_root")
    output_path = destination_path / output_name

    print(f"Processing {dataset_key} dataset...")
    print(f"Test molecules path: {test_mols_path}")
    print(f"Pickle file path: {test_pkl_path}")
    print(f"Output file: {output_path}")

    # Load the dictionary of molecules (smiles -> [mol_objects])
    print("Loading test_mols.pkl...")
    mol_dic = load_pkl(test_pkl_path)
    print(f"Loaded {len(mol_dic)} molecules from pickle.")

    corrected_smiles_map = load_corrected_smiles_map(test_mols_path)

    processed_drugs_test = {}
    conf_count, mol_count = 0, 0
    total_mols = len(mol_dic)
    
    for i, (geom_smiles, true_confs) in enumerate(mol_dic.items(), start=1):
        geom_smiles_corrected = corrected_smiles_map.get(geom_smiles)

        try:
            num_confs = len(true_confs)

            if process_type == "clean":
                true_confs = clean_confs(geom_smiles, true_confs)
                num_confs = len(true_confs)
                if num_confs == 0:
                    continue
                corrected_smi = correct_smiles(true_confs)
            else:
                corrected_smi = None 

            gn_count = Counter([Chem.MolToSmiles(Chem.RemoveHs(c), canonical=True, isomericSmiles=True) for c in true_confs])

            sample_dict = {
                "geom_smiles": geom_smiles,
                "geom_smiles_c": geom_smiles_corrected,
                "confs": true_confs,
                "num_confs": num_confs,
                "pickle_path": None,
                "sub_smiles_counts": gn_count,
                "corrected_smi": corrected_smi,
            }
            processed_drugs_test[geom_smiles] = sample_dict
            mol_count += 1
            if i % 100 == 0:
                print(f"Processed {i}/{total_mols}: num confs {num_confs}, {geom_smiles[:20]}...")
            conf_count += num_confs
            
        except Exception as e:
            print(f"{i} {geom_smiles} --- Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"[{dataset_key}] processed molecules: {mol_count}")
    print(f"[{dataset_key}] processed conformers: {conf_count}")

    sorted_data = OrderedDict(
        sorted(processed_drugs_test.items(), key=lambda item: len(item[1]['geom_smiles']))
    )

    with open(output_path, 'wb') as f:
        cloudpickle.dump(sorted_data, f, protocol=4)

    print(f"[{dataset_key}] saved processed data to {output_path}")

    return {
        "dataset": dataset_key,
        "molecules": mol_count,
        "conformers": conf_count,
        "output_path": str(output_path),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process test dataset for MolGen3D")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["drugs", "qm9", "xl", "all"],
        required=True,
        help="Dataset to process (or 'all' to process every dataset).",
    )
    parser.add_argument(
        "--process_type",
        type=str,
        default="distinct",
        choices=["distinct", "clean"],
        help="Process type (default: distinct)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help=(
            "Optional override for the dataset root on disk. "
            "Defaults to the 'geom_dataset_root' base path configured in paths.yaml."
        ),
    )
    
    args = parser.parse_args()
    
    resolved_base_path = _resolve_base_path(args.base_path)
    print(f"Resolved base path to {resolved_base_path}")

    if args.dataset.lower() == "all":
        datasets_to_run = DATASET_ORDER
    else:
        datasets_to_run = (args.dataset.lower(),)

    aggregate_stats = []
    total_mols, total_confs = 0, 0

    for ds in datasets_to_run:
        stats = process_dataset(ds, args.process_type, resolved_base_path)
        aggregate_stats.append(stats)
        total_mols += stats["molecules"]
        total_confs += stats["conformers"]

    print("\n=== Processing summary ===")
    for stats in aggregate_stats:
        print(
            f"{stats['dataset']}: {stats['molecules']} molecules | "
            f"{stats['conformers']} conformers -> {stats['output_path']}"
        )

    if len(aggregate_stats) > 1:
        print(f"TOTAL: {total_mols} molecules | {total_confs} conformers")