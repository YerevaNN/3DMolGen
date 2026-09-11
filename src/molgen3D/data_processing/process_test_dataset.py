from rdkit import Chem
import csv
import os
import cloudpickle  # type: ignore
import argparse
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict
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


def process_casf16_dataset(variant: str, casf16_dir: Path | None) -> dict:
    if casf16_dir is None:
        casf16_dir = get_base_path("casf16_root")
    subdir = "ligands" if variant == "casf16" else "ligands_opt"
    ligand_dir = Path(casf16_dir) / subdir
    output_name = "casf16_ligands_smi.pickle" if variant == "casf16" else "casf16_ligands_opt_smi.pickle"
    output_path = get_base_path("data_root") / output_name

    print(f"Processing {variant.upper()} from {ligand_dir}")
    print(f"Output: {output_path}")

    mol_files = defaultdict(list)
    for f in sorted(ligand_dir.glob("*.mol2")):
        parts = f.stem.split("_conf")
        base = parts[0] if len(parts) > 1 else f.stem
        mol_files[base].append(f)

    processed = {}
    mol_count, conf_count = 0, 0
    for base, files in sorted(mol_files.items()):
        mols = []
        for f in sorted(files):
            mol = Chem.MolFromMol2File(str(f), removeHs=True, sanitize=False)
            if mol is None or mol.GetNumConformers() == 0:
                print(f"  Failed: {f}")
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"  Sanitization failed ({f}): {e}")
                continue
            mols.append(mol)
        if not mols:
            continue
        geom_smiles = Chem.MolToSmiles(mols[0], isomericSmiles=True)
        sub_smiles_counts = Counter(
            [Chem.MolToSmiles(Chem.RemoveHs(m), canonical=True, isomericSmiles=True) for m in mols]
        )
        processed[geom_smiles] = {
            "geom_smiles": geom_smiles,
            "confs": mols,
            "num_confs": len(mols),
            "sub_smiles_counts": sub_smiles_counts,
            "corrected_smi": None,
        }
        mol_count += 1
        conf_count += len(mols)

    sorted_data = OrderedDict(
        sorted(processed.items(), key=lambda item: len(item[0]))
    )
    with open(output_path, "wb") as fh:
        cloudpickle.dump(sorted_data, fh, protocol=4)

    print(f"[{variant.upper()}] {mol_count} molecules | {conf_count} conformers -> {output_path}")
    return {"dataset": variant, "molecules": mol_count, "conformers": conf_count, "output_path": str(output_path)}


_CASF16_CHEMBL_VARIANTS = {
    "casf16_ref_chembl": {
        "csv": "casf16_ref_chembl3d_exact_intersection.csv",
        "ligands": "CASF16_REF/ref_chembl3d_exact_intersection_ligands",
        "output": "casf16_ref_chembl_smi.pickle",
    },
    "casf16_core_chembl": {
        "csv": "casf16_core_chembl3d_exact_intersection.csv",
        "ligands": "CASF16/core_chembl3d_exact_intersection_ligands",
        "output": "casf16_core_chembl_smi.pickle",
    },
}


def process_casf16_chembl_dataset(variant: str, source_dir: Path | None) -> dict:
    """Build a test set from the CASF16 x ChEMBL-3D exact-intersection CSVs.

    Rows are keyed by ``casf_heavy_isomeric_smiles`` (byte-identical to the matched
    ``chembl3d_isomeric_smiles``), so ligands sharing a SMILES collapse into one
    prompt. The crystal pose from each row's mol2 is kept as a ground-truth conf so
    the pickle stays usable by run_eval later, but inference only needs
    ``sub_smiles_counts``.
    """
    spec = _CASF16_CHEMBL_VARIANTS[variant]
    if source_dir is None:
        source_dir = get_base_path("casf16_chembl_root")
    source_dir = Path(source_dir)
    csv_path = source_dir / spec["csv"]
    ligand_dir = source_dir / spec["ligands"]
    output_path = get_base_path("data_root") / spec["output"]

    print(f"Processing {variant} from {csv_path}")
    print(f"Ligand mol2 dir: {ligand_dir}")
    print(f"Output: {output_path}")

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Read {len(rows)} rows.")

    processed: dict = {}
    conf_count, mol2_failures = 0, 0
    for row in rows:
        smiles = (row.get("casf_heavy_isomeric_smiles") or "").strip()
        if not smiles:
            print(f"Skipping row without SMILES: {row.get('ligand_id')}")
            continue

        entry = processed.setdefault(
            smiles,
            {
                "geom_smiles": smiles,
                "confs": [],
                "num_confs": 0,
                "sub_smiles_counts": Counter(),
                "corrected_smi": None,
                "ligand_ids": [],
                "chembl3d_mol_ids": [],
            },
        )
        entry["sub_smiles_counts"][smiles] += 1
        entry["ligand_ids"].append(row.get("ligand_id"))
        entry["chembl3d_mol_ids"].append(row.get("chembl3d_mol_id"))

        mol_path = ligand_dir / (row.get("source_file") or "")
        mol = (
            Chem.MolFromMol2File(str(mol_path), removeHs=True, sanitize=False)
            if mol_path.exists()
            else None
        )
        if mol is None or mol.GetNumConformers() == 0:
            print(f"  Failed to read mol2: {mol_path}")
            mol2_failures += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"  Sanitization failed ({mol_path}): {e}")
            mol2_failures += 1
            continue
        entry["confs"].append(mol)
        conf_count += 1

    for entry in processed.values():
        entry["num_confs"] = len(entry["confs"])

    sorted_data = OrderedDict(sorted(processed.items(), key=lambda item: len(item[0])))
    with open(output_path, "wb") as fh:
        cloudpickle.dump(sorted_data, fh, protocol=4)

    print(
        f"[{variant}] {len(sorted_data)} unique SMILES | {conf_count} conformers | "
        f"{mol2_failures} mol2 failures -> {output_path}"
    )
    return {
        "dataset": variant,
        "molecules": len(sorted_data),
        "conformers": conf_count,
        "output_path": str(output_path),
    }


def process_rnp_dataset(source_dir: Path | None) -> dict:
    """Build a test set from the RNP crystal-ligand release.

    ``rnp_1346_smiles.csv`` lists one row per PDB system (``system_id``, ``smiles``,
    ``num_heavy_atoms``) and ``ligand_sdfs/{system_id}.sdf`` holds that system's
    crystal pose. Rows are keyed by canonical isomeric SMILES, so systems sharing a
    ligand collapse into a single prompt with several crystal conformers.
    """
    if source_dir is None:
        source_dir = get_base_path("rnp_root")
    source_dir = Path(source_dir)
    csv_path = source_dir / "rnp_1346_smiles.csv"
    ligand_dir = source_dir / "ligand_sdfs"
    output_path = get_base_path("data_root") / "rnp_smi.pickle"

    print(f"Processing rnp from {csv_path}")
    print(f"Ligand sdf dir: {ligand_dir}")
    print(f"Output: {output_path}")

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Read {len(rows)} rows.")

    processed: dict = {}
    conf_count, sdf_failures, smiles_mismatches = 0, 0, 0
    for row in rows:
        system_id = (row.get("system_id") or "").strip()
        raw_smiles = (row.get("smiles") or "").strip()
        if not system_id or not raw_smiles:
            print(f"Skipping incomplete row: {row}")
            continue

        ref = Chem.MolFromSmiles(raw_smiles)
        if ref is None:
            print(f"  Unparsable SMILES for {system_id}: {raw_smiles}")
            continue
        smiles = Chem.MolToSmiles(ref, canonical=True, isomericSmiles=True)

        sdf_path = ligand_dir / f"{system_id}.sdf"
        mol = None
        if sdf_path.exists():
            mols = [m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)]
            mol = mols[0] if mols and mols[0] is not None else None
        if mol is None or mol.GetNumConformers() == 0:
            print(f"  Failed to read sdf: {sdf_path}")
            sdf_failures += 1
            continue

        conf_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
        if conf_smiles != smiles:
            print(f"  SMILES mismatch for {system_id}: csv={smiles} sdf={conf_smiles}")
            smiles_mismatches += 1

        entry = processed.setdefault(
            smiles,
            {
                "geom_smiles": smiles,
                "confs": [],
                "num_confs": 0,
                "sub_smiles_counts": Counter(),
                "corrected_smi": None,
                "system_ids": [],
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            },
        )
        entry["sub_smiles_counts"][conf_smiles] += 1
        entry["system_ids"].append(system_id)
        entry["confs"].append(mol)
        conf_count += 1

    for entry in processed.values():
        entry["num_confs"] = len(entry["confs"])

    sorted_data = OrderedDict(sorted(processed.items(), key=lambda item: len(item[0])))
    with open(output_path, "wb") as fh:
        cloudpickle.dump(sorted_data, fh, protocol=4)

    print(
        f"[RNP] {len(sorted_data)} unique SMILES | {conf_count} conformers | "
        f"{sdf_failures} sdf failures | {smiles_mismatches} smiles mismatches -> {output_path}"
    )
    return {
        "dataset": "rnp",
        "molecules": len(sorted_data),
        "conformers": conf_count,
        "output_path": str(output_path),
    }


_DRUGLIKE_INT_FIELDS = ("n_pdbs", "n_systems", "n_instances", "hbd", "hba", "rotb")
_DRUGLIKE_FLOAT_FIELDS = ("mw", "clogp", "tpsa")


def _load_druglike_conformers(conformers_root: Path) -> dict:
    """Index the PDB-derived crystal conformers by shortlist CCD code.

    ``conformer_manifest.tsv`` has one row per extracted ligand instance, giving the
    shortlist CCD it belongs to plus the ``raw/{ccd}_{name}/{ligand_id}.sdf`` file
    holding that instance's crystal pose. Rows whose ``status`` is not ``extracted``
    never produced a file (the ligand was not resolvable in the PLINDER system) and
    are skipped.

    The manifest records absolute paths from the machine that built it, so each SDF
    is looked up under ``conformers_root`` first and only falls back to the recorded
    path, keeping the dataset relocatable.
    """
    manifest_path = conformers_root / "conformer_manifest.tsv"
    if not manifest_path.exists():
        print(f"Conformer manifest not found at {manifest_path}. Building prompt-only test set.")
        return {}

    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    by_ccd: dict[str, list[dict]] = defaultdict(list)
    unresolved = 0
    for row in rows:
        if (row.get("status") or "").strip() != "extracted":
            unresolved += 1
            continue
        recorded = Path((row.get("local_sdf") or "").strip())
        # raw/{ccd}_{name}/{ligand_id}.sdf relative to the conformers root.
        relocated = conformers_root / "raw" / recorded.parent.name / recorded.name
        row["_sdf_path"] = relocated if relocated.exists() else recorded
        by_ccd[(row.get("shortlist_ccd") or "").strip()].append(row)

    print(
        f"Conformer manifest: {len(rows)} rows | {sum(len(v) for v in by_ccd.values())} extracted | "
        f"{unresolved} unresolved -> {len(by_ccd)} CCDs"
    )
    return by_ccd


def process_druglike_dataset(tsv_path: Path | None, conformers_root: Path | None = None) -> dict:
    """Build a test set from the curated drug-like shortlist TSV.

    The shortlist ships SMILES plus PDB-derived metadata (CCD code, how many PDB
    entries/systems/instances contain the ligand). Ground-truth conformers come
    from the PLINDER conformer extraction under ``conformers_root``: every PDB
    instance of a shortlist ligand contributes its crystal pose, so a molecule
    crystallised many times gets many reference conformers and ``run_eval`` can
    compute COV/MAT against them rather than only ``--posebusters-only``.

    Rows are keyed by canonical isomeric SMILES so that two shortlist entries
    resolving to the same molecule collapse into a single prompt. A crystal pose is
    only accepted when its molecular graph matches the shortlist SMILES ignoring
    stereochemistry, which is the same comparison ``clean_confs`` applies at
    evaluation time; truncated or partially modelled ligands are dropped instead of
    being scored as the wrong molecule.

    When the manifest is absent the shortlist still yields a prompt-only test set
    with empty ``confs``, usable with ``run_eval --posebusters-only``.
    """
    if tsv_path is None:
        tsv_path = get_base_path("druglike_shortlist")
    tsv_path = Path(tsv_path)
    if conformers_root is None:
        conformers_root = get_base_path("druglike_conformers_root")
    conformers_root = Path(conformers_root)
    output_path = get_base_path("data_root") / "druglike_smi.pickle"

    print(f"Processing druglike shortlist from {tsv_path}")
    print(f"Conformers root: {conformers_root}")
    print(f"Output: {output_path}")

    with open(tsv_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    print(f"Read {len(rows)} rows.")

    conformers_by_ccd = _load_druglike_conformers(conformers_root)

    processed: dict = {}
    parse_failures = 0
    conf_count, sdf_failures, graph_mismatches, stereo_mismatches = 0, 0, 0, 0
    for row in rows:
        raw_smiles = (row.get("smiles") or "").strip()
        name = (row.get("name") or "").strip()
        if not raw_smiles:
            print(f"Skipping row without SMILES: {name}")
            continue

        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            print(f"  Unparsable SMILES for {name}: {raw_smiles}")
            parse_failures += 1
            continue
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        flat_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

        properties = {"inchikey": (row.get("inchikey") or "").strip() or None}
        for field in _DRUGLIKE_INT_FIELDS:
            value = (row.get(field) or "").strip()
            properties[field] = int(value) if value else None
        for field in _DRUGLIKE_FLOAT_FIELDS:
            value = (row.get(field) or "").strip()
            properties[field] = float(value) if value else None

        entry = processed.setdefault(
            smiles,
            {
                "geom_smiles": smiles,
                "confs": [],
                "num_confs": 0,
                "sub_smiles_counts": Counter(),
                "corrected_smi": None,
                "names": [],
                "ccds": [],
                "categories": [],
                "benchmark_non_covalent": [],
                "pdb_ids": [],
                "system_ids": [],
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "properties": properties,
            },
        )
        entry["names"].append(name)
        entry["ccds"].append((row.get("ccd") or "").strip())
        entry["categories"].append((row.get("category") or "").strip())
        entry["benchmark_non_covalent"].append((row.get("benchmark_non_covalent") or "").strip())

        ccd = (row.get("ccd") or "").strip()
        accepted = 0
        for conf_row in conformers_by_ccd.get(ccd, []):
            sdf_path = conf_row["_sdf_path"]
            conf_mol = None
            if sdf_path.exists():
                mols = [m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)]
                conf_mol = mols[0] if mols and mols[0] is not None else None
            if conf_mol is None or conf_mol.GetNumConformers() == 0:
                print(f"  Failed to read sdf: {sdf_path}")
                sdf_failures += 1
                continue

            no_hs = Chem.RemoveHs(conf_mol)
            conf_flat = Chem.MolToSmiles(no_hs, canonical=True, isomericSmiles=False)
            if conf_flat != flat_smiles:
                # Truncated / partially modelled ligand: a different molecule, so it
                # cannot serve as a reference pose for this prompt.
                print(
                    f"  Graph mismatch for {name} ({conf_row.get('system_ligand_ccd')} in "
                    f"{conf_row.get('system_id')}): shortlist={flat_smiles} sdf={conf_flat}"
                )
                graph_mismatches += 1
                continue

            conf_smiles = Chem.MolToSmiles(no_hs, canonical=True, isomericSmiles=True)
            if conf_smiles != smiles:
                stereo_mismatches += 1

            entry["sub_smiles_counts"][conf_smiles] += 1
            entry["confs"].append(conf_mol)
            entry["pdb_ids"].append((conf_row.get("pdb_id") or "").strip())
            entry["system_ids"].append((conf_row.get("system_id") or "").strip())
            conf_count += 1
            accepted += 1

        if accepted == 0:
            # No crystal pose available: keep a single prompt so the molecule is still
            # generated for posebusters-only evaluation.
            print(f"  No conformers for {name} (ccd={ccd}); keeping prompt-only entry.")
            entry["sub_smiles_counts"][smiles] += 1

    for entry in processed.values():
        entry["num_confs"] = len(entry["confs"])

    sorted_data = OrderedDict(sorted(processed.items(), key=lambda item: len(item[0])))
    with open(output_path, "wb") as fh:
        cloudpickle.dump(sorted_data, fh, protocol=4)

    print(
        f"[DRUGLIKE] {len(sorted_data)} unique SMILES | {conf_count} conformers | "
        f"{parse_failures} SMILES parse failures | {sdf_failures} sdf failures | "
        f"{graph_mismatches} graph mismatches dropped | {stereo_mismatches} stereo mismatches kept "
        f"-> {output_path}"
    )
    return {
        "dataset": "druglike",
        "molecules": len(sorted_data),
        "conformers": conf_count,
        "output_path": str(output_path),
    }


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
                "pickle_path": f"{folder_name.lower()}/{geom_smiles.replace('/', '_')}.pickle",
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
        choices=[
            "drugs",
            "qm9",
            "xl",
            "all",
            "casf16",
            "casf16_opt",
            "casf16_ref_chembl",
            "casf16_core_chembl",
            "casf16_chembl",
            "rnp",
            "druglike",
        ],
        required=True,
        help=(
            "Dataset to process ('all' processes every GEOM dataset, "
            "'casf16_chembl' processes both CASF16 x ChEMBL-3D intersection sets, "
            "'rnp' processes the RNP crystal-ligand release, "
            "'druglike' processes the curated drug-like shortlist TSV)."
        ),
    )
    parser.add_argument(
        "--process_type",
        type=str,
        default="distinct",
        choices=["distinct", "clean"],
        help="Process type for GEOM datasets (default: distinct)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help=(
            "Optional override for the dataset root on disk. "
            "For GEOM datasets defaults to 'geom_dataset_root' in paths.yaml. "
            "For casf16/casf16_opt defaults to 'casf16_root' in paths.yaml. "
            "For the casf16_*_chembl sets defaults to 'casf16_chembl_root'. "
            "For rnp defaults to 'rnp_root'. "
            "For druglike this is the shortlist TSV file, defaulting to "
            "'druglike_shortlist' in paths.yaml."
        ),
    )
    parser.add_argument(
        "--druglike_conformers_root",
        type=str,
        default=None,
        help=(
            "Optional override for the PLINDER conformer extraction directory used to "
            "populate druglike ground-truth conformers. Defaults to "
            "'druglike_conformers_root' in paths.yaml."
        ),
    )

    args = parser.parse_args()

    _CASF16_VARIANTS = ("casf16", "casf16_opt")
    dataset_arg = args.dataset.lower()
    if dataset_arg == "druglike":
        source_path = Path(args.base_path).expanduser() if args.base_path else None
        conformers_root = (
            Path(args.druglike_conformers_root).expanduser()
            if args.druglike_conformers_root
            else None
        )
        stats = process_druglike_dataset(source_path, conformers_root)
        aggregate_stats = [stats]
        total_mols, total_confs = stats["molecules"], stats["conformers"]
    elif dataset_arg == "rnp":
        source_dir = Path(args.base_path).expanduser() if args.base_path else None
        stats = process_rnp_dataset(source_dir)
        aggregate_stats = [stats]
        total_mols, total_confs = stats["molecules"], stats["conformers"]
    elif dataset_arg in _CASF16_CHEMBL_VARIANTS or dataset_arg == "casf16_chembl":
        source_dir = Path(args.base_path).expanduser() if args.base_path else None
        variants = (
            tuple(_CASF16_CHEMBL_VARIANTS) if dataset_arg == "casf16_chembl" else (dataset_arg,)
        )
        aggregate_stats = [process_casf16_chembl_dataset(v, source_dir) for v in variants]
        total_mols = sum(s["molecules"] for s in aggregate_stats)
        total_confs = sum(s["conformers"] for s in aggregate_stats)
    elif dataset_arg in _CASF16_VARIANTS:
        casf16_dir = Path(args.base_path).expanduser() if args.base_path else None
        stats = process_casf16_dataset(dataset_arg, casf16_dir)
        aggregate_stats = [stats]
        total_mols, total_confs = stats["molecules"], stats["conformers"]
    else:
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