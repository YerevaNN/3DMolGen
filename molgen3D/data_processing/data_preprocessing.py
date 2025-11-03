import argparse
import glob
import json
import os
import os.path as osp
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger as log
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from molgen3D.data_processing.smiles_encoder_decoder import encode_cartesian_v2
from molgen3D.data_processing.utils import (
    JsonlSplitWriter,
    encode_cartesian_raw,
    filter_mols,
    save_processed_pickle,
)
from molgen3D.utils.utils import load_pkl

random.seed(42)
RDLogger.DisableLog("rdApp.*")
def read_mol(
    args: Tuple[str, int, int, Any, str, str]
) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    mol_path, max_confs, precision, embedding_func, pickle_dir, geom_root = args
    mol_object = load_pkl(mol_path)
    geom_smiles = mol_object["smiles"]

    local_failures: Dict[str, int] = defaultdict(int)
    mols = filter_mols(mol_object, failures=local_failures, max_confs=max_confs)

    nonisomeric_smiles, dotted_smiles, isomeric_smiles = set(), set(), set()
    samples, filtered_mols = [], []

    for mol in mols:
        try:
            noniso = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), canonical=True, isomericSmiles=False)
            nonisomeric_smiles.add(noniso)
            if "." in noniso:
                dotted_smiles.add(noniso)
        except Exception:
            pass

        try:
            embedded_smile, iso_smile = embedding_func(mol, precision)
        except Exception as exc:
            log.error("Error encoding conformer | path={} | failure={}", mol_path, exc)
            local_failures["encoding_error"] += 1
            continue

        samples.append(
            json.dumps(
                {
                    "canonical_smiles": iso_smile,
                    "embedded_smiles": embedded_smile,
                }
            )
            + "\n"
        )
        isomeric_smiles.add(iso_smile)
        filtered_mols.append(mol)

    if len(samples) == 0:
        log.warning("No samples after filtering | path={}", mol_path)
        local_failures["no_samples_after_filtering"] += 1
        return None

    if len(nonisomeric_smiles) > 1:
        log.info(
            "multiple_distinct_nonisomeric_smiles | path={} | distinct_smiles={}",
            mol_path,
            nonisomeric_smiles,
        )
        for dotted in dotted_smiles:
            log.info("dot_in_conformer_smiles | path={} | smile={}", mol_path, dotted)

    processed_pickle_path = None
    if filtered_mols:
        try:
            processed_pickle_path = save_processed_pickle(
                split_dir=pickle_dir,
                geom_smiles=geom_smiles,
                mols=filtered_mols,
            )
            # log.debug("Saved processed pickle | path={} | output={}", mol_path, processed_pickle_path)
        except Exception as exc:
            log.error("Failed to write processed pickle | path={} | failure={}", mol_path, exc)

    stats = {
        "path": mol_path,
        "geom_smiles": geom_smiles,
        "confs_count_pre_filter": len(mol_object.get("conformers", [])),
        "confs_count_post_filter": len(samples),
        "nonisomeric_smiles_post_filter": len(nonisomeric_smiles),
        "isomeric_smiles_post_filter": isomeric_smiles,
        "num_distinct_smiles_with_dot": len(dotted_smiles),
        "has_dotted_smiles": bool(dotted_smiles),
        "failures": local_failures,
        "processed_pickle_path": processed_pickle_path,
    }

    return samples, stats


def preprocess(
    geom_raw_path: str,
    indices_path: str,
    embedding_type: str,
    num_workers: int = 20,
    precision: int = 4,
    dataset_type: str = "drugs",
    splits: Optional[str] = None,
    dest_path: Optional[str] = None,
    max_confs: int = 30,
) -> None:
    if dest_path is None:
        raise ValueError("dest_path must be provided for preprocessing output")

    embedding_func = encode_cartesian_raw if embedding_type == "cartesian" else encode_cartesian_v2

    overall_total_input_mols = overall_total_confs = overall_total_mols = 0
    overall_multi_distinct_graphs = overall_mol_with_dotted_smiles = overall_total_dotted_smiles = 0
    overall_failure_counts: Dict[str, int] = defaultdict(int)

    strings_root = osp.join(dest_path, "processed_strings")
    pickles_root = osp.join(dest_path, "processed_pickles")
    split_writers = {
        split: JsonlSplitWriter(osp.join(strings_root, split), split)
        for split in ("train", "valid", "test")
    }
    split_pickle_dirs = {
        split: osp.join(pickles_root, split)
        for split in ("train", "valid", "test")
    }
    for split_dir in split_pickle_dirs.values():
        os.makedirs(split_dir, exist_ok=True)

    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    requested_splits = [splits] if splits else list(split_name_to_index.keys())
    log.info("Reading files from %s", geom_raw_path)


    split_indices_array = np.load(indices_path, allow_pickle=True)

    pickle_glob = osp.join(geom_raw_path, f"{dataset_type}/*.pickle")
    pickle_paths = np.array(sorted(glob.glob(pickle_glob)))
    if pickle_paths.size == 0:
        raise FileNotFoundError(f"No pickle files found under pattern {pickle_glob}")

    for split_name in requested_splits:
        split_idx = split_name_to_index[split_name]
        split_indices = np.array(sorted(split_indices_array[split_idx]), dtype=int)

        if split_indices.size == 0:
            log.warning("No indices found for split %s", split_name)
            continue

        if split_indices.max() >= len(pickle_paths):
            raise IndexError(
                f"Split index {split_indices.max()} out of range for available pickle files ({len(pickle_paths)})."
            )

        mol_paths = pickle_paths[split_indices]

        log.info("Processing split %s with %d samples", split_name, len(mol_paths))

        split_total_input = len(mol_paths)
        overall_total_input_mols += split_total_input

        conf_count_post = conf_count_pre = mol_count_post = 0
        split_num_mol_with_multi_distinct_graphs = split_num_mol_with_dotted_smiles = total_dotted_smiles = 0
        split_smiles_map: Dict[str, set] = defaultdict(set)
        failure_counts: Dict[str, int] = defaultdict(int)

        with tqdm(total=len(mol_paths), dynamic_ncols=True, mininterval=0.2) as pbar:
            with Pool(processes=num_workers) as pool:
                chunk_size = max(1, len(mol_paths) // max(num_workers * 8, 1))
                processed = 0

                for result in pool.imap_unordered(
                    read_mol,
                    (
                        (
                            path,
                            max_confs,
                            precision,
                            embedding_func,
                            split_pickle_dirs[split_name],
                            geom_raw_path,
                        )
                        for path in mol_paths
                    ),
                    chunksize=chunk_size,
                ):
                    if result is None:
                        continue

                    samples, stats = result

                    conf_count_pre += stats["confs_count_pre_filter"]
                    conf_count_post += stats["confs_count_post_filter"]
                    overall_total_confs += stats["confs_count_post_filter"]
                    total_dotted_smiles += stats.get("num_distinct_smiles_with_dot", 0)

                    if stats["nonisomeric_smiles_post_filter"] > 1:
                        split_num_mol_with_multi_distinct_graphs += 1
                    if stats.get("has_dotted_smiles", False):
                        split_num_mol_with_dotted_smiles += 1

                    for reason, count in stats["failures"].items():
                        failure_counts[reason] += int(count)

                    if stats["confs_count_post_filter"] > 0:
                        mol_count_post += 1
                        overall_total_mols += 1

                    pickle_path = stats.get("processed_pickle_path")
                    if pickle_path:
                        for iso in stats.get("isomeric_smiles_post_filter", ()):
                            split_smiles_map[iso].add(pickle_path)

                    split_writers[split_name].write(samples)

                    processed += 1
                    pbar.update()
                    if (processed & 63) == 0:
                        pbar.refresh()

        split_report = {
            "split": split_name,
            "num_input_molecules": split_total_input,
            "num_output_molecules": mol_count_post,
            "num_input_conformers": conf_count_pre,
            "total_conformers_after": conf_count_post,
            "avg_conformers_per_molecule_after": float(conf_count_post) / mol_count_post,
            "success_rate": mol_count_post / split_total_input,
            "failure_counts": dict(failure_counts),
            "molecules_with_multiple_distinct_graphs": split_num_mol_with_multi_distinct_graphs,
            "molecules_with_dotted_smiles": split_num_mol_with_dotted_smiles,
            "num_distinct_isomeric_smiles": len(split_smiles_map),
            "total_dotted_smiles": total_dotted_smiles,
        }
        log.info(json.dumps({"split_summary": split_report}, ensure_ascii=False, separators=(",", ":")))

        overall_multi_distinct_graphs += split_num_mol_with_multi_distinct_graphs
        overall_mol_with_dotted_smiles += split_num_mol_with_dotted_smiles
        overall_total_dotted_smiles += total_dotted_smiles
        for reason, count in failure_counts.items():
            overall_failure_counts[reason] += count

        with open(osp.join(dest_path, f"{split_name}_isomeric_smiles_map.jsonl"), "w") as fh:
            for iso_smiles, paths in split_smiles_map.items():
                fh.write(
                    json.dumps({"isomeric_smiles": iso_smiles, "paths": sorted(paths)}, separators=(",", ":"))
                    + "\n"
                )

        with open(osp.join(dest_path, f"{split_name}_distinct_isomeric_smiles.jsonl"), "w") as fh:
            for iso_smiles in sorted(split_smiles_map.keys()):
                fh.write(json.dumps(iso_smiles) + "\n")

    for writer in split_writers.values():
        writer.close()

    grand_total = sum(writer.total_samples for writer in split_writers.values())
    overall_success_rate = float(overall_total_mols) / max(1, overall_total_input_mols)

    run_summary = {
        "grand_total_samples_written": grand_total,
        "total_input_molecules": overall_total_input_mols,
        "molecules_after_filter": overall_total_mols,
        "conformers_after_filter": overall_total_confs,
        "avg_confs_per_mol_after": float(overall_total_confs) / max(1, overall_total_mols),
        "overall_success_rate": overall_success_rate,
        "molecules_with_multiple_distinct_graphs": overall_multi_distinct_graphs,
        "molecules_with_dotted_smiles": overall_mol_with_dotted_smiles,
        "total_dotted_smiles": overall_total_dotted_smiles,
        "overall_failure_counts": dict(overall_failure_counts),
    }
    log.info(json.dumps({"run_summary": run_summary}, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geom_raw_path",
        "-p",
        type=str,
        default="/nfs/ap/mnt/sxtn2/chem/GEOM_data/rdkit_folder",
        help="Path to the GEOM rdkit folder.",
    )
    parser.add_argument(
        "--dest",
        "-d",
        type=str,
        default="/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed",
        help="Destination directory for processed outputs.",
    )
    parser.add_argument(
        "--embedding_type",
        "-et",
        type=str,
        default="cartesian",
        help="Embedding type to use (cartesian, cartesian_v2).",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=4,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Numeric precision for encoded coordinates.",
    )
    parser.add_argument(
        "--dataset_type",
        "-dt",
        type=str,
        default="drugs",
        help="Dataset type (drugs, qm9).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        choices=["train", "valid", "test"],
        default=None,
        help="Optional single split to process.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Run name, appended to destination directory.",
    )
    parser.add_argument(
        "--indices_path",
        type=str,
        default="/nfs/ap/mnt/sxtn2/chem/GEOM_data/splits/splits/split0.npy",
        help="Path to numpy file containing split indices.",
    )
    parser.add_argument(
        "--max_confs",
        type=int,
        default=30,
        help="Maximum number of conformers per molecule.",
    )

    args = parser.parse_args()

    dest_path = osp.join(args.dest, args.run_name)
    os.makedirs(dest_path, exist_ok=True)
    log.add(
        osp.join(dest_path, "preprocessing.log"),
        mode="w",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    preprocess(
        geom_raw_path=args.geom_raw_path,
        indices_path=args.indices_path,
        embedding_type=args.embedding_type,
        dest_path=dest_path,
        max_confs=args.max_confs,
        num_workers=args.num_workers,
        precision=args.precision,
        dataset_type=args.dataset_type,
        splits=args.splits,
    )
