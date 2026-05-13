import argparse
import glob
import json
import os
import os.path as osp
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from loguru import logger as log
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from molgen3D.data_processing.utils import (
    encode_mol_with_embedding,
    JsonlSplitWriter,
    filter_mols,
    get_coordinate_ranges_for_embedding,
    get_embedding_func_and_config,
    save_processed_pickle,
)
from molgen3D.data_processing.smiles_encoder_decoder import BinConfig
from molgen3D.utils.utils import load_pkl

RDLogger.DisableLog("rdApp.*")


def read_mol(
    args: Tuple,
) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    mol_path, max_confs, precision, embedding_func, bin_size, ranges, do_filter, pickle_dir, _geom_root = args[:9]
    bin_config = args[9] if len(args) > 9 else None
    use_isomeric_smiles = args[10] if len(args) > 10 else False
    try:
        return _read_mol_impl(
            mol_path, max_confs, precision, embedding_func, bin_size, ranges, do_filter, pickle_dir,
            bin_config=bin_config,
            use_isomeric_smiles=use_isomeric_smiles,
        )
    except Exception as exc:
        log.error("Unhandled exception in read_mol | path={} | error={}", mol_path, exc)
        return None


def _read_mol_impl(
    mol_path: str,
    max_confs: int,
    precision: int,
    embedding_func: Any,
    bin_size: float,
    ranges: List[Tuple[float, float]],
    do_filter: bool,
    pickle_dir: str,
    bin_config: Optional[BinConfig] = None,
    use_isomeric_smiles: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    mol_object = load_pkl(mol_path)
    geom_smiles = mol_object["smiles"]

    local_failures: Dict[str, int] = defaultdict(int)
    mols = filter_mols(mol_object, failures=local_failures, max_confs=max_confs)

    nonisomeric_smiles, dotted_smiles, isomeric_smiles = set(), set(), set()
    samples: List[str] = []
    filtered_mols: List[Chem.Mol] = []

    for mol in mols:
        if do_filter:
            pos = mol.GetConformer().GetPositions()
            out_of_range = False
            for i in range(3):
                min_val, max_val = ranges[i]
                if np.any(pos[:, i] < min_val) or np.any(pos[:, i] > max_val):
                    out_of_range = True
                    break
            if out_of_range:
                local_failures["coord_out_of_range"] += 1
                continue

        try:
            embedded_smile, iso_smile = encode_mol_with_embedding(
                mol,
                embedding_func,
                precision=precision,
                bin_size=bin_size,
                ranges=ranges,
                bin_config=bin_config,
            )
        except Exception as exc:
            log.error("Error encoding conformer | path={} | failure={}", mol_path, exc)
            local_failures["encoding_error"] += 1
            continue

        # Compute nonisomeric SMILES only for conformers that encoded successfully
        noniso = None
        try:
            noniso = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), canonical=True, isomericSmiles=False)
            nonisomeric_smiles.add(noniso)
            if "." in noniso:
                dotted_smiles.add(noniso)
        except Exception:
            pass

        canonical_smiles = iso_smile if use_isomeric_smiles else (noniso or iso_smile)
        samples.append(
            json.dumps(
                {
                    "canonical_smiles": canonical_smiles,
                    "embedded_smiles": embedded_smile,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        isomeric_smiles.add(iso_smile)
        filtered_mols.append(mol)

    if len(nonisomeric_smiles) > 1:
        log.info(
            "multiple_distinct_nonisomeric_smiles | path={} | distinct_smiles={}",
            mol_path,
            nonisomeric_smiles,
        )
        for dotted in dotted_smiles:
            log.info("dot_in_conformer_smiles | path={} | smile={}", mol_path, dotted)

    processed_pickle_path = None
    if filtered_mols and pickle_dir:
        try:
            processed_pickle_path = save_processed_pickle(
                split_dir=pickle_dir,
                geom_smiles=geom_smiles,
                mols=filtered_mols,
            )
        except Exception as exc:
            log.error("Failed to write processed pickle | path={} | failure={}", mol_path, exc)

    if not samples:
        log.warning("No samples after filtering | path={}", mol_path)
        local_failures["no_samples_after_filtering"] += 1

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
    bin_size: float = 0.104,
    ranges: str = "[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
    filter_ranges: str = None,
    bin_config_path: Optional[str] = None,
    use_isomeric_smiles: bool = False,
) -> None:
    if dest_path is None:
        raise ValueError("dest_path must be provided for preprocessing output")

    embedding_func, bin_config = get_embedding_func_and_config(
        embedding_type=embedding_type,
        bin_config_path=bin_config_path,
    )

    overall_total_input_mols = overall_total_confs = overall_total_mols = 0
    overall_multi_distinct_graphs = overall_mol_with_dotted_smiles = overall_total_dotted_smiles = 0
    overall_failure_counts: Dict[str, int] = defaultdict(int)

    strings_root = osp.join(dest_path, "processed_strings")
    split_writers = {
        split: JsonlSplitWriter(osp.join(strings_root, split), split)
        for split in ("train", "valid", "test")
    }
    split_pickle_dirs = {
        split: osp.join(dest_path, "processed_pickles", split) for split in ("train", "valid", "test")
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

    parsed_ranges = get_coordinate_ranges_for_embedding(ranges, bin_config=bin_config)

    do_filter = False
    if filter_ranges is not None:
        if isinstance(filter_ranges, str):
            do_filter = filter_ranges.lower() in ("true", "1", "yes", "on")
        else:
            do_filter = bool(filter_ranges)

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
        split_geom_to_iso_map: Dict[str, Set[str]] = defaultdict(set)
        failure_counts: Dict[str, int] = defaultdict(int)

        job_args = [
            (
                path,
                max_confs,
                precision,
                embedding_func,
                bin_size,
                parsed_ranges,
                do_filter,
                split_pickle_dirs[split_name],
                geom_raw_path,
                bin_config,
                use_isomeric_smiles,
            )
            for path in mol_paths
        ]

        with tqdm(total=len(job_args), dynamic_ncols=True, mininterval=0.2) as pbar:
            with Pool(processes=num_workers) as pool:
                chunk_size = max(1, len(job_args) // max(num_workers * 2, 1))
                processed = 0

                for result in pool.imap_unordered(read_mol, job_args, chunksize=chunk_size):
                    if result is None:
                        failure_counts["unhandled_exception"] += 1
                        processed += 1
                        pbar.update()
                        continue

                    samples, stats = result
                    split_writers[split_name].write(samples)

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

                    geom_smiles = stats.get("geom_smiles")
                    if geom_smiles:
                        for iso in stats.get("isomeric_smiles_post_filter", ()):
                            split_geom_to_iso_map[geom_smiles].add(iso)

                    processed += 1
                    pbar.update()
                    if (processed & 63) == 0:
                        pbar.refresh()

        total_distinct_isos = sum(len(v) for v in split_geom_to_iso_map.values())
        avg_confs_per_mol = conf_count_post / mol_count_post if mol_count_post else 0.0
        success_rate = mol_count_post / split_total_input if split_total_input else 0.0
        split_report = {
            "split": split_name,
            "num_input_molecules": split_total_input,
            "num_output_molecules": mol_count_post,
            "num_input_conformers": conf_count_pre,
            "total_conformers_after": conf_count_post,
            "avg_conformers_per_molecule_after": avg_confs_per_mol,
            "success_rate": success_rate,
            "failure_counts": dict(failure_counts),
            "molecules_with_multiple_distinct_graphs": split_num_mol_with_multi_distinct_graphs,
            "molecules_with_dotted_smiles": split_num_mol_with_dotted_smiles,
            "num_distinct_isomeric_smiles": total_distinct_isos,
            "total_dotted_smiles": total_dotted_smiles,
        }
        log.info(json.dumps({"split_summary": split_report}, ensure_ascii=False, separators=(",", ":")))

        overall_multi_distinct_graphs += split_num_mol_with_multi_distinct_graphs
        overall_mol_with_dotted_smiles += split_num_mol_with_dotted_smiles
        overall_total_dotted_smiles += total_dotted_smiles
        for reason, count in failure_counts.items():
            overall_failure_counts[reason] += count

        mapping_path = osp.join(dest_path, f"{split_name}_geom_to_isomeric_smiles.jsonl")
        with open(mapping_path, "w") as fh:
            for geom_smiles in sorted(split_geom_to_iso_map.keys()):
                iso_smiles_set = split_geom_to_iso_map[geom_smiles]
                for iso_smiles in sorted(iso_smiles_set):
                    fh.write(
                        json.dumps(
                            {
                                "geom_smiles": geom_smiles,
                                "isomeric_smiles": iso_smiles,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

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
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geom_raw_path",
        "-p",
        type=str,
        default="/data/molgen/rdkit_folder",
        help="Path to the GEOM rdkit folder.",
    )
    parser.add_argument(
        "--dest",
        "-d",
        type=str,
        default="/data/molgen/",
        help="Destination directory for processed outputs.",
    )
    parser.add_argument(
        "--embedding_type",
        "-et",
        type=str,
        choices=["cartesian", "cartesian_v2", "cartesian_binned", "cartesian_binned_v2",
                 "uniform_binned", "quantile_binned"],
        default="cartesian_v2",
        help="Embedding type to use for enrichment.",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=max(4, os.cpu_count() or 4),
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
        default="/data/molgen/splits/splits/split0.npy",
        help="Path to numpy file containing split indices.",
    )
    parser.add_argument(
        "--max_confs",
        type=int,
        default=30,
        help="Maximum number of conformers per molecule.",
    )
    parser.add_argument(
        "--bin_size",
        type=float,
        default=0.104,
        help="Legacy bin size for cartesian_binned* embeddings.",
    )
    parser.add_argument(
        "--ranges",
        type=str,
        default="[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
        help="Legacy ranges for cartesian_binned* embeddings.",
    )
    parser.add_argument(
        "--filter_ranges",
        type=str,
        default=None,
        help="Filter ranges for binned embedding.",
    )
    parser.add_argument(
        "--bin_config_path",
        type=str,
        default=None,
        help="Path to BinConfig JSON (required for uniform_binned / quantile_binned).",
    )

    parser.add_argument(
        "--isomeric",
        action="store_true",
        help="Alias for --use_isomeric_smiles.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["energy", "weight", "none"],
        default="energy",
        help="Sort conformers by energy, weight, or keep original order.",
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

    use_isomeric = args.isomeric

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
        bin_size=args.bin_size,
        ranges=args.ranges,
        filter_ranges=args.filter_ranges,
        bin_config_path=args.bin_config_path,
        use_isomeric_smiles=use_isomeric,
    )
    