import argparse
import json
import os
import os.path as osp
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger as log
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from molgen3D.data_processing.smiles_encoder_decoder import (
    encode_cartesian_binned,
    encode_cartesian_binned_v2,
    encode_cartesian_with_config,
)
from molgen3D.data_processing.utils import (
    JsonlSplitWriter,
    filter_revisited_mols,
    get_embedding_func_and_config,
    get_revisited_split_path,
    parse_coordinate_ranges,
    save_processed_pickle,
)

RDLogger.DisableLog("rdApp.*")

_REVISITED_DATA: Optional[List] = None


def _process_revisited_mol_impl(
    idx: int,
    max_confs: int,
    precision: int,
    embedding_func: Any,
    bin_size: float,
    parsed_ranges: List[Tuple[float, float]],
    do_filter: bool,
    bin_config: Any,
    use_isomeric_smiles: bool,
) -> Tuple[List[str], Dict[str, Any], List[Chem.Mol]]:
    smiles, mols = _REVISITED_DATA[idx]
    local_failures: Dict[str, int] = defaultdict(int)

    valid = filter_revisited_mols(
        smiles=smiles,
        mols=mols,
        failures=local_failures,
        max_confs=max_confs,
    )

    nonisomeric_smiles, dotted_smiles, isomeric_smiles = set(), set(), set()
    samples: List[str] = []
    filtered_mols: List[Chem.Mol] = []

    for mol in valid:
        if do_filter:
            pos = mol.GetConformer().GetPositions()
            out_of_range = False
            for i in range(3):
                min_val, max_val = parsed_ranges[i]
                if np.any(pos[:, i] < min_val) or np.any(pos[:, i] > max_val):
                    out_of_range = True
                    break
            if out_of_range:
                local_failures["coord_out_of_range"] += 1
                continue

        try:
            if embedding_func is encode_cartesian_with_config:
                embedded_smile, iso_smile = embedding_func(mol, bin_config)
            elif embedding_func in (encode_cartesian_binned, encode_cartesian_binned_v2):
                embedded_smile, iso_smile = embedding_func(
                    mol,
                    bin_size=bin_size,
                    ranges=parsed_ranges,
                )
            else:
                embedded_smile, iso_smile = embedding_func(mol, precision=precision)
        except Exception as exc:
            log.error("Error encoding conformer | smiles={} | failure={}", smiles, exc)
            local_failures["encoding_error"] += 1
            continue

        noniso = None
        try:
            noniso = Chem.MolToSmiles(
                Chem.RemoveHs(mol, sanitize=False),
                canonical=True,
                isomericSmiles=False,
            )
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

    if not samples:
        local_failures["no_samples_after_filtering"] += 1

    stats = {
        "path": smiles,
        "geom_smiles": smiles,
        "confs_count_pre_filter": len(mols),
        "confs_count_post_filter": len(samples),
        "nonisomeric_smiles_post_filter": len(nonisomeric_smiles),
        "isomeric_smiles_post_filter": isomeric_smiles,
        "num_distinct_smiles_with_dot": len(dotted_smiles),
        "has_dotted_smiles": bool(dotted_smiles),
        "failures": local_failures,
        "processed_pickle_path": None,
    }
    return samples, stats, filtered_mols


def _process_revisited_by_idx(args: Tuple) -> Optional[Tuple]:
    idx = args[0]
    try:
        return _process_revisited_mol_impl(idx, *args[1:])
    except Exception as exc:
        log.error("Unhandled exception | idx={} | error={}", idx, exc)
        return None


def preprocess_revisited(
    geom_raw_path: str,
    embedding_type: str,
    num_workers: int = 20,
    precision: int = 4,
    splits: Optional[str] = None,
    dest_path: Optional[str] = None,
    max_confs: int = 30,
    bin_size: float = 0.104,
    ranges: str = "[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
    filter_ranges: str = None,
    bin_config_path: Optional[str] = None,
    isomeric: bool = False,
    use_centered: bool = True,
) -> None:
    global _REVISITED_DATA

    if dest_path is None:
        raise ValueError("dest_path must be provided for preprocessing output")

    embedding_func, bin_config = get_embedding_func_and_config(
        embedding_type=embedding_type,
        bin_config_path=bin_config_path,
    )
    parsed_ranges = parse_coordinate_ranges(ranges)

    do_filter = False
    if filter_ranges is not None:
        if isinstance(filter_ranges, str):
            do_filter = filter_ranges.lower() in ("true", "1", "yes", "on")
        else:
            do_filter = bool(filter_ranges)

    requested_splits = [splits] if splits else ["train", "valid", "test"]

    strings_root = osp.join(dest_path, "processed_strings")
    split_writers = {
        split: JsonlSplitWriter(osp.join(strings_root, split), split, chunk_size=50_000)
        for split in ("train", "valid", "test")
    }
    split_pickle_dirs = {
        split: osp.join(dest_path, "processed_pickles", split) for split in ("train", "valid", "test")
    }
    for split_dir in split_pickle_dirs.values():
        os.makedirs(split_dir, exist_ok=True)

    overall_total_input = overall_total_confs = overall_total_mols = 0
    overall_failure_counts: Dict[str, int] = defaultdict(int)

    for split_name in requested_splits:
        data_path = get_revisited_split_path(
            geom_raw_path=geom_raw_path,
            split_name=split_name,
            use_centered=use_centered,
        )
        if not osp.exists(data_path):
            raise FileNotFoundError(f"Revisited split file not found: {data_path}")

        log.info("Loading {} ...", data_path)
        with open(data_path, "rb") as fh:
            _REVISITED_DATA = pickle.load(fh)
        log.info("Loaded {:,} molecules for split '{}'", len(_REVISITED_DATA), split_name)

        n = len(_REVISITED_DATA)
        overall_total_input += n

        job_args = [
            (
                idx,
                max_confs,
                precision,
                embedding_func,
                bin_size,
                parsed_ranges,
                do_filter,
                bin_config,
                isomeric,
            )
            for idx in range(n)
        ]

        conf_count_pre = conf_count_post = mol_count_post = 0
        split_num_mol_with_multi_distinct_graphs = split_num_mol_with_dotted_smiles = total_dotted_smiles = 0
        failure_counts: Dict[str, int] = defaultdict(int)
        split_geom_to_iso_map: Dict[str, Set[str]] = defaultdict(set)

        geom_to_iso_path = osp.join(dest_path, f"{split_name}_geom_to_isomeric_smiles.jsonl")
        geom_to_iso_fh = open(geom_to_iso_path, "w")
        try:
            with tqdm(total=n, dynamic_ncols=True, mininterval=0.2) as pbar:
                with Pool(processes=num_workers) as pool:
                    chunk_size = max(1, n // max(num_workers * 2, 1))
                    for result in pool.imap_unordered(
                        _process_revisited_by_idx,
                        job_args,
                        chunksize=chunk_size,
                    ):
                        if result is None:
                            failure_counts["unhandled_exception"] += 1
                            pbar.update()
                            continue

                        samples, stats, filtered_mols = result
                        split_writers[split_name].write(samples)

                        geom_smiles = stats.get("geom_smiles")
                        if filtered_mols and geom_smiles:
                            try:
                                processed_pickle_path = save_processed_pickle(
                                    split_dir=split_pickle_dirs[split_name],
                                    geom_smiles=geom_smiles,
                                    mols=filtered_mols,
                                )
                                stats["processed_pickle_path"] = processed_pickle_path
                            except Exception as exc:
                                log.error(
                                    "Failed to write processed pickle | geom_smiles={} | failure={}",
                                    geom_smiles,
                                    exc,
                                )

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

                        if geom_smiles:
                            for iso in stats.get("isomeric_smiles_post_filter", ()):
                                split_geom_to_iso_map[geom_smiles].add(iso)
                                geom_to_iso_fh.write(
                                    json.dumps(
                                        {
                                            "geom_smiles": geom_smiles,
                                            "isomeric_smiles": iso,
                                        },
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )

                        pbar.update()
        finally:
            geom_to_iso_fh.close()
            _REVISITED_DATA = None

        avg_confs = conf_count_post / mol_count_post if mol_count_post else 0.0
        success_rate = mol_count_post / n if n else 0.0
        split_report = {
            "split": split_name,
            "num_input_molecules": n,
            "num_output_molecules": mol_count_post,
            "num_input_conformers": conf_count_pre,
            "total_conformers_after": conf_count_post,
            "avg_conformers_per_molecule_after": avg_confs,
            "success_rate": success_rate,
            "failure_counts": dict(failure_counts),
            "molecules_with_multiple_distinct_graphs": split_num_mol_with_multi_distinct_graphs,
            "molecules_with_dotted_smiles": split_num_mol_with_dotted_smiles,
            "total_dotted_smiles": total_dotted_smiles,
        }
        log.info(json.dumps({"split_summary": split_report}, ensure_ascii=False, separators=(",", ":")))

        for reason, count in failure_counts.items():
            overall_failure_counts[reason] += count

    for writer in split_writers.values():
        writer.close()

    grand_total = sum(writer.total_samples for writer in split_writers.values())
    overall_success_rate = float(overall_total_mols) / max(1, overall_total_input)
    run_summary = {
        "grand_total_samples_written": grand_total,
        "total_input_molecules": overall_total_input,
        "molecules_after_filter": overall_total_mols,
        "conformers_after_filter": overall_total_confs,
        "avg_confs_per_mol_after": float(overall_total_confs) / max(1, overall_total_mols),
        "overall_success_rate": overall_success_rate,
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
        required=True,
        help="Path to the revisited GEOM split pickle files.",
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
        choices=[
            "cartesian",
            "cartesian_v2",
            "cartesian_binned",
            "cartesian_binned_v2",
            "uniform_binned",
            "quantile_binned",
        ],
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
        "--max_confs",
        type=int,
        default=30,
        help="Maximum number of conformers per molecule.",
    )
    parser.add_argument(
        "--bin_size",
        type=float,
        default=0.104,
        help="Bin size for binned embedding.",
    )
    parser.add_argument(
        "--ranges",
        type=str,
        default="[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
        help="Ranges for binned embedding.",
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
        "--use_isomeric_smiles",
        action="store_true",
        help="If set, writes isomeric SMILES to canonical_smiles; otherwise writes non-isomeric SMILES.",
    )
    parser.add_argument(
        "--isomeric",
        action="store_true",
        help="Alias for --use_isomeric_smiles.",
    )
    parser.add_argument(
        "--use_centered",
        action="store_true",
        default=False,
        help="Load *_data_centered.pickle instead of *_data.pickle.",
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

    preprocess_revisited(
        geom_raw_path=args.geom_raw_path,
        embedding_type=args.embedding_type,
        dest_path=dest_path,
        max_confs=args.max_confs,
        num_workers=args.num_workers,
        precision=args.precision,
        splits=args.splits,
        bin_size=args.bin_size,
        ranges=args.ranges,
        filter_ranges=args.filter_ranges,
        bin_config_path=args.bin_config_path,
        isomeric=args.isomeric or args.use_isomeric_smiles,
        use_centered=args.use_centered,
    )
