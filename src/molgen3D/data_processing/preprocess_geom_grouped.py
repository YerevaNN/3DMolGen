import argparse
import ast
import glob
import json
import os
import os.path as osp
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, Optional, Set, Tuple, List

import numpy as np
from loguru import logger as log
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from molgen3D.data_processing.utils import JsonlSplitWriter
from molgen3D.data_processing.smiles_encoder_decoder import (
    encode_cartesian_binned,
    encode_cartesian_v2,
    encode_cartesian_binned_v2,
)
from molgen3D.utils.utils import load_pkl

RDLogger.DisableLog("rdApp.*")

# Data layout inspection (sample file: /home/chem-project/data/rdkit_folder/drugs/...)
# Energy is stored at conformer["totalenergy"] (float).
# Weight is stored at conformer["boltzmannweight"] (float).
# Conf id is stored at conformer["geom_id"] (int) when present.


def infer_geom_id(mol_path: str) -> str:
    return osp.splitext(osp.basename(mol_path))[0]


def copy_single_conformer_mol(mol: Chem.Mol) -> Chem.Mol:
    copied = Chem.Mol(mol)
    if copied.GetNumConformers() > 1:
        conf = Chem.Conformer(copied.GetConformer(0))
        copied.RemoveAllConformers()
        copied.AddConformer(conf, assignId=True)
    return copied


def _get_prop_float(props: Dict[str, Any], key: str) -> Optional[float]:
    if key in props:
        try:
            return float(props[key])
        except Exception:
            return None
    return None


def extract_conf_meta(
    conf_meta: Optional[Dict[str, Any]], mol: Chem.Mol
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    energy = weight = None
    conf_id = None

    if conf_meta:
        if "totalenergy" in conf_meta:
            energy = conf_meta.get("totalenergy")
        elif "relativeenergy" in conf_meta:
            energy = conf_meta.get("relativeenergy")

        if "boltzmannweight" in conf_meta:
            weight = conf_meta.get("boltzmannweight")
        elif "weight" in conf_meta:
            weight = conf_meta.get("weight")

        if "geom_id" in conf_meta:
            conf_id = conf_meta.get("geom_id")

    if mol is not None:
        mol_props = mol.GetPropsAsDict()
        if energy is None:
            energy = _get_prop_float(mol_props, "totalenergy")
        if weight is None:
            weight = _get_prop_float(mol_props, "boltzmannweight")
        if conf_id is None and "geom_id" in mol_props:
            try:
                conf_id = int(mol_props["geom_id"])
            except Exception:
                conf_id = None

        if mol.GetNumConformers() > 0:
            conf_props = mol.GetConformer().GetPropsAsDict()
            if energy is None:
                energy = _get_prop_float(conf_props, "totalenergy")
            if weight is None:
                weight = _get_prop_float(conf_props, "boltzmannweight")
            if conf_id is None and "geom_id" in conf_props:
                try:
                    conf_id = int(conf_props["geom_id"])
                except Exception:
                    conf_id = None

    try:
        energy = float(energy) if energy is not None else None
    except Exception:
        energy = None

    try:
        weight = float(weight) if weight is not None else None
    except Exception:
        weight = None

    return energy, weight, conf_id


def filter_conformers_keep_dotted(
    mol_object: Dict[str, Any],
    failures: Dict[str, int],
) -> List[Tuple[Chem.Mol, Dict[str, Any], int]]:
    confs = mol_object.get("conformers", [])
    smiles = mol_object.get("smiles")

    if smiles and "." in smiles:
        failures["dot_in_smiles"] += 1

    mol_from_smiles = Chem.MolFromSmiles(smiles) if smiles else None
    if mol_from_smiles is None:
        failures["mol_from_smiles_failed"] += 1
        return []

    selected: List[Tuple[Chem.Mol, Dict[str, Any], int]] = []
    for idx, conf in enumerate(confs):
        mol = conf.get("rd_mol") if isinstance(conf, dict) else None
        if mol is None:
            failures["missing_rd_mol"] += 1
            continue

        num_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if not num_neighbors or np.max(num_neighbors) > 4:
            failures["large_degree"] += 1
            continue

        selected.append((mol, conf if isinstance(conf, dict) else {}, idx))

    return selected


def read_mol(
    args: Tuple[str, int, int, Any, float, List[Tuple[float, float]], str]
) -> Optional[
    Tuple[
        Optional[str],
        Dict[str, Any],
        str,
        Set[str],
        Dict[str, List[Dict[str, Any]]],
    ]
]:
    mol_path = args[0]
    try:
        return _read_mol_impl(args)
    except Exception as exc:
        log.error("Unhandled exception in read_mol | path={} | error={}", mol_path, exc)
        return None


def _read_mol_impl(
    args: Tuple[str, int, int, Any, float, List[Tuple[float, float]], str]
) -> Tuple[
    Optional[str],
    Dict[str, Any],
    str,
    Set[str],
    Dict[str, List[Dict[str, Any]]],
]:
    mol_path, max_confs, precision, embedding_func, bin_size, ranges, sort_by = args
    mol_object = load_pkl(mol_path)
    geom_key = mol_object.get("smiles")
    geom_id = infer_geom_id(mol_path)

    local_failures: Dict[str, int] = defaultdict(int)
    candidates = filter_conformers_keep_dotted(mol_object, failures=local_failures)

    nonisomeric_smiles, dotted_smiles, isomeric_smiles = set(), set(), set()
    sample_isomers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pickle_isomers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    scored_candidates: List[
        Tuple[float, int, Chem.Mol, Dict[str, Any], Optional[float], Optional[float], Optional[int]]
    ] = []
    for mol, conf_meta, idx in candidates:
        energy, weight, conf_id = extract_conf_meta(conf_meta, mol)
        if energy is None:
            local_failures["missing_energy"] += 1
        if sort_by == "energy":
            scored_value = energy if energy is not None else float("inf")
        elif sort_by == "weight":
            # Higher weight is better; missing weights should not displace known weights.
            scored_value = -(weight if weight is not None else float("-inf"))
        else:
            scored_value = float(idx)

        scored_candidates.append((scored_value, idx, mol, conf_meta, energy, weight, conf_id))

    scored_candidates.sort(key=lambda x: (x[0], x[1]))
    scored_candidates = scored_candidates[:max_confs]

    for _, _, mol, _conf_meta, energy, weight, conf_id in scored_candidates:
        try:
            if embedding_func in (encode_cartesian_binned, encode_cartesian_binned_v2):
                embedded_smile, iso_smile = embedding_func(
                    mol,
                    bin_size=bin_size,
                    ranges=ranges,
                )
            else:
                embedded_smile, iso_smile = embedding_func(mol, precision)
        except Exception as exc:
            log.error("Error encoding conformer | path={} | failure={}", mol_path, exc)
            local_failures["encoding_error"] += 1
            continue

        # Compute nonisomeric SMILES only for conformers that encoded successfully
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

        isomeric_smiles.add(iso_smile)

        sample_entry: Dict[str, Any] = {
            "embedded_smiles": embedded_smile,
            "energy": energy,
            "weight": weight,
            "geom_id": str(geom_id),
        }
        if conf_id is not None:
            sample_entry["conf_id"] = conf_id
        sample_isomers[iso_smile].append(sample_entry)

        pickle_entry: Dict[str, Any] = {
            "mol": copy_single_conformer_mol(mol),
            "embedded_smiles": embedded_smile,
            "energy": energy,
            "weight": weight,
            "geom_id": str(geom_id),
        }
        if conf_id is not None:
            pickle_entry["conf_id"] = conf_id
        pickle_isomers[iso_smile].append(pickle_entry)

    if len(nonisomeric_smiles) > 1:
        log.info(
            "multiple_distinct_nonisomeric_smiles | path={} | distinct_smiles={}",
            mol_path,
            nonisomeric_smiles,
        )
        for dotted in dotted_smiles:
            log.info("dot_in_conformer_smiles | path={} | smile={}", mol_path, dotted)

    json_line = None
    if sample_isomers:
        json_line = (
            json.dumps(
                {
                    "geom_key": geom_key,
                    "geom_id": str(geom_id),
                    "isomers": sample_isomers,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    stats = {
        "path": mol_path,
        "geom_smiles": geom_key,
        "confs_count_pre_filter": len(mol_object.get("conformers", [])),
        "confs_count_post_filter": sum(len(v) for v in sample_isomers.values()),
        "nonisomeric_smiles_post_filter": len(nonisomeric_smiles),
        "isomeric_smiles_post_filter": isomeric_smiles,
        "num_distinct_smiles_with_dot": len(dotted_smiles),
        "has_dotted_smiles": bool(dotted_smiles),
        "failures": local_failures,
        "processed_pickle_path": None,
    }

    if not sample_isomers:
        log.warning("No samples after filtering | path={}", mol_path)
        local_failures["no_samples_after_filtering"] += 1

    return json_line, stats, geom_id, isomeric_smiles, pickle_isomers


def save_grouped_pickle(output_path: str, iso_to_confs: Dict[str, List[Dict[str, Any]]]) -> None:
    parent = osp.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(iso_to_confs, fh)


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
    sort_by: str = "energy",
) -> None:
    if dest_path is None:
        raise ValueError("dest_path must be provided for preprocessing output")

    embedding_registry = {
        "cartesian_v2": encode_cartesian_v2,
        "cartesian": encode_cartesian_v2,
        "cartesian_binned": encode_cartesian_binned,
        "cartesian_binned_v2": encode_cartesian_binned_v2,
    }
    if embedding_type not in embedding_registry:
        raise ValueError(f"Unsupported embedding_type '{embedding_type}'. Options: {sorted(embedding_registry)}")
    embedding_func = embedding_registry[embedding_type]

    overall_total_input_mols = overall_total_confs = overall_total_mols = 0
    overall_multi_distinct_graphs = overall_mol_with_dotted_smiles = overall_total_dotted_smiles = 0
    overall_failure_counts: Dict[str, int] = defaultdict(int)

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

    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    requested_splits = [splits] if splits else list(split_name_to_index.keys())
    log.info("Reading files from %s", geom_raw_path)

    split_indices_array = np.load(indices_path, allow_pickle=True)

    try:
        parsed_ranges = ast.literal_eval(f"[{ranges}]")
        parsed_ranges = [tuple(r) for r in parsed_ranges]
    except Exception as exc:
        log.error("Failed to parse ranges: {} | failure={}", ranges, exc)
        parsed_ranges = [(-13.0, 13.0), (-13.0, 13.0), (-13.0, 13.0)]

    pickle_glob = osp.join(geom_raw_path, f"{dataset_type}/*.pickle")
    pickle_paths = np.array(sorted(glob.glob(pickle_glob)))
    if pickle_paths.size == 0:
        raise FileNotFoundError(f"No pickle files found under pattern {pickle_glob}")

    for split_name in requested_splits:
        split_idx = split_name_to_index[split_name]
        split_indices = np.array(sorted(split_indices_array[split_idx]), dtype=int)

        if split_indices.size == 0:
            log.warning("No indices found for split {}", split_name)
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

        geom_to_iso_path = osp.join(dest_path, f"{split_name}_geom_to_isomeric_smiles.jsonl")
        iso_to_geom_path = osp.join(dest_path, f"{split_name}_isomeric_to_geom.jsonl")
        geom_to_iso_fh = open(geom_to_iso_path, "w")
        iso_to_geom_fh = open(iso_to_geom_path, "w")
        try:
            job_args = [
                (
                    path,
                    max_confs,
                    precision,
                    embedding_func,
                    bin_size,
                    parsed_ranges,
                    sort_by,
                )
                for path in mol_paths
            ]

            with tqdm(total=len(job_args), dynamic_ncols=True, mininterval=0.2) as pbar:
                with Pool(processes=num_workers) as pool:
                    chunk_size = max(1, len(job_args) // max(num_workers * 2, 1))
                    processed = 0

                    for result in pool.imap_unordered(read_mol, job_args, chunksize=chunk_size):
                        if result is None:
                            continue

                        json_line, stats, geom_id, iso_set, iso_to_pickle_confs = result

                        if json_line:
                            split_writers[split_name].write([json_line])

                        if iso_to_pickle_confs:
                            processed_pickle_path = osp.join(split_pickle_dirs[split_name], f"{geom_id}.pkl")
                            try:
                                save_grouped_pickle(processed_pickle_path, iso_to_pickle_confs)
                            except Exception as exc:
                                log.error(
                                    "Failed to write processed pickle | path={} | failure={}",
                                    processed_pickle_path,
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

                        geom_smiles = stats.get("geom_smiles")
                        if geom_smiles and iso_set:
                            for iso in iso_set:
                                split_geom_to_iso_map[geom_smiles].add(iso)
                                geom_to_iso_fh.write(
                                    json.dumps(
                                        {
                                            "geom_key": geom_smiles,
                                            "geom_id": str(geom_id),
                                            "isomeric_smiles": iso,
                                        },
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )
                                iso_to_geom_fh.write(
                                    json.dumps(
                                        {
                                            "isomeric_smiles": iso,
                                            "geom_key": geom_smiles,
                                            "geom_id": str(geom_id),
                                        },
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )

                        processed += 1
                        pbar.update()
                        if (processed & 63) == 0:
                            pbar.refresh()
        finally:
            geom_to_iso_fh.close()
            iso_to_geom_fh.close()

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
        choices=["cartesian", "cartesian_v2", "cartesian_binned", "cartesian_binned_v2"],
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
        help="Bin size for binned embedding.",
    )
    parser.add_argument(
        "--ranges",
        type=str,
        default="[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
        help="Ranges for binned embedding.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["energy", "weight", "none"],
        default="energy",
        help="Sort conformers by energy, weight, or keep original order.",
    )

    args = parser.parse_args()

    random.seed(42)
    dest_path = osp.join(args.dest, args.run_name)
    os.makedirs(dest_path, exist_ok=True)
    enqueue_logs = os.environ.get("LOGURU_ENQUEUE", "1") not in {"0", "false", "False"}
    log.add(
        osp.join(dest_path, "preprocessing.log"),
        mode="w",
        enqueue=enqueue_logs,
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
        sort_by=args.sort_by,
        bin_size=args.bin_size,
        ranges=args.ranges,
    )
