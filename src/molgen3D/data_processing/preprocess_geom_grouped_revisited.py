import argparse
import json
import os
import os.path as osp
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger as log
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from molgen3D.data_processing.utils import (
    encode_mol_with_embedding,
    JsonlSplitWriter,
    copy_single_conformer_mol,
    extract_conf_meta,
    filter_revisited_conformers_keep_dotted,
    get_coordinate_ranges_for_embedding,
    get_embedding_func_and_config,
    get_revisited_split_path,
    save_grouped_pickle,
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
    sort_by: str,
    bin_config: Any,
    use_isomeric_smiles: bool,
) -> Tuple[
    Optional[str],
    Dict[str, Any],
    str,
    Set[str],
    Dict[str, List[Dict[str, Any]]],
]:
    smiles, mols = _REVISITED_DATA[idx]
    geom_id = str(idx)
    local_failures: Dict[str, int] = defaultdict(int)

    candidates = filter_revisited_conformers_keep_dotted(
        smiles=smiles,
        mols=mols,
        failures=local_failures,
    )

    scored_candidates: List[
        Tuple[float, int, Chem.Mol, Optional[float], Optional[float], Optional[int]]
    ] = []
    for mol, conf_idx in candidates:
        energy, weight, conf_id = extract_conf_meta(None, mol)
        if energy is None:
            local_failures["missing_energy"] += 1
        if sort_by == "energy":
            scored_value = energy if energy is not None else float("inf")
        elif sort_by == "weight":
            scored_value = -(weight if weight is not None else float("-inf"))
        else:
            scored_value = float(conf_idx)
        scored_candidates.append((scored_value, conf_idx, mol, energy, weight, conf_id))

    scored_candidates.sort(key=lambda x: (x[0], x[1]))
    scored_candidates = scored_candidates[:max_confs]

    nonisomeric_smiles: Set[str] = set()
    dotted_smiles: Set[str] = set()
    isomeric_smiles: Set[str] = set()
    sample_isomers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pickle_isomers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for _, _, mol, energy, weight, conf_id in scored_candidates:
        try:
            embedded_smile, iso_smile = encode_mol_with_embedding(
                mol,
                embedding_func,
                precision=precision,
                bin_size=bin_size,
                ranges=parsed_ranges,
                bin_config=bin_config,
            )
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
        isomeric_smiles.add(canonical_smiles)

        sample_entry: Dict[str, Any] = {
            "embedded_smiles": embedded_smile,
            "energy": energy,
            "weight": weight,
            "geom_id": geom_id,
        }
        if conf_id is not None:
            sample_entry["conf_id"] = conf_id
        sample_isomers[canonical_smiles].append(sample_entry)

        pickle_entry: Dict[str, Any] = {
            "mol": copy_single_conformer_mol(mol),
            "embedded_smiles": embedded_smile,
            "energy": energy,
            "weight": weight,
            "geom_id": geom_id,
        }
        if conf_id is not None:
            pickle_entry["conf_id"] = conf_id
        pickle_isomers[canonical_smiles].append(pickle_entry)

    if len(nonisomeric_smiles) > 1:
        log.info(
            "multiple_distinct_nonisomeric_smiles | path={} | distinct_smiles={}",
            smiles,
            nonisomeric_smiles,
        )
        for dotted in dotted_smiles:
            log.info("dot_in_conformer_smiles | path={} | smile={}", smiles, dotted)

    if not sample_isomers:
        log.warning("No samples after filtering | path={}", smiles)
        local_failures["no_samples_after_filtering"] += 1

    json_line = None
    if sample_isomers:
        json_line = (
            json.dumps(
                {"geom_key": smiles, "geom_id": geom_id, "isomers": sample_isomers},
                separators=(",", ":"),
            )
            + "\n"
        )

    stats = {
        "path": smiles,
        "geom_smiles": smiles,
        "confs_count_pre_filter": len(mols),
        "confs_count_post_filter": sum(len(v) for v in sample_isomers.values()),
        "nonisomeric_smiles_post_filter": len(nonisomeric_smiles),
        "isomeric_smiles_post_filter": isomeric_smiles,
        "num_distinct_smiles_with_dot": len(dotted_smiles),
        "has_dotted_smiles": bool(dotted_smiles),
        "failures": local_failures,
        "processed_pickle_path": None,
    }
    return json_line, stats, geom_id, isomeric_smiles, pickle_isomers


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
    sort_by: str = "energy",
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
    parsed_ranges = get_coordinate_ranges_for_embedding(ranges, bin_config=bin_config)
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
                sort_by,
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
        iso_to_geom_path = osp.join(dest_path, f"{split_name}_isomeric_to_geom.jsonl")
        geom_to_iso_fh = open(geom_to_iso_path, "w")
        iso_to_geom_fh = open(iso_to_geom_path, "w")
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

                        pbar.update()
        finally:
            geom_to_iso_fh.close()
            iso_to_geom_fh.close()
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
        "--dataset_type",
        "-dt",
        type=str,
        default="drugs",
        help="Unused placeholder kept for CLI parity.",
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
        help="Legacy bin size for cartesian_binned* embeddings.",
    )
    parser.add_argument(
        "--ranges",
        type=str,
        default="[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
        help="Legacy ranges for cartesian_binned* embeddings.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["energy", "weight", "none"],
        default="none",
        help="Sort conformers by energy, weight, or keep original order.",
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
        help="If set, use isomeric SMILES keys; otherwise use non-isomeric keys when available.",
    )
    parser.add_argument(
        "--use_centered",
        action="store_true",
        default=False,
        help="Load *_data_centered.pickle instead of *_data.pickle.",
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
        sort_by=args.sort_by,
        bin_config_path=args.bin_config_path,
        isomeric=args.isomeric,
        use_centered=args.use_centered,
    )
