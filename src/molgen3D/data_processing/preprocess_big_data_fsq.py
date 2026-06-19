import argparse
import json
import os
import os.path as osp
import random
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger as log
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from molgen3D.data_processing.smiles_encoder_decoder import encode_cartesian_v2
from molgen3D.data_processing.utils import (
    center_conformer,
    encode_mol_with_embedding,
    JsonlSplitWriter,
    copy_single_conformer_mol,
    filter_revisited_mols,
    get_coordinate_ranges_for_embedding,
    get_embedding_func_and_config,
    load_pkl,
    save_processed_pickle,
)

RDLogger.DisableLog("rdApp.*")

_BIG_DATA_ITEMS: Optional[List[Tuple[str, List[Chem.Mol]]]] = None


def _parse_shard_id(value: str) -> int:
    if value is None or str(value).strip() == "":
        raise argparse.ArgumentTypeError(
            "empty shard id; set SLURM_ARRAY_TASK_ID, pass an explicit integer, "
            "or omit --shard_id to process all shards sequentially"
        )
    try:
        shard_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shard id: {value!r}") from exc
    if shard_id < 0:
        raise argparse.ArgumentTypeError("shard id must be non-negative")
    return shard_id


@dataclass
class _FSQMolState:
    geom_smiles: str
    confs_count_pre_filter: int
    failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    samples: List[str] = field(default_factory=list)
    processed_mols: List[Chem.Mol] = field(default_factory=list)
    isomeric_smiles: Set[str] = field(default_factory=set)
    nonisomeric_smiles: Set[str] = field(default_factory=set)
    dotted_smiles: Set[str] = field(default_factory=set)
    pending: int = 0
    closed: bool = False


@dataclass
class _FSQPendingRecord:
    state: _FSQMolState
    enriched_text: str
    canonical_smiles: str
    iso_smile: str
    noniso_smile: Optional[str]
    mol: Optional[Chem.Mol]


def _process_big_data_mol_impl(
    idx: int,
    max_confs: Optional[int],
    precision: int,
    embedding_func: Any,
    bin_size: float,
    parsed_ranges: List[Tuple[float, float]],
    do_filter: bool,
    use_isomeric_smiles: bool,
    bin_config: Any,
    do_center: bool = False,
) -> Tuple[List[str], Dict[str, Any], str, Set[str], List[Chem.Mol]]:
    geom_smiles, mols = _BIG_DATA_ITEMS[idx]

    local_failures: Dict[str, int] = defaultdict(int)
    filtered_input = filter_revisited_mols(
        smiles=geom_smiles,
        mols=mols,
        failures=local_failures,
        max_confs=max_confs,
    )

    nonisomeric_smiles: Set[str] = set()
    dotted_smiles: Set[str] = set()
    isomeric_smiles: Set[str] = set()
    samples: List[str] = []
    processed_mols: List[Chem.Mol] = []

    for mol in filtered_input:
        if do_center:
            try:
                mol = center_conformer(mol)
            except Exception as exc:
                log.error("Error centering conformer | smiles={} | failure={}", geom_smiles, exc)
                local_failures["centering_error"] += 1
                continue

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
            embedded_smile, iso_smile = encode_mol_with_embedding(
                mol,
                embedding_func,
                precision=precision,
                bin_size=bin_size,
                ranges=parsed_ranges,
                bin_config=bin_config,
            )
        except Exception as exc:
            log.error("Error encoding conformer | smiles={} | failure={}", geom_smiles, exc)
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
        processed_mols.append(copy_single_conformer_mol(mol))

    if len(nonisomeric_smiles) > 1:
        log.info(
            "multiple_distinct_nonisomeric_smiles | smiles={} | distinct_smiles={}",
            geom_smiles,
            nonisomeric_smiles,
        )
        for dotted in dotted_smiles:
            log.info("dot_in_conformer_smiles | smiles={} | dotted={}", geom_smiles, dotted)

    if not samples:
        log.warning("No samples after filtering | smiles={}", geom_smiles)
        local_failures["no_samples_after_filtering"] += 1

    stats = {
        "geom_smiles": geom_smiles,
        "confs_count_pre_filter": len(mols),
        "confs_count_post_filter": len(samples),
        "nonisomeric_smiles_post_filter": len(nonisomeric_smiles),
        "num_distinct_smiles_with_dot": len(dotted_smiles),
        "has_dotted_smiles": bool(dotted_smiles),
        "failures": local_failures,
    }
    return samples, stats, geom_smiles, isomeric_smiles, processed_mols


def _prepare_big_data_mol_for_fsq(
    idx: int,
    max_confs: Optional[int],
    precision: int,
    parsed_ranges: List[Tuple[float, float]],
    do_filter: bool,
    use_isomeric_smiles: bool,
    do_center: bool = False,
    keep_mols: bool = True,
    fsq_max_tokens: Optional[int] = None,
    fsq_token_counter: Optional[Any] = None,
) -> Tuple[_FSQMolState, List[_FSQPendingRecord]]:
    geom_smiles, mols = _BIG_DATA_ITEMS[idx]
    state = _FSQMolState(geom_smiles=geom_smiles, confs_count_pre_filter=len(mols))

    filtered_input = filter_revisited_mols(
        smiles=geom_smiles,
        mols=mols,
        failures=state.failures,
        max_confs=max_confs,
    )

    pending: List[_FSQPendingRecord] = []
    for mol in filtered_input:
        if do_center:
            try:
                mol = center_conformer(mol)
            except Exception as exc:
                log.error("Error centering conformer | smiles={} | failure={}", geom_smiles, exc)
                state.failures["centering_error"] += 1
                continue

        if do_filter:
            pos = mol.GetConformer().GetPositions()
            out_of_range = False
            for i in range(3):
                min_val, max_val = parsed_ranges[i]
                if np.any(pos[:, i] < min_val) or np.any(pos[:, i] > max_val):
                    out_of_range = True
                    break
            if out_of_range:
                state.failures["coord_out_of_range"] += 1
                continue

        try:
            enriched_text, iso_smile = encode_cartesian_v2(mol, precision=precision)
        except Exception as exc:
            log.error("Error cartesian-encoding conformer | smiles={} | failure={}", geom_smiles, exc)
            state.failures["encoding_error"] += 1
            continue

        if fsq_max_tokens is not None and fsq_token_counter is not None:
            try:
                n_fsq_tokens = fsq_token_counter(enriched_text)
            except Exception as exc:
                log.error("Error FSQ-tokenizing conformer | smiles={} | failure={}", geom_smiles, exc)
                state.failures["fsq_tokenization_error"] += 1
                continue
            if n_fsq_tokens > fsq_max_tokens:
                state.failures["fsq_too_many_tokens"] += 1
                continue

        noniso = None
        try:
            noniso = Chem.MolToSmiles(
                Chem.RemoveHs(mol, sanitize=False),
                canonical=True,
                isomericSmiles=False,
            )
        except Exception:
            pass

        canonical_smiles = iso_smile if use_isomeric_smiles else (noniso or iso_smile)
        pending.append(
            _FSQPendingRecord(
                state=state,
                enriched_text=enriched_text,
                canonical_smiles=canonical_smiles,
                iso_smile=iso_smile,
                noniso_smile=noniso,
                mol=mol if keep_mols else None,
            )
        )

    state.pending = len(pending)
    state.closed = True
    return state, pending


def _accept_fsq_record(record: _FSQPendingRecord, fsq_text: str) -> None:
    state = record.state
    state.samples.append(
        json.dumps(
            {
                "canonical_smiles": record.canonical_smiles,
                "embedded_smiles": fsq_text,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    state.isomeric_smiles.add(record.iso_smile)
    if record.noniso_smile is not None:
        state.nonisomeric_smiles.add(record.noniso_smile)
        if "." in record.noniso_smile:
            state.dotted_smiles.add(record.noniso_smile)
    if record.mol is not None:
        state.processed_mols.append(copy_single_conformer_mol(record.mol))


def _flush_fsq_records(
    records: List[_FSQPendingRecord],
    fsq_encoder,
) -> None:
    if not records:
        return

    try:
        fsq_texts = fsq_encoder.encode_texts([record.enriched_text for record in records])
    except Exception as batch_exc:
        log.warning(
            "Batched FSQ encoding failed for {} conformers; retrying individually | failure={}",
            len(records),
            batch_exc,
        )
        for record in records:
            try:
                fsq_text = fsq_encoder.encode_texts([record.enriched_text])[0]
                _accept_fsq_record(record, fsq_text)
            except Exception as exc:
                log.error(
                    "Error FSQ-encoding conformer | smiles={} | failure={}",
                    record.state.geom_smiles,
                    exc,
                )
                record.state.failures["fsq_encoding_error"] += 1
            finally:
                record.state.pending -= 1
        return

    for record, fsq_text in zip(records, fsq_texts):
        _accept_fsq_record(record, fsq_text)
        record.state.pending -= 1


def _finalize_fsq_state(state: _FSQMolState) -> Tuple[List[str], Dict[str, Any], str, Set[str], List[Chem.Mol]]:
    if len(state.nonisomeric_smiles) > 1:
        log.info(
            "multiple_distinct_nonisomeric_smiles | smiles={} | distinct_smiles={}",
            state.geom_smiles,
            state.nonisomeric_smiles,
        )
        for dotted in state.dotted_smiles:
            log.info("dot_in_conformer_smiles | smiles={} | dotted={}", state.geom_smiles, dotted)

    if not state.samples:
        log.warning("No samples after filtering | smiles={}", state.geom_smiles)
        state.failures["no_samples_after_filtering"] += 1

    stats = {
        "geom_smiles": state.geom_smiles,
        "confs_count_pre_filter": state.confs_count_pre_filter,
        "confs_count_post_filter": len(state.samples),
        "nonisomeric_smiles_post_filter": len(state.nonisomeric_smiles),
        "num_distinct_smiles_with_dot": len(state.dotted_smiles),
        "has_dotted_smiles": bool(state.dotted_smiles),
        "failures": state.failures,
    }
    return state.samples, stats, state.geom_smiles, state.isomeric_smiles, state.processed_mols


def _process_big_data_by_idx(args: Tuple) -> Optional[Tuple]:
    idx = args[0]
    try:
        return _process_big_data_mol_impl(idx, *args[1:])
    except Exception as exc:
        log.error("Unhandled exception | idx={} | error={}", idx, exc)
        return None



def _collect_results(
    result,
    split_writer: JsonlSplitWriter,
    split_pickle_dir: Optional[str],
    mapping_fh,
    failure_counts: Dict[str, int],
    counters: Dict[str, int],
) -> None:
    if result is None:
        failure_counts["unhandled_exception"] += 1
        return

    samples, stats, geom_smiles, iso_set, processed_mols = result
    if samples:
        split_writer.write(samples)

    counters["total_input_confs"] += stats["confs_count_pre_filter"]
    counters["total_confs"] += stats["confs_count_post_filter"]
    counters["total_dotted_smiles"] += stats.get("num_distinct_smiles_with_dot", 0)

    if stats["nonisomeric_smiles_post_filter"] > 1:
        counters["molecules_with_multiple_distinct_graphs"] += 1
    if stats.get("has_dotted_smiles", False):
        counters["molecules_with_dotted_smiles"] += 1

    for reason, count in stats["failures"].items():
        failure_counts[reason] += int(count)

    if processed_mols and split_pickle_dir:
        save_processed_pickle(split_pickle_dir, geom_smiles, processed_mols)
    if samples:
        counters["total_mols"] += 1

    for iso in sorted(iso_set):
        counters["num_distinct_isomeric_smiles"] += 1
        mapping_fh.write(
            json.dumps(
                {"geom_smiles": geom_smiles, "isomeric_smiles": iso},
                separators=(",", ":"),
            )
            + "\n"
        )


def preprocess_big_data(
    input_path: str,
    embedding_type: str,
    dest_path: str,
    split_name: str = "train",
    num_workers: int = 1,
    precision: int = 4,
    max_confs: Optional[int] = 30,
    bin_size: float = 0.104,
    ranges: str = "[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
    filter_ranges: Optional[str] = None,
    bin_config_path: Optional[str] = None,
    fsq_ckpt_path: Optional[str] = None,
    fsq_device: str = "auto",
    fsq_max_tokens: Optional[int] = 218,
    use_isomeric_smiles: bool = False,
    center: bool = False,
    save_pickles: bool = True,
    batch_size: int = 100_000,
    fsq_batch_size: int = 128,
) -> None:
    global _BIG_DATA_ITEMS

    import gc

    if embedding_type == "fsq" and num_workers > 1:
        fsq_device_text = str(fsq_device).lower()
        if fsq_device_text.startswith("cuda") or fsq_device_text == "auto":
            import torch

            if fsq_device_text.startswith("cuda") or torch.cuda.is_available():
                raise ValueError(
                    "FSQ encoding on CUDA must use --num_workers 1. "
                    "For multiprocessing, set --fsq_device cpu."
                )

    embedding_func, bin_config = get_embedding_func_and_config(
        embedding_type=embedding_type,
        bin_config_path=bin_config_path,
        fsq_ckpt_path=fsq_ckpt_path,
        fsq_device=fsq_device,
        fsq_max_tokens=fsq_max_tokens,
    )
    parsed_ranges = get_coordinate_ranges_for_embedding(ranges, bin_config=bin_config)

    do_filter = False
    if filter_ranges is not None:
        if isinstance(filter_ranges, str):
            do_filter = filter_ranges.lower() in ("true", "1", "yes", "on")
        else:
            do_filter = bool(filter_ranges)

    if embedding_type == "fsq":
        from molgen3D.data_processing.smiles_encoder_decoder_fsq import get_fsq_encoder

        _preprocess_big_data_fsq_streaming(
            input_path=input_path,
            dest_path=dest_path,
            split_name=split_name,
            max_confs=max_confs,
            precision=precision,
            parsed_ranges=parsed_ranges,
            do_filter=do_filter,
            use_isomeric_smiles=use_isomeric_smiles,
            center=center,
            save_pickles=save_pickles,
            batch_size=batch_size,
            fsq_batch_size=fsq_batch_size,
            fsq_encoder=get_fsq_encoder(),
        )
        return

    log.info("Loading grouped big-data pickle from {}", input_path)
    grouped_data = load_pkl(input_path)
    if not isinstance(grouped_data, dict):
        raise TypeError(f"Expected dict[str, list[Mol]] input, got {type(grouped_data)!r}")

    all_keys = sorted(grouped_data.keys())
    total_input_mols = len(all_keys)
    log.info("Loaded {:,} molecules for split '{}'", total_input_mols, split_name)

    strings_root = osp.join(dest_path, "processed_strings")
    split_writer = JsonlSplitWriter(osp.join(strings_root, split_name), split_name)
    split_pickle_dir = osp.join(dest_path, "processed_pickles", split_name) if save_pickles else None
    if split_pickle_dir:
        os.makedirs(split_pickle_dir, exist_ok=True)

    failure_counts: Dict[str, int] = defaultdict(int)
    counters: Dict[str, int] = defaultdict(int)

    mapping_path = osp.join(dest_path, f"{split_name}_geom_to_isomeric_smiles.jsonl")
    n_batches = (total_input_mols + batch_size - 1) // batch_size

    with open(mapping_path, "w") as mapping_fh:
        try:
            with tqdm(total=total_input_mols, dynamic_ncols=True, mininterval=0.2) as pbar:
                for batch_idx in range(n_batches):
                    b_start = batch_idx * batch_size
                    b_end = min(b_start + batch_size, total_input_mols)
                    batch_keys = all_keys[b_start:b_end]

                    _BIG_DATA_ITEMS = [
                        (k, grouped_data.pop(k)) for k in batch_keys
                    ]
                    if batch_idx % 10 == 0:
                        gc.collect()

                    batch_len = len(_BIG_DATA_ITEMS)
                    job_args = [
                        (idx, max_confs, precision, embedding_func, bin_size,
                         parsed_ranges, do_filter, use_isomeric_smiles,
                         bin_config, center)
                        for idx in range(batch_len)
                    ]

                    if num_workers <= 1:
                        for result in map(_process_big_data_by_idx, job_args):
                            _collect_results(
                                result, split_writer, split_pickle_dir,
                                mapping_fh, failure_counts, counters,
                            )
                            pbar.update()
                    else:
                        pool = Pool(processes=num_workers)
                        try:
                            imap_chunk = max(1, batch_len // max(num_workers * 4, 1))
                            for result in pool.imap_unordered(
                                _process_big_data_by_idx, job_args,
                                chunksize=imap_chunk,
                            ):
                                _collect_results(
                                    result, split_writer, split_pickle_dir,
                                    mapping_fh, failure_counts, counters,
                                )
                                pbar.update()
                        finally:
                            pool.close()
                            pool.join()

                    _BIG_DATA_ITEMS = None
                    split_writer.flush()
                    mapping_fh.flush()

                    if (batch_idx & 3) == 0:
                        pbar.refresh()

        finally:
            split_writer.close()
            _BIG_DATA_ITEMS = None

    total_mols = counters["total_mols"]
    total_confs = counters["total_confs"]
    avg_confs_per_mol = total_confs / total_mols if total_mols else 0.0
    success_rate = total_mols / total_input_mols if total_input_mols else 0.0
    run_summary = {
        "split": split_name,
        "grand_total_samples_written": split_writer.total_samples,
        "total_input_molecules": total_input_mols,
        "molecules_after_filter": total_mols,
        "num_input_conformers": counters["total_input_confs"],
        "conformers_after_filter": total_confs,
        "avg_confs_per_mol_after": avg_confs_per_mol,
        "success_rate": success_rate,
        "molecules_with_multiple_distinct_graphs": counters["molecules_with_multiple_distinct_graphs"],
        "molecules_with_dotted_smiles": counters["molecules_with_dotted_smiles"],
        "total_dotted_smiles": counters["total_dotted_smiles"],
        "num_distinct_isomeric_smiles": counters["num_distinct_isomeric_smiles"],
        "overall_failure_counts": dict(failure_counts),
    }
    log.info(json.dumps({"run_summary": run_summary}, ensure_ascii=False, separators=(",", ":")))


def _preprocess_big_data_fsq_streaming(
    input_path: str,
    dest_path: str,
    split_name: str,
    max_confs: Optional[int],
    precision: int,
    parsed_ranges: List[Tuple[float, float]],
    do_filter: bool,
    use_isomeric_smiles: bool,
    center: bool,
    save_pickles: bool,
    batch_size: int,
    fsq_batch_size: int,
    fsq_encoder,
) -> None:
    global _BIG_DATA_ITEMS

    import gc

    fsq_batch_size = max(1, int(fsq_batch_size))
    from molgen3D.data_processing.smiles_encoder_decoder_fsq import tokenize_enriched

    fsq_max_tokens = getattr(fsq_encoder, "max_tokens", None)
    fsq_token_counter = (lambda text: len(tokenize_enriched(text))) if fsq_max_tokens is not None else None
    log.info("Loading grouped big-data pickle from {}", input_path)
    grouped_data = load_pkl(input_path)
    if not isinstance(grouped_data, dict):
        raise TypeError(f"Expected dict[str, list[Mol]] input, got {type(grouped_data)!r}")

    all_keys = sorted(grouped_data.keys())
    total_input_mols = len(all_keys)
    log.info(
        "Loaded {:,} molecules for split '{}' | fsq_batch_size={} | fsq_device={}",
        total_input_mols,
        split_name,
        fsq_batch_size,
        fsq_encoder.device,
    )

    strings_root = osp.join(dest_path, "processed_strings")
    split_writer = JsonlSplitWriter(osp.join(strings_root, split_name), split_name)
    split_pickle_dir = osp.join(dest_path, "processed_pickles", split_name) if save_pickles else None
    if split_pickle_dir:
        os.makedirs(split_pickle_dir, exist_ok=True)

    failure_counts: Dict[str, int] = defaultdict(int)
    counters: Dict[str, int] = defaultdict(int)
    mapping_path = osp.join(dest_path, f"{split_name}_geom_to_isomeric_smiles.jsonl")
    n_batches = (total_input_mols + batch_size - 1) // batch_size

    pending_records: List[_FSQPendingRecord] = []
    open_states: List[_FSQMolState] = []

    def collect_ready(mapping_fh, pbar) -> None:
        ready_idx = 0
        while ready_idx < len(open_states):
            state = open_states[ready_idx]
            if not state.closed or state.pending != 0:
                break
            result = _finalize_fsq_state(state)
            _collect_results(
                result,
                split_writer,
                split_pickle_dir,
                mapping_fh,
                failure_counts,
                counters,
            )
            pbar.update()
            ready_idx += 1
        if ready_idx:
            del open_states[:ready_idx]

    def flush_some(limit: int, mapping_fh, pbar) -> None:
        while len(pending_records) >= limit:
            records = pending_records[:limit]
            del pending_records[:limit]
            _flush_fsq_records(records, fsq_encoder)
            collect_ready(mapping_fh, pbar)

    with open(mapping_path, "w") as mapping_fh:
        try:
            with tqdm(total=total_input_mols, dynamic_ncols=True, mininterval=0.2) as pbar:
                for batch_idx in range(n_batches):
                    b_start = batch_idx * batch_size
                    b_end = min(b_start + batch_size, total_input_mols)
                    batch_keys = all_keys[b_start:b_end]

                    _BIG_DATA_ITEMS = [(key, grouped_data.pop(key)) for key in batch_keys]
                    batch_len = len(_BIG_DATA_ITEMS)

                    for idx in range(batch_len):
                        try:
                            state, records = _prepare_big_data_mol_for_fsq(
                                idx=idx,
                                max_confs=max_confs,
                                precision=precision,
                                parsed_ranges=parsed_ranges,
                                do_filter=do_filter,
                                use_isomeric_smiles=use_isomeric_smiles,
                                do_center=center,
                                keep_mols=save_pickles,
                                fsq_max_tokens=fsq_max_tokens,
                                fsq_token_counter=fsq_token_counter,
                            )
                        except Exception as exc:
                            log.error("Unhandled FSQ prep exception | idx={} | error={}", idx, exc)
                            failure_counts["unhandled_exception"] += 1
                            pbar.update()
                            continue

                        open_states.append(state)
                        pending_records.extend(records)
                        collect_ready(mapping_fh, pbar)
                        flush_some(fsq_batch_size, mapping_fh, pbar)

                    if pending_records:
                        _flush_fsq_records(pending_records, fsq_encoder)
                        pending_records.clear()
                        collect_ready(mapping_fh, pbar)

                    _BIG_DATA_ITEMS = None
                    split_writer.flush()
                    mapping_fh.flush()
                    if batch_idx % 10 == 0:
                        gc.collect()
                    if (batch_idx & 3) == 0:
                        pbar.refresh()
        finally:
            split_writer.close()
            _BIG_DATA_ITEMS = None

    total_mols = counters["total_mols"]
    total_confs = counters["total_confs"]
    run_summary = {
        "split": split_name,
        "grand_total_samples_written": split_writer.total_samples,
        "total_input_molecules": total_input_mols,
        "molecules_after_filter": total_mols,
        "num_input_conformers": counters["total_input_confs"],
        "conformers_after_filter": total_confs,
        "avg_confs_per_mol_after": total_confs / total_mols if total_mols else 0.0,
        "success_rate": total_mols / total_input_mols if total_input_mols else 0.0,
        "molecules_with_multiple_distinct_graphs": counters["molecules_with_multiple_distinct_graphs"],
        "molecules_with_dotted_smiles": counters["molecules_with_dotted_smiles"],
        "total_dotted_smiles": counters["total_dotted_smiles"],
        "num_distinct_isomeric_smiles": counters["num_distinct_isomeric_smiles"],
        "overall_failure_counts": dict(failure_counts),
        "fsq_batch_size": fsq_batch_size,
        "fsq_device": str(fsq_encoder.device),
    }
    log.info(json.dumps({"run_summary": run_summary}, ensure_ascii=False, separators=(",", ":")))


def preprocess_big_data_shards(
    input_dir: str,
    embedding_type: str,
    dest_path: str,
    split_name: str = "train",
    shard_id: Optional[int] = None,
    num_workers: int = 1,
    precision: int = 4,
    max_confs: Optional[int] = 30,
    bin_size: float = 0.104,
    ranges: str = "[-13.0, 13.0], [-13.0, 13.0], [-13.0, 13.0]",
    filter_ranges: Optional[str] = None,
    bin_config_path: Optional[str] = None,
    fsq_ckpt_path: Optional[str] = None,
    fsq_device: str = "auto",
    fsq_max_tokens: Optional[int] = 218,
    use_isomeric_smiles: bool = False,
    center: bool = False,
    save_pickles: bool = True,
    batch_size: int = 100_000,
    fsq_batch_size: int = 128,
) -> None:
    """Process shard_NNNN.pkl files one at a time to bound peak RAM.

    With shard_id set, outputs go to dest_path/shard_NNNN/ so parallel
    SLURM array tasks don't collide; run combine_shard_outputs() afterwards.
    """
    import glob as _glob

    shard_paths = sorted(_glob.glob(osp.join(input_dir, "shard_*.pkl")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pkl files found in {input_dir}")

    if shard_id is not None:
        target_name = f"shard_{shard_id:04d}.pkl"
        matched = [p for p in shard_paths if osp.basename(p) == target_name]
        if not matched:
            raise FileNotFoundError(
                f"Shard {target_name} not found in {input_dir}. "
                f"Available: {[osp.basename(p) for p in shard_paths]}"
            )
        shard_paths = matched

    log.info(
        "Processing {} shard(s) from {} → split='{}'",
        len(shard_paths),
        input_dir,
        split_name,
    )

    for shard_path in shard_paths:
        shard_basename = osp.splitext(osp.basename(shard_path))[0]
        log.info("--- shard: {}", shard_basename)

        if shard_id is not None:
            shard_dest = osp.join(dest_path, shard_basename)
        else:
            shard_dest = dest_path
        os.makedirs(shard_dest, exist_ok=True)

        preprocess_big_data(
            input_path=shard_path,
            embedding_type=embedding_type,
            dest_path=shard_dest,
            split_name=split_name,
            num_workers=num_workers,
            precision=precision,
            max_confs=max_confs,
            bin_size=bin_size,
            ranges=ranges,
            filter_ranges=filter_ranges,
            bin_config_path=bin_config_path,
            fsq_ckpt_path=fsq_ckpt_path,
            fsq_device=fsq_device,
            fsq_max_tokens=fsq_max_tokens,
            use_isomeric_smiles=use_isomeric_smiles,
            center=center,
            save_pickles=save_pickles,
            batch_size=batch_size,
            fsq_batch_size=fsq_batch_size,
        )

    log.info("All shards done.")


def combine_shard_outputs(
    dest_path: str,
    split_name: str = "train",
    delete_shard_dirs: bool = False,
) -> None:
    """Merge shard_NNNN/ subdirs into dest_path/processed_strings, processed_pickles, and a combined mapping JSONL."""
    import glob as _glob
    import shutil

    shard_dirs = sorted(_glob.glob(osp.join(dest_path, "shard_*")))
    shard_dirs = [d for d in shard_dirs if osp.isdir(d)]
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found in {dest_path}")

    log.info("Combining {} shard directories under {}", len(shard_dirs), dest_path)

    combined_strings_dir = osp.join(dest_path, "processed_strings", split_name)
    combined_pickles_dir = osp.join(dest_path, "processed_pickles", split_name)
    os.makedirs(combined_strings_dir, exist_ok=True)
    os.makedirs(combined_pickles_dir, exist_ok=True)

    combined_mapping_path = osp.join(dest_path, f"{split_name}_geom_to_isomeric_smiles.jsonl")

    total_jsonl_files = 0
    total_pickle_files = 0
    total_mapping_lines = 0
    aggregate_summary: Dict[str, Any] = defaultdict(int)

    with open(combined_mapping_path, "w") as combined_mapping_fh:
        for shard_dir in shard_dirs:
            shard_name = osp.basename(shard_dir)
            log.info("  merging {}", shard_name)

            shard_strings = osp.join(shard_dir, "processed_strings", split_name)
            if osp.isdir(shard_strings):
                for jsonl_file in sorted(_glob.glob(osp.join(shard_strings, "*.jsonl"))):
                    dest_name = f"{shard_name}_{osp.basename(jsonl_file)}"
                    shutil.move(jsonl_file, osp.join(combined_strings_dir, dest_name))
                    total_jsonl_files += 1

            shard_pickles = osp.join(shard_dir, "processed_pickles", split_name)
            if osp.isdir(shard_pickles):
                for pkl_file in sorted(_glob.glob(osp.join(shard_pickles, "*.pickle"))):
                    dest_name = osp.basename(pkl_file)
                    dest_pkl = osp.join(combined_pickles_dir, dest_name)
                    if osp.exists(dest_pkl):
                        dest_name = f"{shard_name}_{dest_name}"
                        dest_pkl = osp.join(combined_pickles_dir, dest_name)
                    shutil.move(pkl_file, dest_pkl)
                    total_pickle_files += 1

            shard_mapping = osp.join(shard_dir, f"{split_name}_geom_to_isomeric_smiles.jsonl")
            if osp.isfile(shard_mapping):
                with open(shard_mapping, "r") as fh:
                    for line in fh:
                        combined_mapping_fh.write(line)
                        total_mapping_lines += 1

            shard_log_pattern = osp.join(shard_dir, "preprocessing*.log")
            for log_file in _glob.glob(shard_log_pattern):
                with open(log_file) as fh:
                    for line in fh:
                        if '"run_summary"' in line:
                            try:
                                payload = json.loads(line.split(" | ", 1)[-1] if " | " in line else line)
                                summary = payload.get("run_summary", {})
                                for key in (
                                    "grand_total_samples_written",
                                    "total_input_molecules",
                                    "molecules_after_filter",
                                    "num_input_conformers",
                                    "conformers_after_filter",
                                    "molecules_with_multiple_distinct_graphs",
                                    "molecules_with_dotted_smiles",
                                    "total_dotted_smiles",
                                ):
                                    aggregate_summary[key] += summary.get(key, 0)
                                for reason, count in summary.get("overall_failure_counts", {}).items():
                                    aggregate_summary[f"failure_{reason}"] += count
                            except (json.JSONDecodeError, ValueError):
                                pass

            if delete_shard_dirs:
                shutil.rmtree(shard_dir)

    total_mols = aggregate_summary.get("molecules_after_filter", 0)
    total_confs = aggregate_summary.get("conformers_after_filter", 0)
    combined_report = {
        "combined_jsonl_files": total_jsonl_files,
        "combined_pickle_files": total_pickle_files,
        "combined_mapping_lines": total_mapping_lines,
        **dict(aggregate_summary),
        "avg_confs_per_mol_after": total_confs / max(1, total_mols),
    }
    log.info(json.dumps({"combined_summary": combined_report}, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess big-data grouped conformer pkl files into enriched JSONL "
            "training data.  Accepts either a single pkl (--input_path), a "
            "directory of shard_NNNN.pkl files (--input_dir), or --combine to "
            "merge per-shard outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to a single grouped pickle {smiles: [Mol, ...]}.",
    )
    src_group.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing shard_NNNN.pkl files.",
    )
    src_group.add_argument(
        "--combine",
        action="store_true",
        default=False,
        help=(
            "Combine mode: merge per-shard output directories (shard_NNNN/) "
            "under --dest/--run_name into a single unified output.  "
            "Run this after all array jobs finish."
        ),
    )

    parser.add_argument(
        "--shard_id",
        type=_parse_shard_id,
        default=None,
        help=(
            "Only valid with --input_dir.  Process only shard N (0-based, "
            "matches shard_NNNN.pkl).  Omit to process all shards sequentially."
        ),
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
            "fsq",
        ],
        default="cartesian_v2",
        help="Embedding type to use for enrichment.",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=1,
        help="Number of worker processes per shard.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Numeric precision for encoded coordinates.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        choices=["train", "valid", "test"],
        default="train",
        help="Logical output split name.",
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
        help="Maximum number of conformers per molecule. Use 0 or a negative value for all.",
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
        help="Set to true/1/yes/on to filter conformers outside the specified ranges.",
    )
    parser.add_argument(
        "--bin_config_path",
        type=str,
        default=None,
        help="Path to BinConfig JSON (required for uniform_binned / quantile_binned).",
    )
    parser.add_argument(
        "--fsq_ckpt_path",
        type=str,
        default=None,
        help=(
            "Path to the trained FSQ coordinate checkpoint. If omitted, uses "
            "FSQ_CKPT_PATH/VQ_CKPT_PATH or the old baseline checkpoint path."
        ),
    )
    parser.add_argument(
        "--fsq_device",
        type=str,
        default="auto",
        help="Device for FSQ coordinate encoding: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--fsq_max_tokens",
        type=int,
        default=218,
        help=(
            "Maximum enriched-token length accepted by the FSQ coordinate encoder. "
            "Use a non-positive value to disable this check."
        ),
    )
    parser.add_argument(
        "--fsq_batch_size",
        type=int,
        default=128,
        help="Number of conformers per batched FSQ GPU forward pass.",
    )
    parser.add_argument(
        "--isomeric",
        action="store_true",
        help="Use isomeric SMILES keys in output.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        help=(
            "Translate each conformer so its centroid is at the origin before "
            "encoding.  Recommended when coordinates are not already centered."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100_000,
        help=(
            "Number of molecules to process per batch.  Controls peak memory: "
            "only one batch is held in the worker-visible global at a time."
        ),
    )
    parser.add_argument(
        "--no_pickles",
        action="store_true",
        default=False,
        help="Skip saving per-molecule processed pickle files (saves disk space and I/O).",
    )
    parser.add_argument(
        "--delete_shard_dirs",
        action="store_true",
        default=False,
        help="When using --combine, delete per-shard directories after merging.",
    )
    args = parser.parse_args()

    if args.shard_id is not None and args.input_dir is None:
        parser.error("--shard_id requires --input_dir")

    random.seed(42)
    dest_path = osp.join(args.dest, args.run_name)
    os.makedirs(dest_path, exist_ok=True)
    enqueue_logs = os.environ.get("LOGURU_ENQUEUE", "1") not in {"0", "false", "False"}

    if args.combine:
        log.add(
            osp.join(dest_path, "combine.log"),
            mode="w",
            enqueue=enqueue_logs,
            backtrace=False,
            diagnose=False,
        )
        combine_shard_outputs(
            dest_path=dest_path,
            split_name=args.split_name,
            delete_shard_dirs=args.delete_shard_dirs,
        )
    else:
        log_suffix = f"_shard{args.shard_id:04d}" if args.shard_id is not None else ""
        log.add(
            osp.join(dest_path, f"preprocessing{log_suffix}.log"),
            mode="w",
            enqueue=enqueue_logs,
            backtrace=False,
            diagnose=False,
        )

        common_kwargs = dict(
            embedding_type=args.embedding_type,
            dest_path=dest_path,
            split_name=args.split_name,
            num_workers=args.num_workers,
            precision=args.precision,
            max_confs=args.max_confs if args.max_confs > 0 else None,
            bin_size=args.bin_size,
            ranges=args.ranges,
            filter_ranges=args.filter_ranges,
            bin_config_path=args.bin_config_path,
            fsq_ckpt_path=args.fsq_ckpt_path,
            fsq_device=args.fsq_device,
            fsq_max_tokens=args.fsq_max_tokens if args.fsq_max_tokens > 0 else None,
            fsq_batch_size=args.fsq_batch_size,
            use_isomeric_smiles=args.isomeric,
            center=args.center,
            save_pickles=not args.no_pickles,
            batch_size=args.batch_size,
        )

        if args.input_path is not None:
            preprocess_big_data(input_path=args.input_path, **common_kwargs)
        else:
            preprocess_big_data_shards(
                input_dir=args.input_dir,
                shard_id=args.shard_id,
                **common_kwargs,
            )
