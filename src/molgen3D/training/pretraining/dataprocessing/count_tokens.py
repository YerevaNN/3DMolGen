#!/usr/bin/env python3
"""
Minimal dataset token counting script for MolGen3D.

The script samples a configurable number of JSONL entries from one file per dataset
using the production dataloader, reports how many batches and tokens that sample produced,
extrapolates totals for the whole split, and verifies the extrapolation using file sizes.
"""

import argparse
import json
import math
import hashlib
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from transformers import AutoTokenizer

from molgen3D.config.paths import get_data_path, get_tokenizer_path
from molgen3D.training.pretraining.dataprocessing.dataloader import build_dataloader

try:  # optional faster JSON parser
    import orjson as _fast_json
except Exception:  # pragma: no cover
    _fast_json = None

SUMMARY_PATH = Path(__file__).resolve().parent / "qwen_data_summary.txt"


def list_jsonl_files(directory: str) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.jsonl"))


def _json_loads(raw):
    if _fast_json is not None:
        return _fast_json.loads(raw)
    return json.loads(raw)


def _iter_isomer_units(file_path: Path):
    try:
        with file_path.open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json_loads(line)
                except Exception:
                    continue
                isomers = obj.get("isomers") or {}
                if not isinstance(isomers, dict):
                    continue
                for iso_smiles, conf_list in isomers.items():
                    if not conf_list:
                        continue
                    embedded_strings = []
                    for conf in conf_list:
                        embedded = (conf.get("embedded_smiles") or "").strip()
                        if embedded:
                            embedded_strings.append(embedded)
                    iso_smiles = (iso_smiles or "").strip()
                    if not iso_smiles or not embedded_strings:
                        continue
                    yield iso_smiles, embedded_strings
    except OSError:
        return


def _build_isomer_unit_text(iso_smiles: str, confs: List[str]) -> str:
    smiles_str = f"[SMILES]{iso_smiles}[/SMILES]"
    conf_strs = [f"[CONFORMER]{conf}[/CONFORMER]" for conf in confs]
    return smiles_str + "".join(conf_strs)


def _serialize_isomer_unit(iso_smiles: str, confs: List[str], tokenizer, ctx_len: int) -> int:
    full_str = _build_isomer_unit_text(iso_smiles, confs)
    token_ids = tokenizer.encode(full_str, add_special_tokens=False)
    return min(len(token_ids), ctx_len)


_SMILES_OPEN = "[SMILES]"
_SMILES_CLOSE = "[/SMILES]"
_CONF_OPEN = "[CONFORMER]"
_CONF_CLOSE = "[/CONFORMER]"
_TAG_OVERHEAD = len(_SMILES_OPEN) + len(_SMILES_CLOSE) + len(_CONF_OPEN) + len(_CONF_CLOSE)


def _unit_char_len(iso_smiles: str, confs: List[str]) -> int:
    total = len(_SMILES_OPEN) + len(iso_smiles) + len(_SMILES_CLOSE)
    total += sum(len(_CONF_OPEN) + len(conf) + len(_CONF_CLOSE) for conf in confs)
    return total


def _segment_char_lens(iso_smiles: str, confs: List[str]) -> Tuple[int, List[int]]:
    smiles_len = len(_SMILES_OPEN) + len(iso_smiles) + len(_SMILES_CLOSE)
    conf_lens = [
        len(_CONF_OPEN) + len(conf) + len(_CONF_CLOSE) for conf in confs
    ]
    return smiles_len, conf_lens


def _pack_isomer_unit_len(
    used: int, unit_len: int, ctx_len: int
) -> Tuple[int, int, int]:
    sequences = 0
    pad_total = 0
    if used == 0:
        used = unit_len
        if used >= ctx_len:
            sequences += 1
            used = 0
        return used, sequences, pad_total

    needed = 1 + unit_len
    if used + needed <= ctx_len:
        used += needed
        if used >= ctx_len:
            sequences += 1
            used = 0
        return used, sequences, pad_total

    pad_total += ctx_len - used
    sequences += 1
    used = unit_len
    if used >= ctx_len:
        sequences += 1
        used = 0
    return used, sequences, pad_total


def _encode_segmented_unit(
    iso_smiles: str, confs: List[str], tokenizer
) -> Tuple[List[int], List[List[int]]]:
    smiles_tokens = tokenizer.encode(
        f"{_SMILES_OPEN}{iso_smiles}{_SMILES_CLOSE}", add_special_tokens=False
    )
    conf_tokens = [
        tokenizer.encode(f"{_CONF_OPEN}{conf}{_CONF_CLOSE}", add_special_tokens=False)
        for conf in confs
    ]
    return smiles_tokens, conf_tokens


def _chunk_segmented_unit(
    smiles_tokens: List[int], conf_token_lists: List[List[int]], ctx_len: int
) -> Tuple[List[int], int, int]:
    """
    Returns: chunk_lengths, confs_emitted, confs_dropped
    Mirrors dataloader chunking semantics with SMILES repetition.
    """
    if ctx_len <= 0 or not smiles_tokens:
        return [], 0, len(conf_token_lists)
    if len(smiles_tokens) >= ctx_len:
        return [ctx_len], 0, len(conf_token_lists)

    chunks: List[int] = []
    confs_emitted = 0
    confs_dropped = 0
    current_len = len(smiles_tokens)
    available = ctx_len - current_len

    for idx, conf_tokens in enumerate(conf_token_lists):
        if not conf_tokens:
            continue
        conf_len = len(conf_tokens)
        if conf_len > available:
            if current_len == len(smiles_tokens):
                # Truncate conformer, emit chunk, drop remaining conformers.
                chunks.append(ctx_len)
                confs_emitted += 1
                remaining = len(conf_token_lists) - (idx + 1)
                confs_dropped += remaining
                return chunks, confs_emitted, confs_dropped
            # Flush current chunk and retry with fresh SMILES.
            chunks.append(current_len)
            current_len = len(smiles_tokens)
            available = ctx_len - current_len
            if conf_len > available:
                chunks.append(ctx_len)
                confs_emitted += 1
                remaining = len(conf_token_lists) - (idx + 1)
                confs_dropped += remaining
                return chunks, confs_emitted, confs_dropped
        # Conf fits
        current_len += conf_len
        available -= conf_len
        confs_emitted += 1

    if current_len:
        chunks.append(current_len)
    return chunks, confs_emitted, confs_dropped


def _chunk_estimated_lengths(
    smiles_len: int, conf_lens: List[int], ctx_len: int, token_per_char: float
) -> Tuple[List[int], int, int]:
    if ctx_len <= 0:
        return [], 0, len(conf_lens)
    smiles_tokens = max(1, int(round(smiles_len * token_per_char)))
    smiles_tokens = min(smiles_tokens, ctx_len)
    if smiles_tokens >= ctx_len:
        return [ctx_len], 0, len(conf_lens)

    conf_tokens = [max(1, int(round(c * token_per_char))) for c in conf_lens]
    chunks: List[int] = []
    confs_emitted = 0
    confs_dropped = 0
    current_len = smiles_tokens
    available = ctx_len - current_len

    for idx, conf_len in enumerate(conf_tokens):
        if conf_len > available:
            if current_len == smiles_tokens:
                chunks.append(ctx_len)
                confs_emitted += 1
                remaining = len(conf_tokens) - (idx + 1)
                confs_dropped += remaining
                return chunks, confs_emitted, confs_dropped
            chunks.append(current_len)
            current_len = smiles_tokens
            available = ctx_len - current_len
            if conf_len > available:
                chunks.append(ctx_len)
                confs_emitted += 1
                remaining = len(conf_tokens) - (idx + 1)
                confs_dropped += remaining
                return chunks, confs_emitted, confs_dropped
        current_len += conf_len
        available -= conf_len
        confs_emitted += 1

    if current_len:
        chunks.append(current_len)
    return chunks, confs_emitted, confs_dropped


def _pack_chunk_len_stream(
    used: int, chunk_len: int, ctx_len: int
) -> Tuple[int, int, int, int]:
    sequences_total = 0
    pad_end_total = 0
    pad_delim_total = 0
    loss_tokens_total = 0

    if used == 0:
        used = chunk_len
        if used >= ctx_len:
            sequences_total += 1
            loss_tokens_total += max(ctx_len - 1, 0)
            used = 0
        return used, sequences_total, pad_end_total, pad_delim_total, loss_tokens_total

    needed = 1 + chunk_len
    if used + needed <= ctx_len:
        pad_delim_total += 1
        used += needed
        if used >= ctx_len:
            sequences_total += 1
            loss_tokens_total += max(ctx_len - 1, 0)
            used = 0
        return used, sequences_total, pad_end_total, pad_delim_total, loss_tokens_total

    end_pad = ctx_len - used
    pad_end_total += end_pad
    sequences_total += 1
    loss_tokens_total += max((ctx_len - 1) - end_pad, 0)
    used = chunk_len
    if used >= ctx_len:
        sequences_total += 1
        loss_tokens_total += max(ctx_len - 1, 0)
        used = 0
    return used, sequences_total, pad_end_total, pad_delim_total, loss_tokens_total


def _pack_chunks(
    chunk_lengths: List[int], ctx_len: int
) -> Tuple[int, int, int, int]:
    """
    Returns sequences_total, pad_end_total, pad_delim_total, loss_tokens_total.
    """
    sequences_total = 0
    pad_end_total = 0
    pad_delim_total = 0
    loss_tokens_total = 0
    used = 0

    for length in chunk_lengths:
        if used == 0:
            used = length
            if used >= ctx_len:
                sequences_total += 1
                loss_tokens_total += max(ctx_len - 1, 0)
                used = 0
            continue

        needed = 1 + length
        if used + needed <= ctx_len:
            pad_delim_total += 1
            used += needed
            if used >= ctx_len:
                sequences_total += 1
                loss_tokens_total += max(ctx_len - 1, 0)
                used = 0
            continue

        # End-pad and flush
        end_pad = ctx_len - used
        pad_end_total += end_pad
        sequences_total += 1
        loss_tokens_total += max((ctx_len - 1) - end_pad, 0)
        used = length
        if used >= ctx_len:
            sequences_total += 1
            loss_tokens_total += max(ctx_len - 1, 0)
            used = 0

    if used > 0:
        end_pad = ctx_len - used
        pad_end_total += end_pad
        sequences_total += 1
        loss_tokens_total += max((ctx_len - 1) - end_pad, 0)

    return sequences_total, pad_end_total, pad_delim_total, loss_tokens_total


def _scan_isomer_stats(
    files: List[Path], sample_units: int, seed: int
) -> Tuple[int, int, Set[str], List[Tuple[str, List[str]]]]:
    rng = random.Random(seed)
    raw_units = 0
    raw_confs = 0
    unique_smiles: Set[str] = set()
    samples: List[Tuple[str, List[str]]] = []

    for file_path in files:
        for iso_smiles, confs in _iter_isomer_units(file_path):
            raw_units += 1
            raw_confs += len(confs)
            unique_smiles.add(iso_smiles)
            if sample_units <= 0:
                continue
            if len(samples) < sample_units:
                samples.append((iso_smiles, confs))
            else:
                j = rng.randint(0, raw_units - 1)
                if j < sample_units:
                    samples[j] = (iso_smiles, confs)

    return raw_units, raw_confs, unique_smiles, samples


def _sample_chunk_stats(
    sampled_units: List[Tuple[str, List[str]]],
    tokenizer,
    ctx_len: int,
) -> Tuple[List[int], float, float]:
    chunk_lengths: List[int] = []
    confs_emitted = 0
    confs_dropped = 0
    units = 0
    for iso_smiles, confs in sampled_units:
        units += 1
        smiles_tokens, conf_tokens = _encode_segmented_unit(
            iso_smiles, confs, tokenizer
        )
        chunks, emitted, dropped = _chunk_segmented_unit(
            smiles_tokens, conf_tokens, ctx_len
        )
        chunk_lengths.extend(chunks)
        confs_emitted += emitted
        confs_dropped += dropped
    if units == 0:
        return [], 0.0, 0.0
    avg_confs_emitted = confs_emitted / units
    avg_confs_dropped = confs_dropped / units
    return chunk_lengths, avg_confs_emitted, avg_confs_dropped


def _simulate_packing_from_chunks(
    chunk_lengths: List[int],
    ctx_len: int,
    target_chunks: int,
) -> Tuple[int, int, int, int, int]:
    if not chunk_lengths or target_chunks <= 0:
        return 0, 0, 0, 0, 0
    used = 0
    sequences_total = 0
    pad_end_total = 0
    pad_delim_total = 0
    loss_tokens_total = 0
    processed = 0
    idx = 0
    n = len(chunk_lengths)
    while processed < target_chunks:
        chunk_len = chunk_lengths[idx]
        idx = (idx + 1) % n
        (
            used,
            added_sequences,
            added_pad,
            added_delim,
            added_loss,
        ) = _pack_chunk_len_stream(used, chunk_len, ctx_len)
        sequences_total += added_sequences
        pad_end_total += added_pad
        pad_delim_total += added_delim
        loss_tokens_total += added_loss
        processed += 1
    if used > 0:
        end_pad = ctx_len - used
        pad_end_total += end_pad
        sequences_total += 1
        loss_tokens_total += max((ctx_len - 1) - end_pad, 0)
    return sequences_total, pad_end_total, pad_delim_total, loss_tokens_total, processed


def _exact_isomer_scan(
    files: List[Path], tokenizer, ctx_len: int
) -> Dict[str, int]:
    raw_units = 0
    raw_confs = 0
    unique_smiles: Set[str] = set()
    chunk_lengths: List[int] = []
    confs_emitted = 0
    confs_dropped = 0

    for file_path in files:
        for iso_smiles, confs in _iter_isomer_units(file_path):
            raw_units += 1
            raw_confs += len(confs)
            unique_smiles.add(iso_smiles)
            smiles_tokens, conf_tokens = _encode_segmented_unit(
                iso_smiles, confs, tokenizer
            )
            chunks, emitted, dropped = _chunk_segmented_unit(
                smiles_tokens, conf_tokens, ctx_len
            )
            chunk_lengths.extend(chunks)
            confs_emitted += emitted
            confs_dropped += dropped

    sequences_total, pad_end_total, pad_delim_total, loss_tokens_total = _pack_chunks(
        chunk_lengths, ctx_len
    )
    tokens_total = sequences_total * ctx_len
    attended_tokens_total = tokens_total - pad_end_total

    return {
        "raw_units": raw_units,
        "raw_confs": raw_confs,
        "unique_smiles_tags": len(unique_smiles),
        "chunks_total": len(chunk_lengths),
        "confs_emitted": confs_emitted,
        "confs_dropped_oversize": confs_dropped,
        "sequences_total": sequences_total,
        "pad_end_total": pad_end_total,
        "pad_delim_total": pad_delim_total,
        "tokens_total": tokens_total,
        "attended_tokens_total": attended_tokens_total,
        "loss_tokens_total": loss_tokens_total,
        "exact_mode": True,
    }


def _iter_isomer_unit_texts(
    files: Iterable[Path], limit: int
) -> List[str]:
    samples: List[str] = []
    if limit <= 0:
        return samples
    for file_path in files:
        for iso_smiles, confs in _iter_isomer_units(file_path):
            samples.append(_build_isomer_unit_text(iso_smiles, confs))
            if len(samples) >= limit:
                return samples
    return samples


def _estimate_tokens_per_char(
    sample_texts: List[str], tokenizer, batch_size: int, ctx_len: int
) -> float:
    if not sample_texts:
        return 1.0
    total_chars = 0
    total_tokens = 0
    for start in range(0, len(sample_texts), batch_size):
        batch = sample_texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=True,
            max_length=ctx_len,
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        for text, ids in zip(batch, input_ids):
            total_chars += max(len(text), 1)
            total_tokens += max(len(ids), 1)
    if total_chars == 0:
        return 1.0
    return total_tokens / total_chars


def _count_tags_in_text(text: str) -> Tuple[int, int]:
    return text.count(_SMILES_OPEN), text.count(_CONF_OPEN)


def _extract_smiles_from_text(text: str) -> List[str]:
    smiles_list: List[str] = []
    idx = 0
    while True:
        start = text.find(_SMILES_OPEN, idx)
        if start == -1:
            break
        start += len(_SMILES_OPEN)
        end = text.find(_SMILES_CLOSE, start)
        if end == -1:
            break
        smiles = text[start:end]
        if smiles:
            smiles_list.append(smiles)
        idx = end + len(_SMILES_CLOSE)
    return smiles_list


def _estimate_units_per_line(
    files: List[Path],
    max_lines: int,
) -> float:
    if max_lines <= 0:
        return 1.0
    units = 0
    lines = 0
    for file_path in files:
        try:
            with file_path.open("rb") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json_loads(line)
                    except Exception:
                        continue
                    isomers = obj.get("isomers") or {}
                    if not isinstance(isomers, dict):
                        continue
                    line_units = 0
                    for _, conf_list in isomers.items():
                        if conf_list:
                            line_units += 1
                    units += line_units
                    lines += 1
                    if lines >= max_lines:
                        break
        except OSError:
            continue
        if lines >= max_lines:
            break
    if lines == 0:
        return 1.0
    return units / lines


def _sample_isomer_dataloader(
    files: List[Path],
    tokenizer_path: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    max_samples: int,
) -> Dict[str, float]:
    loader = build_dataloader(
        train_path=[str(p) for p in files],
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle_lines=shuffle,
        infinite=False,
        seed=seed,
        min_emb_len=0,
        drop_last=False,
        persistent_workers=False,
        world_size=1,
        rank=0,
        serialization_mode="isomer_units",
        emit_attention_mask=True,
    )
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return {}
    _ensure_dataset_tokenizer(dataset)

    samples = 0
    pad_total = 0
    effective_tokens = 0
    smiles_tags = 0
    conformer_tags = 0
    unique_smiles: Set[str] = set()

    for batch in loader:
        inputs = _extract_inputs(batch)
        attention_mask = _extract_attention_mask(batch)
        if attention_mask is None:
            raise RuntimeError(
                "isomer_units mode requires attention_mask to count tokens correctly."
            )
        bsz = inputs.size(0)
        for idx in range(bsz):
            mask = attention_mask[idx]
            sample = inputs[idx]
            content_tokens = [
                tok for tok, attn in zip(sample.tolist(), mask.tolist()) if attn == 1
            ]
            text = tokenizer.decode(content_tokens, skip_special_tokens=False)
            smi, conf = _count_tags_in_text(text)
            smiles_tags += smi
            conformer_tags += conf
            unique_smiles.update(_extract_smiles_from_text(text))
            pad_total += int((mask == 0).sum().item())
            effective_tokens += int(mask.sum().item())
            samples += 1
            if samples >= max_samples:
                break
        if samples >= max_samples:
            break

    return {
        "samples": float(samples),
        "avg_pad_per_sample": (pad_total / samples) if samples else 0.0,
        "avg_effective_tokens_per_sample": (effective_tokens / samples)
        if samples
        else 0.0,
        "avg_smiles_tags_per_sample": (smiles_tags / samples) if samples else 0.0,
        "avg_conformer_tags_per_sample": (conformer_tags / samples) if samples else 0.0,
        "unique_smiles": float(len(unique_smiles)),
        "total_smiles_tags": float(smiles_tags),
    }


def _ensure_dataset_tokenizer(dataset) -> None:
    """Ensure the dataset has initialized its tokenizer so pad/sep IDs exist."""
    if dataset is None:
        return
    if hasattr(dataset, "_ensure_tokenizer_ready"):
        dataset._ensure_tokenizer_ready()
    elif hasattr(dataset, "tk"):
        _ = dataset.tk


def _extract_inputs(batch):
    """
    Normalize dataloader output into a tensor of input_ids.

    Supports:
      - inputs
      - (inputs, target)
      - {"input": tensor, ...}
    """
    if isinstance(batch, (tuple, list)):
        inputs = batch[0]
    else:
        inputs = batch
    return inputs["input"] if isinstance(inputs, dict) else inputs


def _extract_attention_mask(batch):
    if isinstance(batch, (tuple, list)):
        inputs = batch[0]
    else:
        inputs = batch
    if isinstance(inputs, dict):
        return inputs.get("attention_mask")
    return None


def _count_tagged_spans(
    tokens: List[int], tokenizer
) -> Tuple[int, int, int, int]:
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    smiles_open = "[SMILES]"
    smiles_close = "[/SMILES]"
    conf_open = "[CONFORMER]"
    conf_close = "[/CONFORMER]"

    smiles_tokens = 0
    conf_tokens = 0
    smiles_tags = 0
    conf_tags = 0
    idx = 0

    while idx < len(text):
        next_smiles = text.find(smiles_open, idx)
        next_conf = text.find(conf_open, idx)
        if next_smiles == -1 and next_conf == -1:
            break

        if next_smiles != -1 and (next_conf == -1 or next_smiles < next_conf):
            smiles_tags += 1
            start = next_smiles + len(smiles_open)
            end = text.find(smiles_close, start)
            if end == -1:
                idx = start
                continue
            segment = text[start:end]
            smiles_tokens += len(tokenizer.encode(segment, add_special_tokens=False))
            idx = end + len(smiles_close)
            continue

        conf_tags += 1
        start = next_conf + len(conf_open)
        end = text.find(conf_close, start)
        if end == -1:
            idx = start
            continue
        segment = text[start:end]
        conf_tokens += len(tokenizer.encode(segment, add_special_tokens=False))
        idx = end + len(conf_close)

    return smiles_tokens, conf_tokens, smiles_tags, conf_tags


def count_lines_and_bytes(files: List[Path]) -> Tuple[int, int, List[Dict[str, Any]]]:
    total_lines = 0
    total_bytes = 0
    file_stats: List[Dict[str, Any]] = []
    for file in files:
        try:
            with file.open("rb") as fh:
                line_count = sum(1 for _ in fh)
            byte_count = file.stat().st_size
            total_lines += line_count
            total_bytes += byte_count
            file_stats.append({"path": str(file), "lines": line_count, "bytes": byte_count})
        except OSError:
            continue
    return total_lines, total_bytes, file_stats


def bytes_for_lines(file_path: Path, lines: int) -> Optional[int]:
    if not file_path.is_file() or lines <= 0:
        return None
    try:
        with file_path.open("rb") as fh:
            total = 0
            for i, chunk in enumerate(fh, 1):
                total += len(chunk)
                if i >= lines:
                    break
        return total
    except OSError:
        return None


def sample_dataloader(
    file_path: Path,
    tokenizer_path: str,
    tokenizer,
    seq_len: int,
    target_lines: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    serialization_mode: str,
) -> Optional[Dict[str, float]]:
    loader = build_dataloader(
        train_path=str(file_path),
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle_lines=shuffle,
        infinite=False,
        seed=seed,
        min_emb_len=0,
        drop_last=False,
        persistent_workers=False,
        world_size=1,
        rank=0,
        serialization_mode=serialization_mode,
        emit_attention_mask=(serialization_mode == "isomer_units"),
    )

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None

    _ensure_dataset_tokenizer(dataset)

    pad_id = getattr(dataset, "pad_id", None)
    sep_id = getattr(dataset, "sep_id", None)
    if pad_id is None or sep_id is None:
        raise RuntimeError("MolGen dataset missing pad/sep token IDs.")
    if serialization_mode == "pairs" and pad_id != sep_id:
        raise RuntimeError(
            f"This token counting script assumes pad_id == sep_id for pairs mode, "
            f"but got pad_id={pad_id}, sep_id={sep_id}."
        )

    items = 0
    samples = 0
    batches = 0
    pad_total = 0
    smiles_total = 0
    conformer_total = 0
    smiles_tag_total = 0
    conformer_tag_total = 0

    for batch in loader:
        batches += 1
        inputs = _extract_inputs(batch)
        attention_mask = _extract_attention_mask(batch)
        bsz = inputs.size(0)

        for idx in range(bsz):
            sample = inputs[idx]

            if serialization_mode == "isomer_units":
                if attention_mask is None:
                    raise RuntimeError(
                        "isomer_units mode requires attention_mask to count tokens correctly."
                    )
                mask = attention_mask[idx]
                attended = int(mask.sum().item())
                if attended == 0:
                    continue
                delimiter_count = int(
                    ((sample == pad_id) & (mask == 1)).sum().item()
                )
                items += delimiter_count + 1
                pad_total += int((mask == 0).sum().item())
                content_tokens = [
                    tok for tok, attn in zip(sample.tolist(), mask.tolist()) if attn == 1
                ]
            else:
                pad_count = 0
                for token in reversed(sample.tolist()):
                    if token == pad_id:
                        pad_count += 1
                    else:
                        break
                sep_total = int((sample == sep_id).sum().item())
                # Given pad_id == sep_id, trailing pad tokens are also counted as seps; subtract them.
                sep_count = max(sep_total - pad_count, 0)

                items += sep_count
                pad_total += pad_count
                content_tokens = sample.tolist()[: max(len(sample) - pad_count, 0)]
            samples += 1
            smi_count, conf_count, smi_tags, conf_tags = _count_tagged_spans(
                content_tokens, tokenizer
            )
            smiles_total += smi_count
            conformer_total += conf_count
            smiles_tag_total += smi_tags
            conformer_tag_total += conf_tags

            if items >= target_lines:
                break
        if items >= target_lines:
            break

    if samples == 0:
        return None

    tokens_produced = samples * seq_len
    avg_items_per_sample = items / samples
    avg_pad_per_sample = pad_total / samples
    avg_smiles_per_sample = smiles_total / samples
    avg_conformer_per_sample = conformer_total / samples
    avg_smiles_tags_per_sample = smiles_tag_total / samples
    avg_conformer_tags_per_sample = conformer_tag_total / samples
    effective_tokens = tokens_produced - pad_total

    return {
        "lines_target": float(target_lines),
        "lines_consumed": float(items),
        "samples": float(samples),
        "batches": float(batches),
        "tokens_produced": float(tokens_produced),
        "effective_tokens": float(effective_tokens),
        "avg_pad_per_sample": float(avg_pad_per_sample),
        "avg_items_per_sample": float(avg_items_per_sample),
        "avg_smiles_tokens_per_sample": float(avg_smiles_per_sample),
        "avg_conformer_tokens_per_sample": float(avg_conformer_per_sample),
        "avg_smiles_tags_per_sample": float(avg_smiles_tags_per_sample),
        "avg_conformer_tags_per_sample": float(avg_conformer_tags_per_sample),
        "pad_id": float(pad_id),
        "sep_id": float(sep_id),
    }


def verify_bytes(sample_bytes: Optional[int], sample_lines: int, total_lines: int, total_bytes: int) -> Optional[Dict[str, float]]:
    if not sample_bytes or sample_lines == 0 or total_lines == 0 or total_bytes == 0:
        return None
    factor = total_lines / sample_lines
    expected_bytes = sample_bytes * factor
    diff = abs(expected_bytes - total_bytes)
    diff_pct = (diff / total_bytes) * 100 if total_bytes else 0.0
    return {
        "expected_bytes": expected_bytes,
        "actual_bytes": float(total_bytes),
        "difference": diff,
        "difference_pct": diff_pct,
    }


def _tokenizer_signature(path: Path) -> Optional[str]:
    candidates = [
        path / "tokenizer.json",
        path / "vocab.json",
        path / "merges.txt",
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    hasher = hashlib.sha1()
    for candidate in existing:
        try:
            hasher.update(candidate.name.encode("utf-8"))
            with candidate.open("rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except OSError:
            return None
    return hasher.hexdigest()




def summarize_dataset(
    name: str,
    directory: str,
    tokenizer_aliases: List[str],
    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]],
    tokenizer_info_map: Dict[str, Dict[str, object]],
    seq_len: int,
    sample_lines: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    serialization_mode: str,
    unit_batch_size: int = 64,
    fast_estimate: bool = False,
    sample_units: int = 2000,
    sample_only: bool = False,
    sample_samples: int = 1000,
    sample_lines_for_units: int = 1000,
) -> Optional[Dict[str, Any]]:
    files = list_jsonl_files(directory)
    if not files:
        return None

    total_lines, total_bytes, file_stats = count_lines_and_bytes(files)

    dataset_summary: Dict[str, Any] = {
        "name": name,
        "path": directory,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "tokenizers": {},
    }

    for alias in tokenizer_aliases:
        tokenizer_path, tokenizer = tokenizer_map[alias]
        if serialization_mode == "isomer_units":
            if sample_only:
                avg_units_per_line = _estimate_units_per_line(
                    files, max_lines=max(1, int(sample_lines_for_units))
                )
                sample_stats = _sample_isomer_dataloader(
                    files,
                    tokenizer_path,
                    tokenizer,
                    seq_len,
                    batch_size=max(1, int(batch_size)),
                    shuffle=shuffle,
                    seed=seed,
                    max_samples=max(1, int(sample_samples)),
                )
                avg_units_per_sample = sample_stats.get(
                    "avg_smiles_tags_per_sample", 0.0
                )
                avg_pad_per_sample = sample_stats.get("avg_pad_per_sample", 0.0)
                avg_effective_tokens = sample_stats.get(
                    "avg_effective_tokens_per_sample", 0.0
                )
                total_units = total_lines * float(avg_units_per_line)
                estimated_samples = (
                    int(total_units / avg_units_per_sample)
                    if avg_units_per_sample
                    else 0
                )
                estimated_tokens = estimated_samples * seq_len
                estimated_pad = int(avg_pad_per_sample * estimated_samples)
                estimated_effective = estimated_tokens - estimated_pad
                total_smiles_tags = sample_stats.get("total_smiles_tags", 0.0)
                unique_smiles = sample_stats.get("unique_smiles", 0.0)
                unique_ratio = (
                    unique_smiles / total_smiles_tags if total_smiles_tags else 0.0
                )
                dataset_summary["tokenizers"][alias] = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "sampled_files": len(files),
                    "lines_target_total": total_lines,
                    "lines_consumed_total": total_units,
                    "sample_bytes_total": total_bytes,
                    "batches_sampled": estimated_samples,
                    "tokens_produced_sampled": estimated_tokens,
                    "effective_tokens_sampled": estimated_effective,
                    "avg_items_per_sample": float(avg_units_per_sample),
                    "avg_pad_per_sample": float(avg_pad_per_sample),
                    "avg_smiles_tokens_per_sample": 0.0,
                    "avg_conformer_tokens_per_sample": 0.0,
                    "avg_smiles_tags_per_sample": float(avg_units_per_sample),
                    "avg_conformer_tags_per_sample": float(
                        sample_stats.get("avg_conformer_tags_per_sample", 0.0)
                    ),
                    "valid_ratio": 1.0,
                    "estimated_valid_lines": int(total_units),
                    "estimated_samples": estimated_samples,
                    "estimated_batches": estimated_samples,
                    "estimated_tokens": estimated_tokens,
                    "estimated_pad": estimated_pad,
                    "estimated_effective": estimated_effective,
                    "estimated_smiles_tokens": 0,
                    "estimated_conformer_tokens": 0,
                    "estimated_smiles_tags": int(total_units),
                    "estimated_conformer_tags": int(
                        total_units
                        * sample_stats.get("avg_conformer_tags_per_sample", 0.0)
                        / avg_units_per_sample
                    )
                    if avg_units_per_sample
                    else 0,
                    "unique_smiles_tags": int(unique_ratio * total_units),
                    "total_smiles_tags": int(total_units),
                    "total_conformer_tags": int(
                        total_units
                        * sample_stats.get("avg_conformer_tags_per_sample", 0.0)
                        / avg_units_per_sample
                    )
                    if avg_units_per_sample
                    else 0,
                    "total_units": int(total_units),
                    "total_samples": int(estimated_samples),
                    "total_tokens": int(estimated_tokens),
                    "verification": None,
                    "tokenizer_path": tokenizer_path,
                    "tokenizer_info": {
                        **tokenizer_info_map.get(alias, {}),
                        "sample_only": True,
                        "avg_units_per_line": avg_units_per_line,
                        "sampled_sequences": int(sample_stats.get("samples", 0)),
                        "sampled_lines_for_units": int(sample_lines_for_units),
                    },
                }
                continue
            token_per_char = None
            if fast_estimate:
                raw_units, raw_confs, unique_smiles, sampled_units = _scan_isomer_stats(
                    files, sample_units, seed
                )
                chunk_lengths, avg_confs_emitted, avg_confs_dropped = _sample_chunk_stats(
                    sampled_units, tokenizer, seq_len
                )
                if not chunk_lengths:
                    raise RuntimeError(
                        "No sample chunks generated; increase --sample-units."
                    )
                avg_chunks_per_unit = len(chunk_lengths) / max(len(sampled_units), 1)
                est_chunks_total = int(round(raw_units * avg_chunks_per_unit))
                target_chunks = max(est_chunks_total, len(chunk_lengths) * 20)
                (
                    sim_sequences,
                    sim_pad_end,
                    sim_pad_delim,
                    sim_loss_tokens,
                    sim_chunks,
                ) = _simulate_packing_from_chunks(
                    chunk_lengths, seq_len, target_chunks
                )
                sequences_per_chunk = sim_sequences / max(sim_chunks, 1)
                pad_end_per_chunk = sim_pad_end / max(sim_chunks, 1)
                pad_delim_per_chunk = sim_pad_delim / max(sim_chunks, 1)
                loss_per_chunk = sim_loss_tokens / max(sim_chunks, 1)

                sequences = int(round(est_chunks_total * sequences_per_chunk))
                pad_total = int(round(est_chunks_total * pad_end_per_chunk))
                pad_delim_total = int(round(est_chunks_total * pad_delim_per_chunk))
                loss_tokens_total = int(round(est_chunks_total * loss_per_chunk))
                tokens_produced = sequences * seq_len
                effective_tokens = tokens_produced - pad_total

                confs_emitted = int(round(raw_units * avg_confs_emitted))
                confs_dropped = int(round(raw_units * avg_confs_dropped))
                confs_emitted = max(min(confs_emitted, raw_confs), 0)
                confs_dropped = max(raw_confs - confs_emitted, 0)

                dataset_summary["tokenizers"][alias] = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "sampled_files": len(files),
                    "lines_target_total": total_lines,
                    "lines_consumed_total": raw_units,
                    "sample_bytes_total": total_bytes,
                    "batches_sampled": sequences,
                    "tokens_produced_sampled": tokens_produced,
                    "effective_tokens_sampled": effective_tokens,
                    "avg_items_per_sample": float(raw_units / sequences) if sequences else 0.0,
                    "avg_pad_per_sample": float(pad_total / sequences) if sequences else 0.0,
                    "avg_smiles_tokens_per_sample": 0.0,
                    "avg_conformer_tokens_per_sample": 0.0,
                    "avg_smiles_tags_per_sample": float(raw_units / sequences) if sequences else 0.0,
                    "avg_conformer_tags_per_sample": float(confs_emitted / sequences)
                    if sequences
                    else 0.0,
                    "valid_ratio": 1.0,
                    "estimated_valid_lines": raw_units,
                    "estimated_samples": sequences,
                    "estimated_batches": sequences,
                    "estimated_tokens": tokens_produced,
                    "estimated_pad": pad_total,
                    "estimated_effective": effective_tokens,
                    "estimated_smiles_tokens": 0,
                    "estimated_conformer_tokens": 0,
                    "estimated_smiles_tags": raw_units,
                    "estimated_conformer_tags": confs_emitted,
                    "unique_smiles_tags": len(unique_smiles),
                    "total_smiles_tags": raw_units,
                    "total_conformer_tags": raw_confs,
                    "total_units": raw_units,
                    "total_samples": sequences,
                    "total_tokens": tokens_produced,
                    "chunks_total": est_chunks_total,
                    "confs_emitted": confs_emitted,
                    "confs_dropped_oversize": confs_dropped,
                    "pad_end_total": pad_total,
                    "pad_delim_total": pad_delim_total,
                    "loss_tokens_total": loss_tokens_total,
                    "verification": None,
                    "tokenizer_path": tokenizer_path,
                    "tokenizer_info": {
                        **tokenizer_info_map.get(alias, {}),
                        "fast_estimate": True,
                        "sample_units": len(sampled_units),
                        "avg_chunks_per_unit": avg_chunks_per_unit,
                    },
                }
                continue
            if not fast_estimate:
                exact = _exact_isomer_scan(files, tokenizer, seq_len)
                dataset_summary["tokenizers"][alias] = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "sampled_files": len(files),
                    "lines_target_total": total_lines,
                    "lines_consumed_total": exact["raw_units"],
                    "sample_bytes_total": total_bytes,
                    "batches_sampled": exact["sequences_total"],
                    "tokens_produced_sampled": exact["tokens_total"],
                    "effective_tokens_sampled": exact["attended_tokens_total"],
                    "avg_items_per_sample": float(exact["raw_units"] / exact["sequences_total"])
                    if exact["sequences_total"]
                    else 0.0,
                    "avg_pad_per_sample": float(exact["pad_end_total"] / exact["sequences_total"])
                    if exact["sequences_total"]
                    else 0.0,
                    "avg_smiles_tokens_per_sample": 0.0,
                    "avg_conformer_tokens_per_sample": 0.0,
                    "avg_smiles_tags_per_sample": float(exact["raw_units"] / exact["sequences_total"])
                    if exact["sequences_total"]
                    else 0.0,
                    "avg_conformer_tags_per_sample": float(exact["confs_emitted"] / exact["sequences_total"])
                    if exact["sequences_total"]
                    else 0.0,
                    "valid_ratio": 1.0,
                    "estimated_valid_lines": exact["raw_units"],
                    "estimated_samples": exact["sequences_total"],
                    "estimated_batches": exact["sequences_total"],
                    "estimated_tokens": exact["tokens_total"],
                    "estimated_pad": exact["pad_end_total"],
                    "estimated_effective": exact["attended_tokens_total"],
                    "estimated_smiles_tokens": 0,
                    "estimated_conformer_tokens": 0,
                    "estimated_smiles_tags": exact["raw_units"],
                    "estimated_conformer_tags": exact["confs_emitted"],
                    "unique_smiles_tags": exact["unique_smiles_tags"],
                    "total_smiles_tags": exact["raw_units"],
                    "total_conformer_tags": exact["raw_confs"],
                    "total_units": exact["raw_units"],
                    "total_samples": exact["sequences_total"],
                    "total_tokens": exact["tokens_total"],
                    "chunks_total": exact["chunks_total"],
                    "confs_emitted": exact["confs_emitted"],
                    "confs_dropped_oversize": exact["confs_dropped_oversize"],
                    "pad_end_total": exact["pad_end_total"],
                    "pad_delim_total": exact["pad_delim_total"],
                    "loss_tokens_total": exact["loss_tokens_total"],
                    "verification": None,
                    "tokenizer_path": tokenizer_path,
                    "tokenizer_info": {
                        **tokenizer_info_map.get(alias, {}),
                        "fast_estimate": False,
                    },
                }
                continue
        sum_samples = 0.0
        sum_items = 0.0
        sum_pad = 0.0
        sum_batches = 0.0
        sum_tokens_produced = 0.0
        sum_effective = 0.0
        sum_target_lines = 0.0
        sum_sample_bytes = 0.0
        sum_smiles_tokens = 0.0
        sum_conformer_tokens = 0.0
        sum_smiles_tags = 0.0
        sum_conformer_tags = 0.0

        for file_path in files:
            file_line_count = next((f["lines"] for f in file_stats if f["path"] == str(file_path)), None)
            if file_line_count is None:
                try:
                    with file_path.open("rb") as fh:
                        file_line_count = sum(1 for _ in fh)
                except OSError:
                    continue
            target_lines = min(sample_lines, file_line_count)

            stats = sample_dataloader(
                file_path,
                tokenizer_path,
                tokenizer,
                seq_len,
                target_lines=target_lines,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                serialization_mode=serialization_mode,
            )
            if not stats:
                continue

            sum_samples += stats["samples"]
            sum_items += stats["lines_consumed"]
            sum_pad += stats["avg_pad_per_sample"] * stats["samples"]
            sum_batches += stats["batches"]
            sum_tokens_produced += stats["tokens_produced"]
            sum_effective += stats["effective_tokens"]
            sum_target_lines += stats["lines_target"]
            sum_smiles_tokens += stats["avg_smiles_tokens_per_sample"] * stats["samples"]
            sum_conformer_tokens += stats["avg_conformer_tokens_per_sample"] * stats["samples"]
            sum_smiles_tags += stats["avg_smiles_tags_per_sample"] * stats["samples"]
            sum_conformer_tags += stats["avg_conformer_tags_per_sample"] * stats["samples"]

            sample_bytes = bytes_for_lines(file_path, int(target_lines))
            if sample_bytes:
                sum_sample_bytes += sample_bytes

        if sum_samples == 0 or sum_target_lines == 0:
            dataset_summary["tokenizers"][alias] = None
            continue

        avg_items_per_sample = sum_items / sum_samples
        avg_pad_per_sample = sum_pad / sum_samples
        avg_smiles_per_sample = sum_smiles_tokens / sum_samples
        avg_conformer_per_sample = sum_conformer_tokens / sum_samples
        avg_smiles_tags_per_sample = sum_smiles_tags / sum_samples
        avg_conformer_tags_per_sample = sum_conformer_tags / sum_samples
        valid_ratio = sum_items / sum_target_lines  # fraction of sampled lines that produced usable items

        estimated_valid_lines = total_lines * valid_ratio
        estimated_samples = int(estimated_valid_lines / avg_items_per_sample) if avg_items_per_sample else 0
        estimated_batches = math.ceil(estimated_samples / batch_size) if batch_size else 0
        estimated_tokens = estimated_samples * seq_len
        estimated_pad = int(avg_pad_per_sample * estimated_samples)
        estimated_effective = estimated_tokens - estimated_pad
        estimated_smiles_tokens = int(avg_smiles_per_sample * estimated_samples)
        estimated_conformer_tokens = int(avg_conformer_per_sample * estimated_samples)
        estimated_smiles_tags = int(avg_smiles_tags_per_sample * estimated_samples)
        estimated_conformer_tags = int(avg_conformer_tags_per_sample * estimated_samples)

        # Byte-size sanity check: scale sampled bytes to total lines.
        verification = verify_bytes(
            sum_sample_bytes if sum_sample_bytes else None,
            int(sum_target_lines),
            total_lines,
            total_bytes,
        )

        dataset_summary["tokenizers"][alias] = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "sampled_files": len(files),
            "lines_target_total": int(sum_target_lines),
            "lines_consumed_total": int(sum_items),
            "sample_bytes_total": int(sum_sample_bytes),
            "batches_sampled": int(sum_batches),
            "tokens_produced_sampled": int(sum_tokens_produced),
            "effective_tokens_sampled": int(sum_effective),
            "avg_items_per_sample": float(avg_items_per_sample),
            "avg_pad_per_sample": float(avg_pad_per_sample),
            "avg_smiles_tokens_per_sample": float(avg_smiles_per_sample),
            "avg_conformer_tokens_per_sample": float(avg_conformer_per_sample),
            "avg_smiles_tags_per_sample": float(avg_smiles_tags_per_sample),
            "avg_conformer_tags_per_sample": float(avg_conformer_tags_per_sample),
            "valid_ratio": float(valid_ratio),
            "estimated_valid_lines": int(estimated_valid_lines),
            "estimated_samples": estimated_samples,
            "estimated_batches": estimated_batches,
            "estimated_tokens": estimated_tokens,
            "estimated_pad": estimated_pad,
            "estimated_effective": estimated_effective,
            "estimated_smiles_tokens": estimated_smiles_tokens,
            "estimated_conformer_tokens": estimated_conformer_tokens,
            "estimated_smiles_tags": estimated_smiles_tags,
            "estimated_conformer_tags": estimated_conformer_tags,
            "unique_smiles_tags": 0,
            "total_smiles_tags": estimated_smiles_tags,
            "total_conformer_tags": estimated_conformer_tags,
            "total_units": estimated_valid_lines,
            "total_samples": estimated_samples,
            "total_tokens": estimated_tokens,
            "verification": verification,
        "tokenizer_path": tokenizer_path,
            "tokenizer_info": tokenizer_info_map.get(alias, {}),
        }

    return dataset_summary


def print_train_report(summary: Optional[Dict[str, Any]]) -> None:
    if not summary:
        print("SKIP dataset: no data collected.")
        return

    name = summary.get("name", "dataset").upper()
    print(f"\n{name} DATASET")
    print(f"  path: {summary.get('path')}")
    print(f"  total lines: {summary.get('total_lines', 0):,}")

    for alias, stats in (summary.get("tokenizers") or {}).items():
        print(f"\n  Tokenizer: {alias}")
        if stats is None:
            print("    no samples collected")
            continue
        tok_path = stats.get("tokenizer_path")
        tok_info = stats.get("tokenizer_info") or {}
        if tok_path:
            print(f"    path: {tok_path}")
        if tok_info:
            vocab = tok_info.get("vocab_size")
            added = tok_info.get("added_tokens")
            signature = tok_info.get("signature")
            fast_est = tok_info.get("fast_estimate")
            tpc = tok_info.get("token_per_char")
            if vocab is not None:
                print(f"    vocab_size: {vocab}")
            if added is not None:
                print(f"    added_tokens: {added}")
            if signature:
                print(f"    signature: {signature}")
            if fast_est is not None:
                print(f"    fast_estimate: {fast_est}")
            if tpc:
                print(f"    token_per_char: {tpc:.4f}")
        print(
            "    estimates: units≈{estimated_valid_lines:,}, "
            "samples≈{estimated_samples:,}, tokens≈{estimated_tokens:,}, "
            "pad≈{estimated_pad:,}, effective≈{estimated_effective:,}, "
            "smiles≈{estimated_smiles_tokens:,}, conformer≈{estimated_conformer_tokens:,}, "
            "smiles_tags≈{estimated_smiles_tags:,}, conformer_tags≈{estimated_conformer_tags:,}, "
            "unique_smiles_tags≈{unique_smiles_tags:,}".format(
                estimated_valid_lines=stats.get("estimated_valid_lines", 0),
                estimated_samples=stats.get("estimated_samples", 0),
                estimated_tokens=stats.get("estimated_tokens", 0),
                estimated_pad=stats.get("estimated_pad", 0),
                estimated_effective=stats.get("estimated_effective", 0),
                estimated_smiles_tokens=stats.get("estimated_smiles_tokens", 0),
                estimated_conformer_tokens=stats.get("estimated_conformer_tokens", 0),
                estimated_smiles_tags=stats.get("estimated_smiles_tags", 0),
                estimated_conformer_tags=stats.get("estimated_conformer_tags", 0),
                unique_smiles_tags=stats.get("unique_smiles_tags", 0),
            )
        )
        print(
            "    totals: samples={total_samples:,}, tokens={total_tokens:,}, "
            "units={total_units:,}, smiles_tags={total_smiles_tags:,}, "
            "conformer_tags={total_conformer_tags:,}, unique_smiles_tags={unique_smiles_tags:,}".format(
                total_samples=stats.get("total_samples", 0),
                total_tokens=stats.get("total_tokens", 0),
                total_units=stats.get("total_units", 0),
                total_smiles_tags=stats.get("total_smiles_tags", 0),
                total_conformer_tags=stats.get("total_conformer_tags", 0),
                unique_smiles_tags=stats.get("unique_smiles_tags", 0),
            )
        )
        if stats.get("exact_mode") or stats.get("chunks_total") is not None:
            print(
                "    exact: chunks={chunks_total:,}, confs_emitted={confs_emitted:,}, "
                "confs_dropped={confs_dropped_oversize:,}, pad_end={pad_end_total:,}, "
                "pad_delim={pad_delim_total:,}, loss_tokens={loss_tokens_total:,}".format(
                    chunks_total=stats.get("chunks_total", 0),
                    confs_emitted=stats.get("confs_emitted", 0),
                    confs_dropped_oversize=stats.get("confs_dropped_oversize", 0),
                    pad_end_total=stats.get("pad_end_total", 0),
                    pad_delim_total=stats.get("pad_delim_total", 0),
                    loss_tokens_total=stats.get("loss_tokens_total", 0),
            )
        )


def exhaust_validation_dataset(
    directory: str,
    tokenizer_aliases: List[str],
    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]],
    seq_len: int,
    batch_size: int,
    num_workers: int,
    serialization_mode: str,
) -> Optional[Dict[str, Any]]:
    files = list_jsonl_files(directory)
    if not files:
        return None

    total_lines, total_bytes, _ = count_lines_and_bytes(files)
    result: Dict[str, Any] = {
        "path": directory,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "tokenizers": {},
    }

    for alias in tokenizer_aliases:
        tokenizer_path, tokenizer = tokenizer_map[alias]
        # For PyTorch DataLoader compatibility: prefetch_factor must be an int when workers > 0.
        prefetch_factor = None if num_workers == 0 else 2
        loader = build_dataloader(
            train_path=directory,
            tokenizer_path=tokenizer_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle_lines=False,
            infinite=False,
            seed=0,
            min_emb_len=0,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=prefetch_factor,
            world_size=1,
            rank=0,
            serialization_mode=serialization_mode,
            emit_attention_mask=(serialization_mode == "isomer_units"),
        )

        dataset = getattr(loader, "dataset", None)
        _ensure_dataset_tokenizer(dataset)
        pad_id = getattr(dataset, "pad_id", None)
        sep_id = getattr(dataset, "sep_id", None)
        sample_count = 0
        batch_count = 0
        token_count = 0
        pad_total = 0

        for batch in loader:
            input_ids = _extract_inputs(batch)
            attention_mask = _extract_attention_mask(batch)
            if input_ids.numel() == 0:
                continue

            batch_count += 1
            sample_count += input_ids.size(0)
            token_count += input_ids.numel()
            if serialization_mode == "isomer_units":
                if attention_mask is None:
                    raise RuntimeError(
                        "isomer_units mode requires attention_mask to count tokens correctly."
                    )
                pad_total += int((attention_mask == 0).sum().item())
            else:
                if pad_id is not None:
                    pad_total += int((input_ids == pad_id).sum().item())

        effective_tokens = token_count - pad_total
        utilization = (effective_tokens / token_count * 100) if token_count else 0.0

        result["tokenizers"][alias] = {
            "samples": sample_count,
            "batches": batch_count,
            "token_count": token_count,
            "pad_count": pad_total,
            "effective_tokens": effective_tokens,
            "utilization": utilization,
            "pad_id": pad_id,
            "sep_id": sep_id,
            "tokenizer_path": tokenizer_path,
        }

    return result


def print_validation_report(summary: Optional[Dict[str, Any]]) -> None:
    if not summary:
        print("\nSKIP validation dataset (no files).")
        return

    print("\nVALIDATION DATASET")
    print(f"  path: {summary['path']}")
    print(f"  total lines: {summary.get('total_lines', 0):,}")

    for alias, stats in summary.get("tokenizers", {}).items():
        print(f"\n  Tokenizer: {alias}")
        if stats.get("samples", 0) == 0:
            print("    no samples processed")
            continue
        tok_path = stats.get("tokenizer_path")
        if tok_path:
            print(f"    path: {tok_path}")
        print(
            f"    samples={stats['samples']:,}, tokens={stats['token_count']:,}, "
            f"pad={stats['pad_count']:,}, effective={stats['effective_tokens']:,} "
            f"({stats['utilization']:.2f}% util)"
        )


def print_overall_summary(
    train_summary: Optional[Dict[str, Any]],
    validation_summary: Optional[Dict[str, Any]],
    elapsed: float,
) -> None:
    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {elapsed:.2f}s")

    if train_summary:
        print("\nTRAIN ESTIMATES OVERVIEW")
        train_path = train_summary.get("path")
        sample_file = train_summary.get("sample_file")
        if train_path:
            print(f"  Dataset path: {train_path}")
        if sample_file:
            sample_path = f"{train_path}/{sample_file}" if train_path else sample_file
            print(f"  Sample file: {sample_path}")
        print(f"  Total lines: {train_summary.get('total_lines', 0):,}")
        for alias, stats in (train_summary.get("tokenizers") or {}).items():
            if not stats:
                print(f"  {alias}: no stats")
                continue
            print(
                f"  {alias}: units≈{stats['estimated_valid_lines']:,}, "
                f"samples≈{stats['estimated_samples']:,}, "
                f"tokens≈{stats['estimated_tokens']:,}, pad≈{stats['estimated_pad']:,}, "
                f"effective≈{stats['estimated_effective']:,}"
            )

    if validation_summary:
        print("\nVALIDATION COUNTS OVERVIEW")
        valid_path = validation_summary.get("path")
        if valid_path:
            print(f"  Dataset path: {valid_path}")
        print(f"  Total lines: {validation_summary.get('total_lines', 0):,}")
        for alias, stats in (validation_summary.get("tokenizers") or {}).items():
            if stats.get("samples", 0) == 0:
                print(f"  {alias}: no samples processed")
                continue
            print(
                f"  {alias}: samples={stats['samples']:,}, "
                f"tokens={stats['token_count']:,}, pad={stats['pad_count']:,}, "
                f"effective={stats['effective_tokens']:,} ({stats['utilization']:.2f}%)"
            )


def _write_summary_file(
    train_summary: Optional[Dict[str, Any]],
    validation_summary: Optional[Dict[str, Any]],
    elapsed: float,
) -> None:
    """
    Append the final summary to SUMMARY_PATH for offline inspection.
    """
    lines: list[str] = []
    lines.append("\n" + "=" * 70)
    lines.append("RUN SUMMARY (saved)")
    lines.append("=" * 70)
    lines.append(f"Total runtime: {elapsed:.2f}s")

    if train_summary:
        lines.append("\nTRAIN ESTIMATES OVERVIEW")
        train_path = train_summary.get("path")
        sample_file = train_summary.get("sample_file")
        if train_path:
            lines.append(f"  Dataset path: {train_path}")
        if sample_file:
            sample_path = f"{train_path}/{sample_file}" if train_path else sample_file
            lines.append(f"  Sample file: {sample_path}")
        lines.append(f"  Total lines: {train_summary.get('total_lines', 0):,}")
        for alias, stats in (train_summary.get("tokenizers") or {}).items():
            if not stats:
                lines.append(f"  {alias}: no stats")
                continue
            tok_path = stats.get("tokenizer_path")
            tok_info = stats.get("tokenizer_info") or {}
            if tok_path:
                lines.append(f"    path: {tok_path}")
            if tok_info:
                vocab = tok_info.get("vocab_size")
                added = tok_info.get("added_tokens")
                signature = tok_info.get("signature")
                if vocab is not None:
                    lines.append(f"    vocab_size: {vocab}")
                if added is not None:
                    lines.append(f"    added_tokens: {added}")
                if signature:
                    lines.append(f"    signature: {signature}")
            lines.append(
                f"  {alias}: units≈{stats['estimated_valid_lines']:,}, "
                f"samples≈{stats['estimated_samples']:,}, "
                f"tokens≈{stats['estimated_tokens']:,}, pad≈{stats['estimated_pad']:,}, "
                f"effective≈{stats['estimated_effective']:,}, "
                f"smiles≈{stats.get('estimated_smiles_tokens', 0):,}, "
                f"conformer≈{stats.get('estimated_conformer_tokens', 0):,}, "
                f"smiles_tags≈{stats.get('estimated_smiles_tags', 0):,}, "
                f"conformer_tags≈{stats.get('estimated_conformer_tags', 0):,}, "
                f"unique_smiles_tags≈{stats.get('unique_smiles_tags', 0):,}"
            )

    if validation_summary:
        lines.append("\nVALIDATION COUNTS OVERVIEW")
        valid_path = validation_summary.get("path")
        if valid_path:
            lines.append(f"  Dataset path: {valid_path}")
        lines.append(f"  Total lines: {validation_summary.get('total_lines', 0):,}")
        for alias, stats in (validation_summary.get("tokenizers") or {}).items():
            if stats.get("samples", 0) == 0:
                lines.append(f"  {alias}: no samples processed")
                continue
            lines.append(
                f"  {alias}: samples={stats['samples']:,}, "
                f"tokens={stats['token_count']:,}, pad={stats['pad_count']:,}, "
                f"effective={stats['effective_tokens']:,} ({stats['utilization']:.2f}%)"
            )

    try:
        with SUMMARY_PATH.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as exc:
        print(f"Failed to append final summary to {SUMMARY_PATH}: {exc}")


def dump_json_summary(
    train_summary: Optional[Dict[str, Any]],
    validation_summary: Optional[Dict[str, Any]],
    elapsed: float,
) -> None:
    """
    Emit a machine-readable summary so callers can parse results without scraping stdout.
    """
    payload = {
        "train": train_summary,
        "validation": validation_summary,
        "elapsed_seconds": elapsed,
    }
    try:
        print("\nJSON_SUMMARY_START")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("JSON_SUMMARY_END")
    except Exception:
        # Fallback: avoid crashing the script if serialization fails on unexpected fields.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified MolGen3D token counting tool")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--sample-lines", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokenizers", nargs="+", default=["qwen3_0.6b_origin", "qwen3_0.6b_custom"])
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--shuffle", action="store_true", help="Sample random lines via dataloader shuffle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--validation-num-workers", type=int, default=0)
    parser.add_argument(
        "--train-path",
        type=str,
        default="",
        help="Dataset path or alias for training JSONL directory (defaults to conformers_train).",
    )
    parser.add_argument(
        "--validation-path",
        type=str,
        default="",
        help="Dataset path or alias for validation JSONL directory (defaults to conformers_valid).",
    )
    parser.add_argument(
        "--serialization-mode",
        choices=["pairs", "isomer_units"],
        default="pairs",
        help="Dataset serialization/packing mode.",
    )
    parser.add_argument(
        "--unit-batch-size",
        type=int,
        default=64,
        help="Tokenizer batch size for isomer_units counting.",
    )
    parser.add_argument(
        "--fast-estimate",
        dest="fast_estimate",
        action="store_true",
        help="Use fast approximate token estimation for isomer_units.",
    )
    parser.add_argument(
        "--exact-estimate",
        dest="fast_estimate",
        action="store_false",
        help="Use exact tokenization for isomer_units (slower).",
    )
    parser.set_defaults(fast_estimate=None)
    parser.add_argument(
        "--sample-units",
        type=int,
        default=2000,
        help="Number of units to sample for fast estimate calibration.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Estimate isomer_units stats from sampled sequences only.",
    )
    parser.add_argument(
        "--sample-samples",
        type=int,
        default=1000,
        help="Number of sequences to sample when using --sample-only.",
    )
    parser.add_argument(
        "--sample-lines-for-units",
        type=int,
        default=1000,
        help="Number of JSONL lines to sample for avg units per line.",
    )

    args = parser.parse_args()

    train_path = (
        args.train_path.strip()
        if args.train_path.strip()
        else str(get_data_path("conformers_train"))
    )
    valid_path = (
        args.validation_path.strip()
        if args.validation_path.strip()
        else str(get_data_path("conformers_valid"))
    )

    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]] = {}
    tokenizer_info_map: Dict[str, Dict[str, object]] = {}
    for alias in args.tokenizers:
        tok_path = str(get_tokenizer_path(alias))
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, use_fast=True, fix_mistral_regex=True
        )
        tokenizer_map[alias] = (tok_path, tokenizer)
        tokenizer_info_map[alias] = {
            "vocab_size": len(tokenizer),
            "added_tokens": len(getattr(tokenizer, "get_added_vocab", lambda: {})()),
            "signature": _tokenizer_signature(Path(tok_path)),
        }

    if len(tokenizer_map) > 1:
        signature_to_aliases: Dict[str, List[str]] = {}
        for alias, info in tokenizer_info_map.items():
            signature = info.get("signature")
            if not signature:
                continue
            signature_to_aliases.setdefault(signature, []).append(alias)
        for signature, aliases in signature_to_aliases.items():
            if len(aliases) > 1:
                print(
                    "WARNING: multiple tokenizers share the same signature "
                    f"({signature}): {', '.join(sorted(aliases))}"
                )

    random.seed(args.seed)

    print("MolGen3D dataset token counting")
    print(f"seq_len: {args.seq_len}, sample_lines: {args.sample_lines}, batch_size: {args.batch_size}")

    start_time = time.time()

    train_summary = summarize_dataset(
        "train",
        train_path,
        args.tokenizers,
        tokenizer_map,
        tokenizer_info_map,
        seq_len=args.seq_len,
        sample_lines=args.sample_lines,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
        serialization_mode=args.serialization_mode,
        unit_batch_size=max(1, int(args.unit_batch_size)),
        fast_estimate=bool(args.fast_estimate)
        if args.fast_estimate is not None
        else (args.serialization_mode == "isomer_units"),
        sample_units=max(0, int(args.sample_units)),
        sample_only=bool(args.sample_only),
        sample_samples=max(1, int(args.sample_samples)),
        sample_lines_for_units=max(1, int(args.sample_lines_for_units)),
    )

    validation_summary = None
    if not args.skip_validation:
        validation_summary = exhaust_validation_dataset(
            valid_path,
            args.tokenizers,
            tokenizer_map,
            seq_len=args.seq_len,
            batch_size=args.validation_batch_size,
            num_workers=args.validation_num_workers,
            serialization_mode=args.serialization_mode,
        )

    print_train_report(train_summary)
    if not args.skip_validation:
        print_validation_report(validation_summary)


    elapsed = time.time() - start_time
    print_overall_summary(train_summary, validation_summary, elapsed)
    dump_json_summary(train_summary, validation_summary, elapsed)
    _write_summary_file(train_summary, validation_summary, elapsed)


if __name__ == "__main__":
    main()
