# pretraining/data/dataloaders/jsonl_tagged_packed_simple.py
from __future__ import annotations

import json
import time
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from collections import defaultdict, deque, OrderedDict
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from typing import Callable

try:  # TorchTitan is optional for some downstream uses
    from torchtitan.components.dataloader import BaseDataLoader
    from torchtitan.components.validate import BaseValidator, Validator
    from torchtitan.tools.logging import logger as titan_logger
except Exception:  # pragma: no cover - fallback for environments without torchtitan
    class BaseDataLoader:  # type: ignore[too-many-ancestors]
        """Lightweight stand-in so local tooling can import this module."""

        pass

    Validator = None  # type: ignore[assignment]
    titan_logger = None  # type: ignore[assignment]
    BaseValidator = None  # type: ignore[assignment]

MolGenValidatorClass = None


def _warn_log(msg: str, *args) -> None:
    if args:
        msg = msg % args
    if titan_logger is not None:
        titan_logger.warning(msg)
    else:
        logger.warning(msg)


from molgen3D.training.pretraining.config.custom_job_config import (
    JobConfig as MolGenJobConfig,
    MolGenDataConfig,
)
from molgen3D.training.pretraining.dataprocessing.sequence_packing import (
    PendingUnit,
    SequenceState,
)
from molgen3D.training.pretraining.dataprocessing.text_processing import (
    build_unit,
    is_valid_unit,
)
from molgen3D.training.pretraining.dataprocessing.utils import (
    build_line_index,
    expand_paths,
    idx_path,
    read_line_at,
)

try:  # optional faster JSON parser
    import orjson as _fast_json
except Exception:  # pragma: no cover
    _fast_json = None


def _json_loads(raw):
    """Parse JSON from either str or bytes."""
    if _fast_json is not None:
        # orjson expects bytes
        if isinstance(raw, bytes):
            return _fast_json.loads(raw)
        return _fast_json.loads(raw.encode("utf-8"))
    # stdlib json.loads accepts both str and bytes
    return json.loads(raw)


def _coerce_rng_state(state):
    """
    Torch random RNG expects a tuple (version, state, gauss). Some launchers may
    serialize StatefulDataLoader state via JSON, which converts tuples to lists
    or dict structures. Normalize those back into tuples before calling setstate.
    """

    def _tupleify(obj):
        if isinstance(obj, list):
            return tuple(_tupleify(x) for x in obj)
        return obj

    if state is None:
        return random.Random().getstate()

    if isinstance(state, dict):
        version = state.get("version")
        nested = _tupleify(state.get("state"))
        gauss = state.get("gauss")
        return (version, nested, gauss)

    if isinstance(state, (list, tuple)):
        return tuple(_tupleify(elem) for elem in state)

    return state


def _resolve_special_token_id(tokenizer, attr_name: str, fallback_tokens: Sequence[Optional[str]]) -> Optional[int]:
    """
    Attempts to resolve a tokenizer special token id, falling back to common aliases.
    """
    # Handle wrapped tokenizers (e.g. TorchTitan's HuggingFaceTokenizer)
    if not hasattr(tokenizer, "convert_tokens_to_ids") and hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    token_id = getattr(tokenizer, attr_name, None)
    if token_id is not None:
        try:
            return int(token_id)
        except (TypeError, ValueError):
            pass

    for token in fallback_tokens:
        if not token:
            continue
        
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            converted = tokenizer.convert_tokens_to_ids(token)
        elif hasattr(tokenizer, "token_to_id"):
            converted = tokenizer.token_to_id(token)
        else:
            converted = None

        if isinstance(converted, int) and converted >= 0:
            return int(converted)

    return None


def ensure_tokenizer_pad_token(tokenizer, token: str = "<|endoftext|>") -> int:
    """
    Ensure the tokenizer exposes <|endoftext|> and uses it as the pad token so
    separators/padding share the same id.
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        raise RuntimeError(f"Failed to add {token} token to tokenizer.")

    if tokenizer.pad_token != token:
        tokenizer.add_special_tokens({"pad_token": token})
        tokenizer.pad_token = token

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Failed to configure tokenizer pad token.")
    if pad_id != token_id:
        token_id = pad_id
    return int(token_id)


@dataclass(frozen=True)
class IsomerUnit:
    isomeric_smiles: str
    conf_embedded_strings: List[str]
    geom_id: Optional[str] = None


ROLE_CONTENT = 0
ROLE_DELIM_PAD = 1
ROLE_END_PAD = 2


def serialize_isomer_unit(unit: IsomerUnit, tokenizer, ctx_len: int) -> List[int]:
    smiles_str = f"[SMILES]{unit.isomeric_smiles}[/SMILES]"
    conf_strs = [
        f"[CONFORMER]{conf}[/CONFORMER]" for conf in unit.conf_embedded_strings
    ]
    full_str = smiles_str + "".join(conf_strs)
    unit_ids = tokenizer.encode(full_str, add_special_tokens=False)
    if len(unit_ids) > ctx_len:
        unit_ids = unit_ids[:ctx_len]
    return unit_ids


def _encode_isomer_chunks(
    iso_smiles: str,
    confs: List[str],
    tokenizer,
    ctx_len: int,
    stats: Optional["_DataloaderStats"] = None,
    smiles_tokens: Optional[List[int]] = None,
) -> List[List[int]]:
    """
    Build chunks so every conformer appears in a sequence that includes the SMILES.
    When conformers overflow, repeat the SMILES at the start of the next chunk.
    """
    if ctx_len <= 0:
        return []
    if smiles_tokens is None:
        smiles_str = f"[SMILES]{iso_smiles}[/SMILES]"
        if stats is not None and stats.enabled:
            stats.n_smiles_encode_calls += 1
        smiles_tokens = tokenizer.encode(smiles_str, add_special_tokens=False)
    if not smiles_tokens:
        return []
    if len(smiles_tokens) >= ctx_len:
        return [smiles_tokens[:ctx_len]]

    conf_texts = [f"[CONFORMER]{conf}[/CONFORMER]" for conf in confs]
    conf_token_lists: List[List[int]] = []
    if conf_texts:
        try:
            encoded = tokenizer(conf_texts, add_special_tokens=False)
            if stats is not None and stats.enabled:
                stats.n_batch_tokenizer_calls += 1
                stats.n_conformer_encode_calls += len(conf_texts)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                conf_token_lists = encoded["input_ids"]
            elif hasattr(encoded, "input_ids"):
                conf_token_lists = encoded.input_ids  # type: ignore[assignment]
            else:
                conf_token_lists = list(encoded)
        except Exception:
            conf_token_lists = [
                tokenizer.encode(text, add_special_tokens=False) for text in conf_texts
            ]
            if stats is not None and stats.enabled:
                stats.n_conformer_encode_calls += len(conf_texts)

    chunks: List[List[int]] = []
    current: List[int] = list(smiles_tokens)
    available = ctx_len - len(current)
    for conf_tokens in conf_token_lists:
        if not conf_tokens:
            continue
        if len(conf_tokens) > available:
            # If current chunk only has SMILES, truncate this conformer and stop.
            if len(current) == len(smiles_tokens):
                if stats is not None and stats.enabled:
                    stats.n_conformers_dropped_oversize += 1
                current.extend(conf_tokens[:available])
                chunks.append(current)
                return chunks
            # Otherwise, flush current and start a new chunk with SMILES.
            chunks.append(current)
            current = list(smiles_tokens)
            available = ctx_len - len(current)
            if len(conf_tokens) > available:
                if stats is not None and stats.enabled:
                    stats.n_conformers_dropped_oversize += 1
                current.extend(conf_tokens[:available])
                chunks.append(current)
                return chunks
        else:
            current.extend(conf_tokens)
            available -= len(conf_tokens)

    if current:
        chunks.append(current)
    return chunks


class IsomerUnitPacker:
    def __init__(
        self,
        ctx_len: int,
        pad_id: int,
        ignore_index: int,
        emit_attention_mask: bool = False,
        stats: Optional[_DataloaderStats] = None,
    ) -> None:
        self.ctx_len = int(ctx_len)
        self.pad_id = int(pad_id)
        self.ignore_index = int(ignore_index)
        self.emit_attention_mask = bool(emit_attention_mask)
        self._stats = stats
        self.buf_ids: List[int] = []
        self.buf_roles: List[int] = []

    def state_dict(self) -> Dict[str, List[int]]:
        return {
            "buf_ids": list(self.buf_ids),
            "buf_roles": list(self.buf_roles),
        }

    def load_state_dict(self, s: Dict) -> None:
        self.buf_ids = list(s.get("buf_ids", []))
        self.buf_roles = list(s.get("buf_roles", []))

    def reset(self) -> None:
        self.buf_ids = []
        self.buf_roles = []

    def _append_unit(self, unit_ids: List[int]) -> None:
        self.buf_ids.extend(unit_ids)
        self.buf_roles.extend([ROLE_CONTENT] * len(unit_ids))

    def _flush(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        start = time.perf_counter()
        pad_needed = self.ctx_len - len(self.buf_ids)
        if pad_needed > 0:
            self.buf_ids.extend([self.pad_id] * pad_needed)
            self.buf_roles.extend([ROLE_END_PAD] * pad_needed)

        input_ids = torch.tensor(self.buf_ids, dtype=torch.long)
        features: Dict[str, torch.Tensor] = {"input": input_ids}
        roles_tensor = torch.tensor(self.buf_roles, dtype=torch.long)
        if self.emit_attention_mask:
            features["attention_mask"] = (roles_tensor != ROLE_END_PAD).long()
        labels = torch.full(
            (self.ctx_len,),
            self.ignore_index,
            dtype=torch.long,
        )
        labels[:-1] = input_ids[1:]
        end_pad_mask = roles_tensor[1:] == ROLE_END_PAD
        if end_pad_mask.any():
            labels[:-1][end_pad_mask] = self.ignore_index

        if self._stats is not None and self._stats.enabled:
            self._stats.record_padding(self.buf_roles)
            self._stats.t_flush += time.perf_counter() - start
        self.reset()
        return features, labels

    def add_unit(
        self, unit_ids: List[int]
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        if not unit_ids:
            return
        if len(unit_ids) > self.ctx_len:
            unit_ids = unit_ids[: self.ctx_len]

        if not self.buf_ids:
            self._append_unit(unit_ids)
            if len(self.buf_ids) >= self.ctx_len:
                yield self._flush()
            return

        needed = 1 + len(unit_ids)
        if len(self.buf_ids) + needed <= self.ctx_len:
            self.buf_ids.append(self.pad_id)
            self.buf_roles.append(ROLE_DELIM_PAD)
            self._append_unit(unit_ids)
            if len(self.buf_ids) >= self.ctx_len:
                yield self._flush()
            return

        yield self._flush()
        self._append_unit(unit_ids)
        if len(self.buf_ids) >= self.ctx_len:
            yield self._flush()

    def finalize(
        self,
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        if self.buf_ids:
            yield self._flush()


def pack_units(
    units_ids_iter: Iterable[List[int]],
    pad_id: int,
    ctx_len: int,
    ignore_index: int = -100,
    emit_attention_mask: bool = False,
) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    packer = IsomerUnitPacker(
        ctx_len=ctx_len,
        pad_id=pad_id,
        ignore_index=ignore_index,
        emit_attention_mask=emit_attention_mask,
    )
    for unit_ids in units_ids_iter:
        yield from packer.add_unit(unit_ids)
    yield from packer.finalize()


class TitanStatefulDataLoader(StatefulDataLoader, BaseDataLoader):
    """
    TorchTitan-compatible wrapper around torchdata's StatefulDataLoader.

    TorchTitan expects dataloaders to inherit from BaseDataLoader so checkpoints
    can capture and restore their iteration state. By subclassing here we keep
    all of the behavior from StatefulDataLoader (num_workers, resume support,
    etc.) while satisfying TorchTitan's protocol.
    """

    ...


def _coerce_train_targets(
    train_path: Union[str, Sequence[str], Path]
) -> List[str]:
    """
    Normalize user-provided train_path entries into a sorted, de-duplicated list
    of concrete file paths. At this point the JobConfig has already resolved any
    repo-specific aliases, so we only need to honor relative paths and globs.
    """

    if isinstance(train_path, (str, Path)):
        candidates: List[Union[str, Path]] = [train_path]
    else:
        candidates = list(train_path)

    concrete = [str(Path(candidate)) for candidate in candidates]
    resolved = expand_paths(concrete)
    if not resolved:
        raise FileNotFoundError(f"No JSONL files matched: {train_path}")
    return resolved


class _PreviewLogger:
    """
    Logs the first few decoded samples dispatched by each rank so we can sanity-
    check the dataloader without flooding the logs.
    """

    def __init__(self, tokenizer, limit: int = 4) -> None:
        self._tokenizer = tokenizer
        self._limit = limit
        self._counts: Dict[int, int] = defaultdict(int)
        self._done: set[int] = set()

    def maybe_log(self, rank: int, tensor: torch.Tensor) -> None:
        if rank in self._done:
            return
        try:
            token_ids = tensor.tolist()
            decoded = self._tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            token_ids = "<encode_error>"
            decoded = f"<decode_error: {exc}>"

        if isinstance(token_ids, list) and len(token_ids) > 64:
            token_ids = token_ids[:64] + ["..."]

        seq_idx = self._counts[rank] + 1
        logger.warning(
            f"PREVIEW_SAMPLE rank={rank} idx={seq_idx} "
            f"len={len(token_ids) if isinstance(token_ids, list) else '?'} "
            f"ids={token_ids} decoded={decoded if decoded.strip() else '<empty_decode>'}"
        )
        self._counts[rank] = seq_idx

        if seq_idx >= self._limit:
            self._done.add(rank)


class _DataloaderStats:
    def __init__(self, tag: str, log_every: int = 1000) -> None:
        self.tag = tag
        self.log_every = max(1, int(log_every))
        self.enabled = True
        self.flushes_emitted = 0
        self.tokens_total = 0
        self.n_smiles_encode_calls = 0
        self.n_conformer_encode_calls = 0
        self.n_batch_tokenizer_calls = 0
        self.n_isomers_seen = 0
        self.n_chunks_emitted = 0
        self.n_conformers_seen = 0
        self.n_conformers_kept = 0
        self.n_conformers_dropped_oversize = 0
        self.pad_end_tokens_total = 0
        self.pad_delim_tokens_total = 0
        self.t_read_isomer_units = 0.0
        self.t_encode_isomer_chunks = 0.0
        self.t_fill_isomer_buffer = 0.0
        self.t_flush = 0.0
        self.t_pairs_for_epoch = 0.0
        self._last_log_flush = 0

    def record_padding(self, roles: List[int]) -> None:
        if not self.enabled:
            return
        self.pad_end_tokens_total += roles.count(ROLE_END_PAD)
        self.pad_delim_tokens_total += roles.count(ROLE_DELIM_PAD)

    def record_flush(self, rank: int, seq_len: int) -> None:
        if not self.enabled or rank != 0:
            return
        self.flushes_emitted += 1
        self.tokens_total += int(seq_len)
        if self.flushes_emitted - self._last_log_flush < self.log_every:
            return
        self._last_log_flush = self.flushes_emitted
        encode_calls = self.n_smiles_encode_calls + self.n_conformer_encode_calls
        tokens_total = self.tokens_total or max(1, self.flushes_emitted * int(seq_len))
        end_pad_frac = self.pad_end_tokens_total / max(1, tokens_total)
        delim_per_seq = self.pad_delim_tokens_total / max(1, self.flushes_emitted)
        chunks_per_isomer = (
            self.n_chunks_emitted / max(1, self.n_isomers_seen)
        )
        _warn_log(
            "DATALOADER_STATS tag=%s flushes=%d encode_calls_per_seq=%.3f "
            "chunks_per_isomer=%.3f end_pad_frac=%.3f delim_per_seq=%.3f "
            "ms_read_isomer=%.3f ms_encode_chunks=%.3f ms_fill_buffer=%.3f "
            "ms_flush=%.3f ms_pairs_for_epoch=%.3f",
            self.tag,
            self.flushes_emitted,
            encode_calls / max(1, self.flushes_emitted),
            chunks_per_isomer,
            end_pad_frac,
            delim_per_seq,
            (self.t_read_isomer_units * 1000.0) / max(1, self.flushes_emitted),
            (self.t_encode_isomer_chunks * 1000.0) / max(1, self.flushes_emitted),
            (self.t_fill_isomer_buffer * 1000.0) / max(1, self.flushes_emitted),
            (self.t_flush * 1000.0) / max(1, self.flushes_emitted),
            (self.t_pairs_for_epoch * 1000.0) / max(1, self.flushes_emitted),
        )

class JsonlTaggedPackedDataset(IterableDataset):
    """
    JSONL -> unit string -> fast tokenizer -> atomic sequences separated by <|endoftext|>.
    Buffered shuffle with limited lookahead keeps units intact and deterministic.
    """
    def __init__(
        self,
        train_path: Union[str, Sequence[str]],
        tokenizer_path: str,
        tokenizer=None,
        seq_len: int = 2048,
        min_emb_len: int = 16,
        shuffle_lines: bool = True,
        infinite: bool = True,
        seed: int = 1234,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle_buffer_size: int = 4096,
        lookahead_limit: int = 100,
        ignore_index: int = -100,
        truncate_overflow_units: bool = True,
        preview_enabled: bool = True,
        serialization_mode: str = "pairs",
        emit_attention_mask: bool = False,
        stats_tag: Optional[str] = None,
        stats_log_every: int = 1000,
    ):
        super().__init__()
        self.files = _coerce_train_targets(train_path)
        self.idxs = [build_line_index(p) for p in self.files]
        self._all_pairs: List[Tuple[int, int]] = []
        for fi, idx in enumerate(self.idxs):
            self._all_pairs.extend((fi, li) for li in range(len(idx)))
        self._all_pair_indices = list(range(len(self._all_pairs)))

        if world_size is None or rank is None:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.rank = int(os.environ.get("RANK", "0"))
        else:
            self.world_size = world_size
            self.rank = rank

        self._tokenizer_path = tokenizer_path
        self._tokenizer = tokenizer
        self.sep_id: Optional[int] = None
        self.pad_id: Optional[int] = None

        self.seq_len = int(seq_len)
        self.max_unit_tokens = max(0, self.seq_len - 1)
        self.min_emb_len = int(min_emb_len)
        self.shuffle_lines = bool(shuffle_lines)
        self.infinite = bool(infinite)
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.lookahead_limit = max(1, int(lookahead_limit))
        self.ignore_index = int(ignore_index)
        self.truncate_overflow_units = bool(truncate_overflow_units)
        self.serialization_mode = serialization_mode
        self.emit_attention_mask = bool(emit_attention_mask)
        if self.serialization_mode not in ("pairs", "isomer_units"):
            raise ValueError(
                "serialization_mode must be 'pairs' or 'isomer_units'. "
                f"Got: {self.serialization_mode}"
            )

        # Use a shared seed across ranks so that shuffling produces a single
        # global ordering which is then partitioned deterministically by rank.
        self._epoch = 0
        self._rng = random.Random(seed)
        self._start_k = 0
        self._pairs_total = 0
        self._pair_cursor = self._start_k
        self._pair_buffer: deque[Tuple[int, int]] = deque()
        self._pending_units: List[PendingUnit] = []
        self._sequence_state: Optional[SequenceState] = None
        self._monster_warning_shown = False
        self._truncation_warning_shown = False
        self._preview: Optional[_PreviewLogger] = None
        self._preview_enabled = bool(preview_enabled)
        self._iso_unit_buffer: List[List[int]] = []
        self._iso_packer: Optional[IsomerUnitPacker] = None
        self._smiles_token_cache: "OrderedDict[str, List[int]]" = OrderedDict()
        self._smiles_cache_max = int(os.environ.get("MOLGEN_SMILES_CACHE_MAX", "8192"))
        self._iso_refill_max_pairs = int(os.environ.get("MOLGEN_ISO_REFILL_MAX_PAIRS", "0"))
        self._iso_refill_max_sec = float(os.environ.get("MOLGEN_ISO_REFILL_MAX_SEC", "0"))
        tag = stats_tag or ("train" if shuffle_lines else "val")
        env_log_every = os.environ.get("MOLGEN_DATALOADER_LOG_EVERY")
        effective_log_every = stats_log_every
        if env_log_every:
            try:
                effective_log_every = int(env_log_every)
            except ValueError:
                effective_log_every = stats_log_every
        self._stats = _DataloaderStats(tag=tag, log_every=effective_log_every)
        self._stats.enabled = False
        self._reset_epoch_state()

    @property
    def tk(self):
        self._ensure_tokenizer_ready()
        return self._tokenizer

    def _ensure_tokenizer_ready(self) -> None:
        tokenizer = self._tokenizer
        # If no tokenizer is provided, or if the provided object does not expose
        # the standard HuggingFace tokenization APIs (e.g. TorchTitan's
        # HuggingFaceTokenizer wrapper), fall back to constructing a fresh
        # AutoTokenizer from the configured path so we can reliably control
        # padding / special tokens.
        if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
            tokenizer = AutoTokenizer.from_pretrained(
                str(self._tokenizer_path),
                use_fast=True,
                fix_mistral_regex=True,
            )
            sep_id = ensure_tokenizer_pad_token(tokenizer)
            self._tokenizer = tokenizer
            self.sep_id = sep_id
            self.pad_id = sep_id
        else:
            pad_id = _resolve_special_token_id(
                tokenizer,
                "pad_token_id",
                [
                    getattr(tokenizer, "pad_token", None),
                    "<pad>",
                    "<|endoftext|>",
                ],
            )
            if pad_id is None:
                raise RuntimeError(
                    "Provided tokenizer must expose a pad token id or support resolving one."
                )
            # Use the caller-provided pad token for both padding and separators.
            if getattr(tokenizer, "pad_token_id", None) is None:
                try:
                    tokenizer.pad_token_id = int(pad_id)  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.sep_id = int(pad_id)
            self.pad_id = int(pad_id)
        self._preview = _PreviewLogger(tokenizer, limit=2)
        if self.serialization_mode == "pairs":
            self._sequence_state = SequenceState(
                seq_len=self.seq_len,
                separator_id=self.sep_id,
                pad_id=self.pad_id,
                ignore_index=self.ignore_index,
            )
        else:
            self._sequence_state = None
            self._iso_packer = IsomerUnitPacker(
                ctx_len=self.seq_len,
                pad_id=self.pad_id,
                ignore_index=self.ignore_index,
                emit_attention_mask=self.emit_attention_mask,
                stats=self._stats,
            )

    def _pairs_for_epoch(self) -> List[Tuple[int, int]]:
        start = time.perf_counter()
        indices = list(self._all_pair_indices)
        if self.shuffle_lines:
            self._rng.shuffle(indices)
        pairs: List[Tuple[int, int]] = []
        for global_idx in indices:
            if (global_idx % self.world_size) != self.rank:
                continue
            pairs.append(self._all_pairs[global_idx])
        if self._stats.enabled:
            self._stats.t_pairs_for_epoch += time.perf_counter() - start
        return pairs

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1
        if wid != 0 and self._stats.enabled:
            self._stats.enabled = False

        self._ensure_tokenizer_ready()
        fps = [open(p, "rb") for p in self.files]
        try:
            while True:
                all_pairs = self._pairs_for_epoch()
                worker_pairs = [p for i, p in enumerate(all_pairs) if (i % nworkers) == wid]
                self._pairs_total = len(worker_pairs)
                preview_enabled = (
                    self._preview_enabled and (self._epoch == 0) and (wid == 0)
                )
                if self.serialization_mode == "pairs":
                    yield from self._pack_from_pairs(worker_pairs, fps, preview_enabled)
                else:
                    yield from self._pack_isomer_units(worker_pairs, fps, preview_enabled)

                if not self.infinite:
                    break
                self._epoch += 1
                self._start_k = 0
                self._reset_epoch_state()
        finally:
            for f in fps:
                f.close()

    def state_dict(self) -> Dict:
        sequence_tokens = (
            self._sequence_state.export_tokens() if self._sequence_state else []
        )
        base_state = {
            "epoch": self._epoch,
            "rng_state": self._rng.getstate(),
            "start_k": self._start_k,
            "pairs_total": self._pairs_total,
            "pair_buffer": list(self._pair_buffer),
            "serialization_mode": self.serialization_mode,
        }
        if self.serialization_mode == "pairs":
            base_state.update(
                {
                    "pending_units": [unit.tokens for unit in self._pending_units],
                    "sequence_tokens": sequence_tokens,
                }
            )
        else:
            base_state.update(
                {
                    "iso_unit_buffer": [list(tokens) for tokens in self._iso_unit_buffer],
                    "iso_packer": self._iso_packer.state_dict() if self._iso_packer else {},
                }
            )
        return base_state

    def load_state_dict(self, s: Dict):
        self._epoch = int(s.get("epoch", 0))
        rng_state = s.get("rng_state")
        self._rng.setstate(_coerce_rng_state(rng_state))
        self._start_k = int(s.get("start_k", 0))
        self._pairs_total = int(s.get("pairs_total", 0))
        self._pair_buffer = deque(tuple(pair) for pair in s.get("pair_buffer", []))
        if self.serialization_mode == "pairs":
            pending = s.get("pending_units", [])
            self._pending_units = [
                PendingUnit(tokens=list(tokens), total_len=len(tokens) + 1)
                for tokens in pending
            ]
            if self._sequence_state is None:
                self._ensure_tokenizer_ready()
            self._sequence_state.load_tokens(s.get("sequence_tokens", []))
        else:
            if self._iso_packer is None:
                self._ensure_tokenizer_ready()
            self._iso_unit_buffer = [
                list(tokens) for tokens in s.get("iso_unit_buffer", [])
            ]
            if self._iso_packer is not None:
                self._iso_packer.load_state_dict(s.get("iso_packer", {}))
        self._pair_cursor = self._start_k
        self._monster_warning_shown = False
        self._truncation_warning_shown = False

    def _reset_epoch_state(self) -> None:
        self._pair_buffer.clear()
        self._pending_units = []
        self._iso_unit_buffer = []
        if self._sequence_state is not None:
            self._sequence_state.reset()
        if self._iso_packer is not None:
            self._iso_packer.reset()
        self._pair_cursor = self._start_k
        self._monster_warning_shown = False
        self._truncation_warning_shown = False

    def _fill_pair_buffer(self, worker_pairs: List[Tuple[int, int]]) -> None:
        while (
            len(self._pair_buffer) < self.shuffle_buffer_size
            and self._pair_cursor < len(worker_pairs)
        ):
            self._pair_buffer.append(worker_pairs[self._pair_cursor])
            self._pair_cursor += 1
            self._start_k = self._pair_cursor

    def _ensure_pending_units(
        self,
        worker_pairs: List[Tuple[int, int]],
        fps: List[BinaryIO],
    ) -> None:
        self._fill_pair_buffer(worker_pairs)
        while len(self._pending_units) < self.lookahead_limit + 1:
            unit = self._draw_unit_from_buffer(worker_pairs, fps)
            if unit is None:
                break
            self._pending_units.append(unit)

    def _draw_unit_from_buffer(
        self,
        worker_pairs: List[Tuple[int, int]],
        fps: List[BinaryIO],
    ) -> Optional[PendingUnit]:
        while self._pair_buffer:
            pair = self._pair_buffer.popleft()
            unit = self._read_unit_from_pair(fps, pair)
            if unit is not None:
                return unit
        return None

    def _select_fitting_unit(self) -> Optional[PendingUnit]:
        limit = min(self.lookahead_limit, len(self._pending_units))
        if limit == 0:
            return None

        available_space = self.seq_len - self._sequence_state.used_len
        best_idx: Optional[int] = None
        best_remaining: Optional[int] = None

        for idx in range(limit):
            unit = self._pending_units[idx]
            if unit.total_len > available_space:
                continue
            remaining = available_space - unit.total_len
            if best_remaining is None or remaining < best_remaining:
                best_idx = idx
                best_remaining = remaining
                if remaining == 0:
                    break

        if best_idx is not None:
            return self._pending_units.pop(best_idx)

        return None

    def _evict_monster_unit(self) -> bool:
        threshold = self.seq_len - 1
        limit = min(self.lookahead_limit, len(self._pending_units))
        for idx in range(limit):
            unit = self._pending_units[idx]
            if unit.total_len > threshold:
                del self._pending_units[idx]
                self._maybe_log_monster(len(unit.tokens))
                return True
        return False

    def _maybe_log_monster(self, token_len: int) -> None:
        if self._monster_warning_shown:
            return
        logger.warning(
            "Rank {} skipping unit of {} tokens because it exceeds max tokens {}",
            self.rank,
            token_len,
            self.max_unit_tokens,
        )
        self._monster_warning_shown = True

    def _maybe_log_truncation(self, original_len: int, truncated_len: int) -> None:
        if self._truncation_warning_shown:
            return
        logger.warning(
            "Rank {} truncating unit from {} to {} tokens to fit max tokens {}",
            self.rank,
            original_len,
            truncated_len,
            self.max_unit_tokens,
        )
        self._truncation_warning_shown = True

    def _read_unit_from_pair(
        self,
        fps: List[BinaryIO],
        pair: Tuple[int, int],
    ) -> Optional[PendingUnit]:
        fi, li = pair
        raw = read_line_at(fps[fi], int(self.idxs[fi][li]))
        if not raw:
            return None

        try:
            obj = _json_loads(raw)
        except Exception:
            return None

        canon = (obj.get("canonical_smiles") or "").strip()
        emb = (obj.get("embedded_smiles") or "").strip()
        if not is_valid_unit(canon, emb, min_emb_len=self.min_emb_len):
            return None
        unit = build_unit(canon, emb)
        if self._stats.enabled:
            self._stats.n_smiles_encode_calls += 1
        tokens = self._tokenizer.encode(unit, add_special_tokens=False)
        if not tokens:
            return None
        if len(tokens) > self.max_unit_tokens:
            if not self.truncate_overflow_units:
                self._maybe_log_monster(len(tokens))
                return None
            original_len = len(tokens)
            tokens = tokens[: self.max_unit_tokens]
            self._maybe_log_truncation(original_len, len(tokens))
        return PendingUnit(tokens=tokens, total_len=len(tokens) + 1)

    def _get_smiles_tokens(self, iso_smiles: str) -> List[int]:
        cached = self._smiles_token_cache.get(iso_smiles)
        if cached is not None:
            self._smiles_token_cache.move_to_end(iso_smiles)
            return cached
        if self._stats.enabled:
            self._stats.n_smiles_encode_calls += 1
        smiles_str = f"[SMILES]{iso_smiles}[/SMILES]"
        tokens = self._tokenizer.encode(smiles_str, add_special_tokens=False)
        # Cache only non-empty tokens.
        if tokens:
            self._smiles_token_cache[iso_smiles] = tokens
            if len(self._smiles_token_cache) > self._smiles_cache_max:
                self._smiles_token_cache.popitem(last=False)
        return tokens

    def _read_isomer_units_from_pair(
        self,
        fps: List[BinaryIO],
        pair: Tuple[int, int],
    ) -> List[List[int]]:
        start = time.perf_counter()
        fi, li = pair
        raw = read_line_at(fps[fi], int(self.idxs[fi][li]))
        if not raw:
            return []

        try:
            obj = _json_loads(raw)
        except Exception:
            return []

        isomers = obj.get("isomers") or {}
        if not isinstance(isomers, dict):
            return []

        geom_id = obj.get("geom_id")
        units: List[List[int]] = []
        for iso_smiles, conf_list in isomers.items():
            if not conf_list:
                continue
            if self._stats.enabled:
                self._stats.n_conformers_seen += len(conf_list)
            embedded_strings = []
            for conf in conf_list:
                embedded = (conf.get("embedded_smiles") or "").strip()
                if embedded:
                    embedded_strings.append(embedded)
            if not embedded_strings:
                continue
            iso_smiles = (iso_smiles or "").strip()
            if not iso_smiles:
                continue
            if self._stats.enabled:
                self._stats.n_isomers_seen += 1
                self._stats.n_conformers_kept += len(embedded_strings)
            encode_start = time.perf_counter()
            chunks = _encode_isomer_chunks(
                iso_smiles,
                embedded_strings,
                self._tokenizer,
                self.seq_len,
                stats=self._stats if self._stats.enabled else None,
                smiles_tokens=self._get_smiles_tokens(iso_smiles),
            )
            if self._stats.enabled:
                self._stats.t_encode_isomer_chunks += time.perf_counter() - encode_start
            if chunks:
                if self._stats.enabled:
                    self._stats.n_chunks_emitted += len(chunks)
                units.extend(chunks)
        if self._stats.enabled:
            self._stats.t_read_isomer_units += time.perf_counter() - start
        return units

    def _pack_from_pairs(
        self,
        worker_pairs: List[Tuple[int, int]],
        fps: List[BinaryIO],
        preview_enabled: bool,
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            self._ensure_pending_units(worker_pairs, fps)
            if not self._pending_units:
                break
            unit = self._select_fitting_unit()
            if unit:
                self._sequence_state.append_unit(unit)
                continue
            if self._sequence_state.has_content():
                yield self._finalize_with_logging(preview_enabled)
                continue
            if self._evict_monster_unit():
                continue
            break
        if self._sequence_state.has_content():
            yield self._finalize_with_logging(preview_enabled)

    def _finalize_with_logging(
        self,
        preview_enabled: bool,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inp, lab = self._sequence_state.finalize()
        # Only log previews from rank 0 to avoid noisy multi-rank logs.
        if preview_enabled and self._preview is not None and self.rank == 0:
            self._preview.maybe_log(self.rank, inp)
        self._stats.record_flush(self.rank, self.seq_len)
        return {"input": inp}, lab

    def _fill_isomer_unit_buffer(
        self,
        worker_pairs: List[Tuple[int, int]],
        fps: List[BinaryIO],
    ) -> None:
        start = time.perf_counter()
        was_empty = not self._iso_unit_buffer
        target_fill = self.shuffle_buffer_size
        # For isomer_units, cap the fill size to lookahead_limit to avoid
        # large upfront stalls when conformers are heavy to tokenize.
        if self.lookahead_limit:
            target_fill = min(target_fill, self.lookahead_limit)
        pairs_processed = 0
        while len(self._iso_unit_buffer) < target_fill:
            if (
                self._iso_refill_max_pairs > 0
                and pairs_processed >= self._iso_refill_max_pairs
            ):
                break
            if self._iso_refill_max_sec > 0:
                if (time.perf_counter() - start) >= self._iso_refill_max_sec:
                    break
            self._fill_pair_buffer(worker_pairs)
            if not self._pair_buffer:
                break
            pair = self._pair_buffer.popleft()
            units = self._read_isomer_units_from_pair(fps, pair)
            if units:
                self._iso_unit_buffer.extend(units)
            pairs_processed += 1
        if self.shuffle_lines and was_empty and self._iso_unit_buffer:
            self._rng.shuffle(self._iso_unit_buffer)
        if self._stats.enabled:
            self._stats.t_fill_isomer_buffer += time.perf_counter() - start

    def _next_isomer_unit(
        self,
        worker_pairs: List[Tuple[int, int]],
        fps: List[BinaryIO],
    ) -> Optional[List[int]]:
        if not self._iso_unit_buffer:
            self._fill_isomer_unit_buffer(worker_pairs, fps)
        if not self._iso_unit_buffer:
            return None
        return self._iso_unit_buffer.pop()

    def _finalize_isomer_sample(
        self,
        preview_enabled: bool,
        packed: Tuple[Dict[str, torch.Tensor], torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        features, labels = packed
        if preview_enabled and self._preview is not None and self.rank == 0:
            self._preview.maybe_log(self.rank, features["input"])
        self._stats.record_flush(self.rank, self.seq_len)
        return features, labels

    def _pack_isomer_units(
        self,
        worker_pairs: List[Tuple[int, int]],
        fps: List[BinaryIO],
        preview_enabled: bool,
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        if self._iso_packer is None:
            raise RuntimeError("Isomer unit packer is not initialized.")
        while True:
            unit_ids = self._next_isomer_unit(worker_pairs, fps)
            if unit_ids is None:
                break
            for packed in self._iso_packer.add_unit(unit_ids):
                yield self._finalize_isomer_sample(preview_enabled, packed)
        for packed in self._iso_packer.finalize():
            yield self._finalize_isomer_sample(preview_enabled, packed)

# ---------- Titan factory ----------
def build_dataloader(
    train_path: Union[str, Sequence[str]],
    tokenizer_path: str,
    seq_len: int,
    batch_size: int,
    *, tokenizer=None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_lines: bool = True,
    infinite: bool = True,
    seed: Optional[int] = None,
    min_emb_len: int = 16,
    drop_last: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    lookahead_limit: int = 100,
    truncate_overflow_units: bool = True,
    preview_enabled: bool = True,
    serialization_mode: str = "pairs",
    emit_attention_mask: bool = False,
    shuffle_buffer_size: Optional[int] = None,
    stats_tag: Optional[str] = None,
    stats_log_every: int = 1000,
):
    ds = JsonlTaggedPackedDataset(
        train_path=train_path,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        min_emb_len=min_emb_len,
        shuffle_lines=shuffle_lines,
        infinite=infinite,
        seed=seed if seed is not None else 1234,
        world_size=world_size,
        rank=rank,
        lookahead_limit=lookahead_limit,
        truncate_overflow_units=truncate_overflow_units,
        preview_enabled=preview_enabled,
        serialization_mode=serialization_mode,
        emit_attention_mask=emit_attention_mask,
        shuffle_buffer_size=shuffle_buffer_size or 4096,
        stats_tag=stats_tag,
        stats_log_every=stats_log_every,
    )
    effective_persistent = (
        persistent_workers if persistent_workers is not None else (num_workers > 0)
    )
    # prefetch_factor must be None when num_workers=0
    effective_prefetch = (
        (prefetch_factor if prefetch_factor is not None else 4)
        if num_workers > 0
        else None
    )
    return TitanStatefulDataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=effective_persistent,
        prefetch_factor=effective_prefetch,
    )


def _resolve_train_path(
    data_cfg: MolGenDataConfig, job_config: MolGenJobConfig
) -> str:
    train_path = data_cfg.train_path or getattr(job_config.training, "dataset_path", "")
    if not train_path:
        raise ValueError(
            "MolGen dataloader requires a train_path. "
            "Set molgen_data.train_path_key or training.dataset_path."
        )
    return train_path


def _resolve_tokenizer_path(
    data_cfg: MolGenDataConfig, job_config: MolGenJobConfig
) -> str:
    tokenizer_path = data_cfg.tokenizer_override or getattr(
        job_config.model, "hf_assets_path", ""
    )
    if not tokenizer_path:
        raise ValueError(
            "Tokenizer path is empty. "
            "Set model.hf_assets_path alias or molgen_data.tokenizer_override."
        )
    return tokenizer_path


def build_molgen_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,
    job_config: MolGenJobConfig,
) -> BaseDataLoader:
    """
    TorchTitan dataloader factory that reuses the JSONL packer implemented in-house.
    """

    data_cfg = getattr(job_config, "molgen_data", None)
    if data_cfg is None:
        raise ValueError(
            "Missing 'molgen_data' section in the job config. "
            "Set job.custom_config_module="
            "'molgen3D.training.pretraining.config.custom_job_config'."
        )

    train_path = _resolve_train_path(data_cfg, job_config)
    tokenizer_path = _resolve_tokenizer_path(data_cfg, job_config)

    return build_dataloader(
        train_path=train_path,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        seq_len=job_config.training.seq_len,
        batch_size=job_config.training.local_batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        shuffle_lines=data_cfg.shuffle_lines,
        infinite=data_cfg.infinite,
        seed=data_cfg.seed if data_cfg.seed is not None else job_config.training.seed,
        min_emb_len=data_cfg.min_emb_len,
        drop_last=data_cfg.drop_last,
        persistent_workers=data_cfg.persistent_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        world_size=dp_world_size,
        rank=dp_rank,
        lookahead_limit=data_cfg.lookahead_limit,
        preview_enabled=True,
        serialization_mode=data_cfg.serialization_mode,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
        stats_tag="train",
    )


def _resolve_validation_path(job_config: MolGenJobConfig) -> str:
    validation_path = getattr(job_config.validation, "dataset_path", "")
    if not validation_path:
        raise ValueError(
            "MolGen validation requires a dataset_path. "
            "Set validation.dataset_path (e.g. molgen_data validation alias "
            "in paths.yaml)."
        )
    return validation_path


if Validator is not None:
    class MolGenValidator(Validator):
        """
        Base MolGen validator that wraps TorchTitan's Validator with dataloader setup.
        
        This class handles the basic validation dataloader setup. For extended functionality
        like numerical conformer validation, see helpers/validator.py which extends this class.
        """
        def __init__(
            self,
            job_config: MolGenJobConfig,
            dp_world_size: int,
            dp_rank: int,
            tokenizer,
            parallel_dims,
            loss_fn,
            validation_context,
            maybe_enable_amp,
            metrics_processor,
            validation_dataloader: BaseDataLoader,
            pp_schedule=None,
            pp_has_first_stage=None,
            pp_has_last_stage=None,
        ):
            self.job_config = job_config
            self.tokenizer = tokenizer
            self.parallel_dims = parallel_dims
            self.loss_fn = loss_fn
            self.validation_dataloader = validation_dataloader
            self.validation_context = validation_context
            self.maybe_enable_amp = maybe_enable_amp
            self.metrics_processor = metrics_processor
            self.pp_schedule = pp_schedule
            self.pp_has_first_stage = pp_has_first_stage
            self.pp_has_last_stage = pp_has_last_stage

            if self.job_config.validation.steps == -1 and titan_logger is not None:
                titan_logger.warning(
                    "Setting validation steps to -1 might cause hangs because of "
                    "unequal sample counts across ranks when dataset is exhausted."
                )

    MolGenValidatorClass = MolGenValidator


def build_molgen_validator(
    job_config: MolGenJobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer,
    parallel_dims,
    loss_fn,
    validation_context,
    maybe_enable_amp,
    metrics_processor,
    pp_schedule=None,
    pp_has_first_stage=None,
    pp_has_last_stage=None,
    validator_class=None,
) -> BaseValidator:
    """
    Build a MolGen validator with the validation dataloader.
    
    Args:
        validator_class: Optional custom validator class to use instead of the base MolGenValidator.
                        This allows extended validators (like the numerical validator) to reuse
                        the dataloader setup logic.
    """
    if MolGenValidatorClass is None:
        raise RuntimeError(
            "Torchtitan validator bindings are unavailable. Install torchtitan "
            "and ensure the environment exposes torchtitan.components.validate."
        )

    data_cfg = getattr(job_config, "molgen_data", None)
    if data_cfg is None:
        raise ValueError(
            "Missing 'molgen_data' section in the job config. "
            "Set job.custom_config_module="
            "'molgen3D.training.pretraining.config.custom_job_config'."
        )

    # Keep validation lightweight; for very small validation steps, single-process
    # loading avoids multi-worker startup overhead.
    validation_steps = int(getattr(job_config.validation, "steps", 0))
    if 0 < validation_steps <= 10:
        val_num_workers = 0
    else:
        val_num_workers = min(data_cfg.num_workers, 8)
    infinite_validation = job_config.validation.steps != -1
    validation_path = _resolve_validation_path(job_config)
    # Prebuild .idx files for validation (single-file validation is common).
    try:
        val_files = expand_paths([validation_path])
    except Exception:
        val_files = []
    if dp_rank == 0 and val_files:
        try:
            for path in val_files:
                build_line_index(path)
        except Exception:
            pass
    if val_files and dist.is_available() and dist.is_initialized():
        # Ensure all ranks see the prebuilt indices before constructing the dataset.
        dist.barrier()
    elif val_files and dp_rank != 0:
        # Non-distributed fallback: wait briefly for rank 0 to finish writing .idx.
        deadline = time.time() + 300
        while time.time() < deadline:
            if all(idx_path(path).exists() for path in val_files):
                break
            time.sleep(0.5)

    validation_dataloader = build_dataloader(
        train_path=validation_path,
        tokenizer_path=_resolve_tokenizer_path(data_cfg, job_config),
        tokenizer=tokenizer,
        seq_len=job_config.validation.seq_len,
        batch_size=job_config.validation.local_batch_size,
        num_workers=val_num_workers,
        pin_memory=data_cfg.pin_memory,
        shuffle_lines=False,
        # Mirror TorchTitan's default: only allow finite validation when the user
        # explicitly sets steps=-1, otherwise keep the loader infinite so every
        # rank can always advance to the requested step count.
        infinite=infinite_validation,
        seed=data_cfg.seed if data_cfg.seed is not None else job_config.training.seed,
        min_emb_len=data_cfg.min_emb_len,
        drop_last=False,
        persistent_workers=False if val_num_workers == 0 else data_cfg.persistent_workers,
        prefetch_factor=(
            None if val_num_workers == 0 else data_cfg.prefetch_factor
        ),
        world_size=dp_world_size,
        rank=dp_rank,
        preview_enabled=False,
        lookahead_limit=data_cfg.lookahead_limit,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
        serialization_mode=data_cfg.serialization_mode,
        emit_attention_mask=getattr(data_cfg, "emit_attention_mask", False),
        stats_tag="val",
    )

    # Use custom validator class if provided, otherwise use base MolGenValidator
    cls = validator_class if validator_class is not None else MolGenValidatorClass

    return cls(  # type: ignore[arg-type]
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        validation_dataloader=validation_dataloader,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )


