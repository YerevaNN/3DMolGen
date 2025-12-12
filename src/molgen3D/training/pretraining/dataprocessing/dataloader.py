# pretraining/data/dataloaders/jsonl_tagged_packed_simple.py
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cloudpickle
import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict, deque
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

from molgen3D.training.pretraining.config.custom_job_config import (
    JobConfig as MolGenJobConfig,
    MolGenDataConfig,
)
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.evaluation.utils import extract_between
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
    read_line_at,
)
from molgen3D.utils.utils import get_best_rmsd

try:  # optional faster JSON parser
    import orjson as _fast_json
except Exception:  # pragma: no cover
    _fast_json = None

NUM_NUMERICAL_VALIDATION_PROMPTS = 200
PRETOKENIZED_PROMPTS_PATH = Path("/auto/home/vover/3DMolGen/data/pretokenized_prompts.json")
VALIDATION_PICKLE_PATH = Path("/auto/home/vover/3DMolGen/data/valid_set.pickle")


def _is_primary_rank() -> bool:
    """
    Return True only for the primary logging rank to avoid duplicated work.
    """

    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _json_loads(raw: str):
    if _fast_json is not None:
        return _fast_json.loads(raw.encode("utf-8"))
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
    token_id = getattr(tokenizer, attr_name, None)
    if token_id is not None:
        try:
            return int(token_id)
        except (TypeError, ValueError):
            pass

    for token in fallback_tokens:
        if not token:
            continue
        converted = tokenizer.convert_tokens_to_ids(token)
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
        self._sequence_state = SequenceState(
            seq_len=self.seq_len,
            separator_id=self.sep_id,
            pad_id=self.pad_id,
            ignore_index=self.ignore_index,
        )

    def _pairs_for_epoch(self) -> List[Tuple[int, int]]:
        indices = list(self._all_pair_indices)
        if self.shuffle_lines:
            self._rng.shuffle(indices)
        pairs: List[Tuple[int, int]] = []
        for global_idx in indices:
            if (global_idx % self.world_size) != self.rank:
                continue
            pairs.append(self._all_pairs[global_idx])
        return pairs

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1

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
                yield from self._pack_from_pairs(worker_pairs, fps, preview_enabled)

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
        return {
            "epoch": self._epoch,
            "rng_state": self._rng.getstate(),
            "start_k": self._start_k,
            "pairs_total": self._pairs_total,
            "pair_buffer": list(self._pair_buffer),
            "pending_units": [unit.tokens for unit in self._pending_units],
            "sequence_tokens": sequence_tokens,
        }

    def load_state_dict(self, s: Dict):
        self._epoch = int(s.get("epoch", 0))
        rng_state = s.get("rng_state")
        self._rng.setstate(_coerce_rng_state(rng_state))
        self._start_k = int(s.get("start_k", 0))
        self._pairs_total = int(s.get("pairs_total", 0))
        self._pair_buffer = deque(tuple(pair) for pair in s.get("pair_buffer", []))
        pending = s.get("pending_units", [])
        self._pending_units = [
            PendingUnit(tokens=list(tokens), total_len=len(tokens) + 1)
            for tokens in pending
        ]
        if self._sequence_state is None:
            self._ensure_tokenizer_ready()
        self._sequence_state.load_tokens(s.get("sequence_tokens", []))
        self._pair_cursor = self._start_k
        self._monster_warning_shown = False
        self._truncation_warning_shown = False

    def _reset_epoch_state(self) -> None:
        self._pair_buffer.clear()
        self._pending_units = []
        if self._sequence_state is not None:
            self._sequence_state.reset()
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
        return {"input": inp}, lab

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
    )
    effective_persistent = (
        persistent_workers if persistent_workers is not None else (num_workers > 0)
    )
    effective_prefetch = (
        prefetch_factor
        if prefetch_factor is not None
        else (4 if num_workers > 0 else None)
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
            self._conformer_start_id = self._resolve_token_id("[CONFORMERS]", "[CONFORMER]")
            self._conformer_end_id = self._resolve_token_id(
                "[/CONFORMERS]", "[/CONFORMER]"
            )
            self._eos_id = _resolve_special_token_id(
                tokenizer,
                "eos_id",
                (
                    getattr(tokenizer, "eos_token", None),
                    "<|endoftext|>",
                ),
            )
            self._pad_id = _resolve_special_token_id(
                tokenizer,
                "pad_id",
                (
                    getattr(tokenizer, "pad_token", None),
                    "<|endoftext|>",
                ),
            )

            if self.job_config.validation.steps == -1 and titan_logger is not None:
                titan_logger.warning(
                    "Setting validation steps to -1 might cause hangs because of "
                    "unequal sample counts across ranks when dataset is exhausted."
                )

        def _resolve_token_id(self, *tokens: str) -> Optional[int]:
            for token in tokens:
                if not token:
                    continue
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    return int(token_id)
            return None

        def _build_prompt_tensor(
            self, token_ids: Sequence[Union[int, float]], device: torch.device
        ) -> Optional[torch.Tensor]:
            ids: List[int] = []
            for tid in token_ids:
                try:
                    ids.append(int(tid))
                except (TypeError, ValueError):
                    continue

            if self._conformer_start_id is None:
                logger.warning(
                    "Skipping numerical validation because conformer start token is missing."
                )
                return None

            ids.append(self._conformer_start_id)
            return torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)

        def _greedy_decode(
            self,
            model: torch.nn.Module,
            prompt: torch.Tensor,
            max_new_tokens: int,
            max_total_len: int,
        ) -> torch.Tensor:
            generated = prompt
            for _ in range(max_new_tokens):
                logits = model(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

                token_id = int(next_token.item())
                if self._conformer_end_id is not None and token_id == self._conformer_end_id:
                    break
                if self._eos_id is not None and token_id == self._eos_id:
                    break
                if generated.shape[1] >= max_total_len:
                    break
            return generated

        def _extract_conformer_text(self, token_tensor: torch.Tensor) -> str:
            decoded = self.tokenizer.decode(
                token_tensor[0].tolist(), skip_special_tokens=False
            )
            conformer = extract_between(decoded, "[CONFORMERS]", "[/CONFORMERS]")
            if not conformer:
                conformer = extract_between(decoded, "[CONFORMER]", "[/CONFORMER]")
            return conformer.strip()

        def _best_rmsd(self, generated_mol, ground_truths: Sequence) -> float:
            rmsds: List[float] = []
            for gt in ground_truths:
                try:
                    rmsds.append(float(get_best_rmsd(generated_mol, gt, use_alignmol=False)))
                except Exception:
                    rmsds.append(float("nan"))
            if not rmsds:
                return float("nan")
            return float(np.nanmin(rmsds))

        def _load_prompts(self) -> List[Tuple[str, List[int]]]:
            if not PRETOKENIZED_PROMPTS_PATH.exists():
                logger.warning(
                    f"Numerical validation prompts file not found at {PRETOKENIZED_PROMPTS_PATH}"
                )
                return []
            try:
                with open(PRETOKENIZED_PROMPTS_PATH, "r") as fh:
                    payload = json.load(fh)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning(
                    f"Failed to load pretokenized prompts from {PRETOKENIZED_PROMPTS_PATH}: {exc}"
                )
                return []

            if not isinstance(payload, dict):
                logger.warning(
                    f"Pretokenized prompts should be a dict of SMILES->token list, got {type(payload)}"
                )
                return []

            items = sorted(payload.items(), key=lambda kv: kv[0])[
                :NUM_NUMERICAL_VALIDATION_PROMPTS
            ]
            normalized: List[Tuple[str, List[int]]] = []
            for key, value in items:
                if isinstance(value, list):
                    try:
                        normalized.append((key, [int(v) for v in value]))
                    except Exception:
                        normalized.append((key, []))
                else:
                    normalized.append((key, []))
            return normalized

        def _load_ground_truths(self) -> Dict[str, List]:
            if not VALIDATION_PICKLE_PATH.exists():
                logger.warning(
                    f"Ground truth conformers not found at {VALIDATION_PICKLE_PATH}"
                )
                return {}
            try:
                with open(VALIDATION_PICKLE_PATH, "rb") as fh:
                    data = cloudpickle.load(fh)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning(
                    f"Failed to load ground truth conformers from {VALIDATION_PICKLE_PATH}: {exc}"
                )
                return {}

            if not isinstance(data, dict):
                logger.warning(
                    f"Expected dict[str, list[Mol]] in {VALIDATION_PICKLE_PATH}, got {type(data)}"
                )
                return {}
            return data

        def _log_numerical_metrics(
            self, rmsd_values: List[float], failures: int, step: int
        ) -> None:
            valid = np.array(rmsd_values, dtype=float)
            valid = valid[~np.isnan(valid)]
            metrics: Dict[str, float] = {
                "numerical_val/failures": float(failures),
                "numerical_val/successes": float(valid.size),
            }
            if valid.size > 0:
                metrics.update(
                    {
                        "numerical_val/rmsd_min": float(np.nanmin(valid)),
                        "numerical_val/rmsd_max": float(np.nanmax(valid)),
                        "numerical_val/rmsd_mean": float(np.nanmean(valid)),
                        "numerical_val/rmsd_std": float(np.nanstd(valid)),
                    }
                )

            successes = int(metrics.get("numerical_val/successes", 0))
            suffix = (
                f" | rmsd min={metrics.get('numerical_val/rmsd_min', float('nan')):.4f}"
                f" mean={metrics.get('numerical_val/rmsd_mean', float('nan')):.4f}"
                f" std={metrics.get('numerical_val/rmsd_std', float('nan')):.4f}"
                f" max={metrics.get('numerical_val/rmsd_max', float('nan')):.4f}"
                if valid.size > 0
                else ""
            )
            logger.info(
                f"Numerical validation (step {step}): successes={successes} failures={failures}{suffix}"
            )

            try:  # best effort W&B logging
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except ModuleNotFoundError:
                logger.info("W&B not installed; skipping numerical validation logging.")
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning(f"Failed to log numerical metrics to W&B: {exc}")

        def _run_numerical_validation(
            self, model_parts: List[torch.nn.Module], step: int
        ) -> None:
            if not getattr(self.job_config.validation, "numerical_validation", False):
                return
            if not _is_primary_rank():
                return
            if self.parallel_dims.pp_enabled or len(model_parts) != 1:
                logger.warning(
                    "Numerical validation currently supports single-stage models; skipping."
                )
                return
            if self._conformer_start_id is None or self._conformer_end_id is None:
                logger.warning(
                    "Conformer tokens missing from tokenizer; numerical validation skipped."
                )
                return

            prompts = self._load_prompts()
            ground_truths = self._load_ground_truths()
            if not prompts or not ground_truths:
                return

            model = model_parts[0]
            device = next(model.parameters()).device
            max_seq_len = int(
                getattr(self.job_config.validation, "seq_len", 2048)
                or getattr(self.job_config.training, "seq_len", 2048)
                or 2048
            )

            rmsd_values: List[float] = []
            failures = 0
            was_training = model.training
            model.eval()

            with torch.inference_mode():
                for key, token_ids in prompts:
                    gt_confs = ground_truths.get(key)
                    if not gt_confs:
                        failures += 1
                        continue

                    prompt_tensor = self._build_prompt_tensor(token_ids, device)
                    if prompt_tensor is None:
                        failures += 1
                        continue

                    available = max(max_seq_len - prompt_tensor.shape[1], 1)
                    max_new_tokens = min(512, available)

                    generated_ids = self._greedy_decode(
                        model, prompt_tensor, max_new_tokens, max_seq_len
                    )
                    conformer_text = self._extract_conformer_text(generated_ids)
                    if not conformer_text:
                        failures += 1
                        continue

                    try:
                        generated_mol = decode_cartesian_raw(conformer_text)
                    except Exception:
                        failures += 1
                        continue

                    rmsd_val = self._best_rmsd(generated_mol, gt_confs)
                    if np.isnan(rmsd_val):
                        failures += 1
                    else:
                        rmsd_values.append(rmsd_val)

            if was_training:
                model.train()

            self._log_numerical_metrics(rmsd_values, failures, step)

        @torch.no_grad()
        def validate(
            self,
            model_parts: list[torch.nn.Module],
            step: int,
        ) -> None:
            super().validate(model_parts, step)
            try:
                self._run_numerical_validation(model_parts, step)
            except Exception as exc:  # pragma: no cover - safety
                logger.warning("Numerical validation failed: %s", exc)

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
) -> BaseValidator:
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

    # Use fewer workers for validation to reduce memory usage
    # Validation doesn't need as many workers as training since it's not as performance-critical
    val_num_workers = min(data_cfg.num_workers, 2)  # Cap at 2 workers for validation
    infinite_validation = job_config.validation.steps != -1
    validation_dataloader = build_dataloader(
        train_path=_resolve_validation_path(job_config),
        tokenizer_path=_resolve_tokenizer_path(data_cfg, job_config),
        tokenizer=tokenizer,
        seq_len=job_config.validation.seq_len,
        batch_size=job_config.validation.local_batch_size,
        num_workers=val_num_workers,
        pin_memory=data_cfg.pin_memory,
        shuffle_lines=False,
        # Mirror TorchTitan’s default: only allow finite validation when the user
        # explicitly sets steps=-1, otherwise keep the loader infinite so every
        # rank can always advance to the requested step count.
        infinite=infinite_validation,
        seed=data_cfg.seed if data_cfg.seed is not None else job_config.training.seed,
        min_emb_len=data_cfg.min_emb_len,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=min(data_cfg.prefetch_factor or 2, 2),  # Reduce prefetch for validation
        world_size=dp_world_size,
        rank=dp_rank,
        preview_enabled=False,
    )

    return MolGenValidatorClass(  # type: ignore[arg-type]
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
