# pretraining/data/dataloaders/jsonl_tagged_packed_simple.py
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from collections import defaultdict
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

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
from molgen3D.training.pretraining.dataprocessing.text_processing import (
    ChunkPacker,
    build_unit,
    is_valid_unit,
)
from molgen3D.training.pretraining.dataprocessing.utils import (
    build_line_index,
    expand_paths,
    read_line_at,
)


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
            decoded = self._tokenizer.decode(
                tensor.tolist(),
                skip_special_tokens=False,
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            decoded = f"<decode_error: {exc}>"

        seq_idx = self._counts[rank] + 1
        logger.info("Rank {} preview sample {}: {}", rank, seq_idx, decoded)
        self._counts[rank] = seq_idx

        if seq_idx >= self._limit:
            self._done.add(rank)


class JsonlTaggedPackedDataset(IterableDataset):
    """
    JSONL -> unit string -> HF tokenize(no specials) -> pack to 2048.
    - BOS at start of each chunk
    - EOS between units
    - no unit splitting; over-long units dropped
    Resume works with torchdata.StatefulDataLoader (if available).
    """
    def __init__(
        self,
        train_path: Union[str, Sequence[str]],
        tokenizer_path: str,
        seq_len: int = 2048,
        min_emb_len: int = 16,
        shuffle_lines: bool = True,
        infinite: bool = True,
        seed: int = 1234,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        super().__init__()
        self.files = _coerce_train_targets(train_path)
        self.idxs = [build_line_index(p) for p in self.files]

        # world/rank from env (torchrun/Titan sets these)
        if world_size is None or rank is None:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.rank = int(os.environ.get("RANK", "0"))
        else:
            self.world_size = world_size
            self.rank = rank

        # tokenizer (HF only for simplicity)
        self.tk = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        self.bos_id = _resolve_special_token_id(
            self.tk,
            "bos_token_id",
            (
                getattr(self.tk, "bos_token", None),
                "<|im_start|>",
                "<s>",
                getattr(self.tk, "cls_token", None),
            ),
        )
        self.eos_id = _resolve_special_token_id(
            self.tk,
            "eos_token_id",
            (
                getattr(self.tk, "eos_token", None),
                "</s>",
                "<|im_end|>",
                "<|endoftext|>",
                getattr(self.tk, "sep_token", None),
            ),
        )
        if self.bos_id is None or self.eos_id is None:
            raise RuntimeError(
                "Tokenizer must expose bos/eos tokens. "
                "Tried common aliases (<|im_start|>, <|im_end|>, etc.) but none were found."
            )

        self.seq_len = int(seq_len)
        self.min_emb_len = int(min_emb_len)
        self.shuffle_lines = bool(shuffle_lines)
        self.infinite = bool(infinite)

        self._epoch = 0
        self._rng = random.Random(seed + self.rank * 97)
        self._packer = ChunkPacker(self.seq_len, self.bos_id, self.eos_id)
        self._start_k = 0     # resume cursor within worker's pair list
        self._pairs_total = 0
        self._preview = _PreviewLogger(self.tk)

    def _pairs_for_epoch(self) -> List[Tuple[int,int]]:
        pairs: List[Tuple[int,int]] = []
        base = 0
        for fi, idx in enumerate(self.idxs):
            n = len(idx)
            order = list(range(n))
            if self.shuffle_lines:
                self._rng.shuffle(order)
            for li in order:
                g = base + li
                if (g % self.world_size) == self.rank:
                    pairs.append((fi, li))
            base += n
        return pairs

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1

        while True:
            all_pairs = self._pairs_for_epoch()
            worker_pairs = [p for i, p in enumerate(all_pairs) if (i % nworkers) == wid]
            self._pairs_total = len(worker_pairs)
            k = self._start_k

            fps = [open(p, "rb") for p in self.files]
            try:
                while k < self._pairs_total:
                    fi, li = worker_pairs[k]
                    k += 1
                    self._start_k = k

                    raw = read_line_at(fps[fi], int(self.idxs[fi][li]))
                    if not raw:
                        continue

                    tokens = self._extract_tokens(raw)
                    if tokens is None:
                        continue

                    preview_enabled = (wid == 0)
                    yield from self._pack_tokens(tokens, preview_enabled)
            finally:
                for f in fps: f.close()

            if not self.infinite:
                break
            self._epoch += 1
            # deterministic but different shuffle each epoch per rank
            self._rng.seed(1337 + self._epoch * 100003 + self.rank * 97)
            self._start_k = 0

    # ------ resume for StatefulDataLoader ------
    def state_dict(self) -> Dict:
        return {
            "epoch": self._epoch,
            "rng_state": self._rng.getstate(),
            "start_k": self._start_k,
            "pairs_total": self._pairs_total,
            "packer": self._packer.state_dict(),
        }

    def load_state_dict(self, s: Dict):
        self._epoch = int(s.get("epoch", 0))
        self._rng.setstate(s.get("rng_state", random.Random().getstate()))
        self._start_k = int(s.get("start_k", 0))
        self._pairs_total = int(s.get("pairs_total", 0))
        self._packer.load_state_dict(s.get("packer", {}))

    def _extract_tokens(self, raw: bytes) -> Optional[List[int]]:
        try:
            obj = json.loads(raw)
        except Exception:
            return None

        canon = (obj.get("canonical_smiles") or "").strip()
        emb = (obj.get("embedded_smiles") or "").strip()
        if not is_valid_unit(canon, emb, min_emb_len=self.min_emb_len):
            return None
        unit = build_unit(canon, emb)
        toks = self.tk.encode(unit, add_special_tokens=False)
        return [self.eos_id] + toks

    def _pack_tokens(
        self, toks: List[int], preview_enabled: bool
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        unit_completed = self._packer.try_add_unit(toks)
        if not unit_completed and self._packer.pending_unit:
            yield from self._flush_blocks(preview_enabled)
        yield from self._flush_blocks(preview_enabled)

    def _flush_blocks(
        self,
        preview_enabled: bool,
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        for inp, lab in self._packer.yield_blocks():
            if preview_enabled:
                self._preview.maybe_log(self.rank, inp)
            yield {"input": inp}, lab

# ---------- Titan factory ----------
def build_dataloader(
    train_path: Union[str, Sequence[str]],
    tokenizer_path: str,
    seq_len: int,
    batch_size: int,
    *,
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
):
    ds = JsonlTaggedPackedDataset(
        train_path=train_path,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        min_emb_len=min_emb_len,
        shuffle_lines=shuffle_lines,
        infinite=infinite,
        seed=seed if seed is not None else 1234,
        world_size=world_size,
        rank=rank,
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

    validation_dataloader = build_dataloader(
        train_path=_resolve_validation_path(job_config),
        tokenizer_path=_resolve_tokenizer_path(data_cfg, job_config),
        seq_len=job_config.validation.seq_len,
        batch_size=job_config.validation.local_batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        shuffle_lines=False,
        infinite=False,
        seed=data_cfg.seed if data_cfg.seed is not None else job_config.training.seed,
        min_emb_len=data_cfg.min_emb_len,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=data_cfg.prefetch_factor,
        world_size=dp_world_size,
        rank=dp_rank,
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
