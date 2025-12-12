from __future__ import annotations

import os
from dataclasses import replace

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.components import metrics as titan_metrics
from torchtitan.models import qwen3 as qwen3_module
from torchtitan.protocols.train_spec import register_train_spec
from torchtitan.tools.logging import logger
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter

from molgen3D.training.pretraining.dataprocessing.dataloader import (
    build_molgen_dataloader,
    build_molgen_validator,
)
from molgen3D.training.pretraining.helpers.validator import (
    build_molgen_validator,
)

from molgen3D.training.pretraining.helpers.wsds_scheduler import (
    build_wsds_lr_schedulers,
)

# NOTE: These are Qwen3-0.6B specific. If you change
# the base model, update these values.
QWEN3_BASE_VOCAB = 151_669
QWEN3_PADDED_VOCAB = 151_936


def _safe_rank(default: int = 0) -> int:
    """Return rank if dist is ready; otherwise fall back to env or a default.

    TorchTitan logger hooks can run before torch.distributed is initialized. Using
    a guarded helper avoids raising in those cases while preserving normal behavior
    once the process group is up.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    env_rank = os.environ.get("RANK")
    if env_rank is not None:
        try:
            return int(env_rank)
        except ValueError:
            return default
    return default


def _is_log_rank() -> bool:
    return _safe_rank() == 0


def _log_rank(msg: str, *args) -> None:
    if _is_log_rank():
        logger.info(msg, *args)


class MolGenQwen3StateDictAdapter(Qwen3StateDictAdapter):
    def from_hf(self, hf_state_dict):
        state_dict = super().from_hf(hf_state_dict)
        tok_key = "tok_embeddings.weight"
        lm_key = "lm_head.weight"
        if lm_key not in state_dict and tok_key in state_dict:
            _log_rank(
                "HF checkpoint missing %s; copying weights from %s.",
                lm_key,
                tok_key,
            )
            state_dict[lm_key] = state_dict[tok_key]
        return state_dict


_WANDB_UNINITIALIZED = object()
_WANDB_MODULE = _WANDB_UNINITIALIZED


def _maybe_import_wandb():
    global _WANDB_MODULE
    if _WANDB_MODULE is not _WANDB_UNINITIALIZED:
        return _WANDB_MODULE
    try:
        import wandb  # type: ignore[import]
    except ModuleNotFoundError:
        _log_rank(
            "wandb package is not installed; disabling W&B logging for this run."
        )
        _WANDB_MODULE = None
    else:
        _WANDB_MODULE = wandb
    return _WANDB_MODULE


def _unwrap_module(module: nn.Module) -> nn.Module:
    # Nested wrappers (DDP(FSDP(...))) can appear; unwrap until the leaf module.
    while hasattr(module, "module"):
        module = module.module
    return module


def _get_job_tokenizer_config(job_config):
    tokenizer_cfg = getattr(job_config, "tokenizer", None)
    if tokenizer_cfg is None:
        raise ValueError("Job config is missing the tokenizer section.")

    base_vocab = int(getattr(tokenizer_cfg, "base_vocab_size", 0))
    padded_vocab = int(getattr(tokenizer_cfg, "padded_vocab_size", 0))
    num_new_tokens = getattr(tokenizer_cfg, "num_new_tokens", None)
    if num_new_tokens is None:
        num_new_tokens = getattr(tokenizer_cfg, "added_tokens", 0)
    num_new_tokens = int(num_new_tokens)
    total_vocab = int(
        getattr(tokenizer_cfg, "total_vocab_size", base_vocab + num_new_tokens)
    )

    if base_vocab != QWEN3_BASE_VOCAB:
        raise ValueError(
            f"Unexpected base_vocab_size: {base_vocab} != {QWEN3_BASE_VOCAB}"
        )
    if padded_vocab != QWEN3_PADDED_VOCAB:
        raise ValueError(
            f"Unexpected padded_vocab_size: {padded_vocab} != {QWEN3_PADDED_VOCAB}"
        )
    if total_vocab != base_vocab + num_new_tokens:
        raise ValueError(
            f"Inconsistent vocab sizes: total={total_vocab}, base+new={base_vocab + num_new_tokens}"
        )

    return base_vocab, padded_vocab, num_new_tokens, total_vocab


def _initialize_extra_embeddings(module: nn.Module, base_vocab: int, num_new_tokens: int) -> None:
    if num_new_tokens <= 0:
        return

    embed = getattr(module, "tok_embeddings", None) or getattr(
        module, "embed_tokens", None
    )
    if embed is None or not hasattr(embed, "weight"):
        return

    start = base_vocab
    end = base_vocab + num_new_tokens
    num_rows = int(getattr(embed, "num_embeddings", embed.weight.shape[0]))
    if end > num_rows:
        raise ValueError(
            "Tokenizer config requests more tokens than embedding rows "
            f"(base_vocab={base_vocab}, num_new_tokens={num_new_tokens}, rows={num_rows})"
        )

    with torch.no_grad():
        init_std = embed.embedding_dim ** -0.5
        torch.nn.init.normal_(embed.weight[start:end], mean=0.0, std=init_std)


def _tie_lm_head(module: nn.Module) -> None:
    embed = getattr(module, "tok_embeddings", None) or getattr(
        module, "embed_tokens", None
    )
    head = getattr(module, "lm_head", None) or getattr(module, "output", None)
    if embed is None or head is None or not hasattr(head, "weight"):
        return
    if head.weight is not embed.weight:
        head.weight = embed.weight


def _parallelize_with_molgen(model, parallel_dims, job_config):
    base_vocab, padded_vocab, num_new_tokens, total_vocab = _get_job_tokenizer_config(
        job_config
    )

    if _is_log_rank():
        embed = getattr(model, "tok_embeddings", None) or getattr(
            model, "embed_tokens", None
        )
        embed_rows = int(getattr(embed, "num_embeddings", -1)) if embed is not None else -1
        hidden_dim = int(getattr(embed, "embedding_dim", -1)) if embed is not None else -1
        _log_rank(
            "Tokenizer ready: total=%d | base=%d | added=%d | embedding rows=%d | hidden=%d | tied=%s",
            total_vocab,
            base_vocab,
            num_new_tokens,
            embed_rows,
            hidden_dim,
            getattr(model.model_args, "enable_weight_tying", False),
        )
        total_params, _ = model.model_args.get_nparams_and_flops(
            model, job_config.training.seq_len
        )
        _log_rank(
            "MolGen Qwen3 summary: model=%s flavor=%s params=%s vocab=%d seq_len=%d dtype=%s",
            job_config.model.name,
            job_config.model.flavor,
            f"{total_params:,}",
            getattr(model.model_args, "vocab_size", -1),
            job_config.training.seq_len,
            job_config.training.dtype,
        )

    trained_model = qwen3_module.parallelize_qwen3(model, parallel_dims, job_config)
    underlying = _unwrap_module(trained_model)

    run_cfg = getattr(job_config, "molgen_run", None)
    init_mode = getattr(run_cfg, "init_mode", "scratch") if run_cfg is not None else "scratch"
    if init_mode in ("scratch", "hf_pretrain"):
        _log_rank(
            "Initializing extra embeddings: init_mode=%s base_vocab=%d num_new_tokens=%d",
            init_mode,
            base_vocab,
            num_new_tokens,
        )
        _initialize_extra_embeddings(underlying, base_vocab, num_new_tokens)
    _tie_lm_head(underlying)

    if _is_log_rank():
        embed = getattr(underlying, "tok_embeddings", None) or getattr(
            underlying, "embed_tokens", None
        )
        if embed is not None:
            wt = getattr(embed, "weight", None)
            if wt is not None:
                is_flat = getattr(wt, "_is_flat_param", False) or (
                    "FlatParameter" in wt.__class__.__name__
                )
                wt_shape = tuple(wt.shape) if hasattr(wt, "shape") else "<unknown>"
                wt_device = getattr(wt, "device", "<unknown>")
                _log_rank(
                    "Embedding weight details: type=%s shape=%s device=%s flat_param=%s",
                    wt.__class__.__name__,
                    wt_shape,
                    wt_device,
                    is_flat,
                )
            rows = int(getattr(embed, "num_embeddings", embed.weight.shape[0]))
            _log_rank(
                "MolGen Qwen3 embeddings: rows=%d base=%d new=%d padded=%d",
                rows,
                base_vocab,
                num_new_tokens,
                padded_vocab,
            )
            if padded_vocab > 0 and rows != padded_vocab:
                _log_rank(
                    "MolGen WARN: embedding rows (%d) differ from padded_vocab_size (%d)",
                    rows,
                    padded_vocab,
                )

            head = getattr(underlying, "lm_head", None) or getattr(
                underlying, "output", None
            )
            if (
                head is not None
                and hasattr(head, "weight")
                and head.weight.data_ptr() != embed.weight.data_ptr()
            ):
                _log_rank(
                    "MolGen WARN: LM head and embeddings do not share weights."
                )

    return trained_model


def _patch_metric_logging() -> None:
    if getattr(titan_metrics, "_molgen_logger_patched", False):
        return

    def _build_metric_logger(job_config, parallel_dims, tag=None):
        metrics_config = job_config.metrics
        has_logging = metrics_config.enable_tensorboard or metrics_config.enable_wandb

        if not has_logging:
            return titan_metrics.BaseLogger()

        if not metrics_config.save_for_all_ranks:
            metrics_rank = titan_metrics._get_metrics_rank(parallel_dims, job_config)
            if _safe_rank() != metrics_rank:
                return titan_metrics.BaseLogger()

        dump_dir = job_config.job.dump_folder
        base_log_dir = os.path.join(dump_dir, metrics_config.save_tb_folder or "tb")

        if job_config.fault_tolerance.enable:
            base_log_dir = os.path.join(
                base_log_dir, f"replica_{job_config.fault_tolerance.replica_id}"
            )

        if metrics_config.save_for_all_ranks:
            base_log_dir = os.path.join(
                base_log_dir, f"rank_{_safe_rank()}"
            )

        logger_container = titan_metrics.LoggerContainer()

        if metrics_config.enable_wandb:
            wandb_mod = _maybe_import_wandb()
            if wandb_mod is not None:
                wandb_logger = titan_metrics.WandBLogger(base_log_dir, job_config, tag)
                logger_container.add_logger(wandb_logger)
            else:
                _log_rank("Skipping W&B logger because wandb is unavailable.")

        if metrics_config.enable_tensorboard:
            tensorboard_logger = titan_metrics.TensorBoardLogger(base_log_dir, tag)
            logger_container.add_logger(tensorboard_logger)

        return logger_container

    titan_metrics._build_metric_logger = _build_metric_logger

    _original_wandb_init = titan_metrics.WandBLogger.__init__

    def _wandb_init(self, log_dir, job_config, tag=None):
        wandb_lib = _maybe_import_wandb()
        if wandb_lib is None:
            raise ModuleNotFoundError("wandb is not installed.")

        self.wandb = wandb_lib
        self.tag = tag

        run_dir = os.getenv("WANDB_DIR", log_dir)
        run_id = os.getenv("WANDB_RUN_ID")
        run_name = os.getenv("RUN_NAME") or os.getenv("WANDB_RUN_NAME")
        entity = os.getenv("WANDB_ENTITY") or os.getenv("WANDB_TEAM")
        project = os.getenv("WANDB_PROJECT", "torchtitan")

        os.makedirs(run_dir, exist_ok=True)
        self.wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            dir=run_dir,
            id=run_id,
            resume="allow" if run_id else None,
            config=job_config.to_dict(),
        )

    titan_metrics.WandBLogger.__init__ = _wandb_init
    titan_metrics._molgen_logger_patched = True


_patch_metric_logging()

_BASE_SPEC = qwen3_module.get_train_spec()

register_train_spec(
    "molgen_qwen3",
    replace(
        _BASE_SPEC,
        build_dataloader_fn=build_molgen_dataloader,
        build_validator_fn=build_molgen_validator,
        build_lr_schedulers_fn=build_wsds_lr_schedulers,
        parallelize_fn=_parallelize_with_molgen,
        state_dict_adapter=MolGenQwen3StateDictAdapter,
    ),
)
