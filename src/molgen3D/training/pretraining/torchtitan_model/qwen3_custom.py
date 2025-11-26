from __future__ import annotations

import os
from dataclasses import replace

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.components import metrics as titan_metrics
from torchtitan.components.checkpoint import AsyncMode, CheckpointManager, MODEL
from torchtitan.models import qwen3 as qwen3_module
from torchtitan.protocols.train_spec import register_train_spec
from torchtitan.tools.logging import logger
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter

from molgen3D.training.pretraining.dataprocessing.dataloader import (
    build_molgen_dataloader,
    build_molgen_validator,
)
from molgen3D.training.pretraining.helpers.wsds_scheduler import (
    build_wsds_lr_schedulers,
)


def _is_log_rank() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


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


_WANDB_MODULE = None


def _maybe_import_wandb():
    global _WANDB_MODULE
    if _WANDB_MODULE is not None:
        return _WANDB_MODULE
    try:
        import wandb  # type: ignore[import]
    except ModuleNotFoundError:
        _log_rank(
            "wandb package is not installed; disabling W&B logging for this run."
        )
        _WANDB_MODULE = None
        return None
    _WANDB_MODULE = wandb
    return _WANDB_MODULE


def _maybe_extend_embeddings(model, job_config) -> None:
    target_vocab = getattr(job_config.model, "vocab_size_override", None)
    if not target_vocab:
        return

    embed = getattr(model, "tok_embeddings", None)
    output = getattr(model, "output", None)
    if embed is None or output is None:
        return

    current = int(embed.num_embeddings)
    requested = int(target_vocab)
    base_vocab = getattr(job_config.model, "tokenizer_base_vocab_size", requested)
    added_tokens = getattr(
        job_config.model,
        "tokenizer_added_tokens",
        max(0, requested - base_vocab),
    )
    desired = max(current, requested)
    if desired == current:
        model.model_args.vocab_size = current
        _log_rank(
            "Tokenizer vocabulary (%d total | base=%d | added=%d) already matches model embeddings (%d). No resize.",
            requested,
            base_vocab,
            added_tokens,
            current,
        )
        return

    device = embed.weight.device
    dtype = embed.weight.dtype
    hidden_dim = embed.embedding_dim
    new_rows = desired - current

    new_embed = nn.Embedding(
        desired,
        hidden_dim,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        new_embed.weight[:current].copy_(embed.weight)
        nn.init.trunc_normal_(
            new_embed.weight[current:],
            mean=0.0,
            std=model.model_args.dim ** -0.5,
        )
    model.tok_embeddings = new_embed
    model.model_args.vocab_size = desired

    new_output = nn.Linear(
        output.in_features,
        desired,
        bias=False,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        new_output.weight[:current].copy_(output.weight)
        if not model.model_args.enable_weight_tying:
            final_out_std = model.model_args.dim ** -0.5
            nn.init.trunc_normal_(
                new_output.weight[current:],
                mean=0.0,
                std=final_out_std,
            )

    if model.model_args.enable_weight_tying:
        new_output.weight = model.tok_embeddings.weight

    model.output = new_output
    _log_rank(
        "Tokenizer vocabulary expanded from %d -> %d tokens (base=%d, added=%d, hidden=%d, tied=%s). "
        "Initialized %d new rows using truncated normal.",
        current,
        desired,
        base_vocab,
        added_tokens,
        hidden_dim,
        model.model_args.enable_weight_tying,
        new_rows,
    )


def _parallelize_with_resize(model, parallel_dims, job_config):
    _maybe_extend_embeddings(model, job_config)
    if _is_log_rank():
        embed_rows = int(getattr(model.tok_embeddings, "num_embeddings", -1))
        hidden_dim = int(getattr(model.tok_embeddings, "embedding_dim", -1))
        _log_rank(
            "Tokenizer ready: total=%d | base=%d | added=%d | embedding rows=%d | hidden=%d | tied=%s",
            getattr(job_config.model, "vocab_size_override", embed_rows),
            getattr(job_config.model, "tokenizer_base_vocab_size", embed_rows),
            getattr(job_config.model, "tokenizer_added_tokens", 0),
            embed_rows,
            hidden_dim,
            model.model_args.enable_weight_tying,
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
    return qwen3_module.parallelize_qwen3(model, parallel_dims, job_config)


def _patch_metric_logging() -> None:
    if getattr(titan_metrics, "_molgen_logger_patched", False):
        return

    def _build_metric_logger(job_config, parallel_dims, tag=None):
        metrics_config = job_config.metrics
        has_logging = (
            metrics_config.enable_tensorboard or metrics_config.enable_wandb
        )
        should_log = has_logging
        if has_logging and not metrics_config.save_for_all_ranks:
            metrics_rank = titan_metrics._get_metrics_rank(
                parallel_dims, job_config
            )
            should_log = torch.distributed.get_rank() == metrics_rank

        if not should_log:
            return titan_metrics.BaseLogger()

        dump_dir = job_config.job.dump_folder
        base_log_dir = os.path.join(
            dump_dir, metrics_config.save_tb_folder or "tb"
        )

        if job_config.fault_tolerance.enable:
            base_log_dir = os.path.join(
                base_log_dir,
                f"replica_{job_config.fault_tolerance.replica_id}",
            )

        if metrics_config.save_for_all_ranks:
            base_log_dir = os.path.join(
                base_log_dir, f"rank_{torch.distributed.get_rank()}"
            )

        logger_container = titan_metrics.LoggerContainer()

        if metrics_config.enable_wandb:
            if _maybe_import_wandb() is not None:
                wandb_logger = titan_metrics.WandBLogger(
                    base_log_dir, job_config, tag
                )
                logger_container.add_logger(wandb_logger)
            else:
                _log_rank(
                    "Skipping W&B logger because wandb is unavailable in this environment."
                )

        if metrics_config.enable_tensorboard:
            tensorboard_logger = titan_metrics.TensorBoardLogger(
                base_log_dir, tag
            )
            logger_container.add_logger(tensorboard_logger)

        return logger_container

    titan_metrics._build_metric_logger = _build_metric_logger

    original_wandb_init = titan_metrics.WandBLogger.__init__

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


_PATCHED_CHECKPOINTER = False


def _patch_checkpoint_manager() -> None:
    global _PATCHED_CHECKPOINTER
    if _PATCHED_CHECKPOINTER:
        return

    original_save = CheckpointManager.save

    def _save(self, curr_step: int, last_step: bool = False):
        should_export = getattr(self, "_molgen_save_hf", False) and self._should_save(
            curr_step, last_step
        )
        original_save(self, curr_step, last_step)
        if should_export:
            self._molgen_export_hf(curr_step)

    def _export_hf(self, curr_step: int):
        if self.sd_adapter is None:
            logger.warning(
                "save_hf_per_checkpoint enabled but no state dict adapter provided."
            )
            return

        checkpoint_id = self._create_checkpoint_id(curr_step)
        export_dir = f"{checkpoint_id}_hf"
        states = self.states[MODEL].state_dict()

        self.dcp_save(
            states,
            checkpoint_id=export_dir,
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=True,
        )

        tokenizer = getattr(self, "_molgen_tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(export_dir)

    CheckpointManager.save = _save
    CheckpointManager._molgen_export_hf = _export_hf  # type: ignore[attr-defined]
    _PATCHED_CHECKPOINTER = True


_patch_metric_logging()
_patch_checkpoint_manager()

_BASE_SPEC = qwen3_module.get_train_spec()

register_train_spec(
    "molgen_qwen3",
    replace(
        _BASE_SPEC,
        build_dataloader_fn=build_molgen_dataloader,
        build_validator_fn=build_molgen_validator,
        build_lr_schedulers_fn=build_wsds_lr_schedulers,
        parallelize_fn=_parallelize_with_resize,
        state_dict_adapter=MolGenQwen3StateDictAdapter,
    ),
)
