from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Optional

import sys

import torch
import torch.distributed as dist
import tyro
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer
from transformers import AutoTokenizer
from safetensors.torch import load_file as load_safetensor, save_file as save_safetensor
import shutil

from molgen3D.config import paths
from molgen3D.training.pretraining.torchtitan_model.qwen3_custom import (
    QWEN3_BASE_VOCAB,
    QWEN3_PADDED_VOCAB,
)
from molgen3D.training.pretraining.config.custom_job_config import MolGenRunConfig
from molgen3D.training.pretraining.helpers.wsds_scheduler import set_active_job_config
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    ensure_tokenizer_pad_token,
)

_TEE_ENABLED = False
_RUN_NAME_PREFIX_LEN = len("YYMMDD-HHMM-")


def _current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    env_rank = os.environ.get("RANK")
    if env_rank is not None:
        try:
            return int(env_rank)
        except ValueError:
            return 0
    return 0


def _is_log_rank() -> bool:
    return _current_rank() == 0


def _log_rank(msg: str, *args) -> None:
    if _is_log_rank():
        logger.info(msg, *args)


@dataclass
class QwenPretrainRunConfig:
    # Legacy: run_desc kept for compatibility with older launchers. The primary
    # source of truth is job.description in the TOML.
    run_desc: Optional[str] = None
    train_toml: str = "src/molgen3D/config/pretrain/qwen3_06b.toml"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None


@dataclass
class RunLayout:
    run_name: str
    run_hash: str
    logs_dir: Path
    ckpts_dir: Path
    wandb_dir: Path
    reuse_existing_dirs: bool = False


@dataclass
class TokenizerDetails:
    path: Path
    vocab_size: int
    added_tokens: int
    base_vocab_size: int
    padded_vocab_size: int


def _sanitize_description(description: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", description)
    safe = safe.strip("-_")
    return safe or "run"


def _resolve_tag_or_path(value: str) -> Path:
    if ":" in value:
        return paths.resolve_tag(value)
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (paths.REPO_ROOT / candidate)


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (paths.REPO_ROOT / path)


def _extract_run_hash_from_name(run_name: str) -> Optional[str]:
    if len(run_name) <= _RUN_NAME_PREFIX_LEN:
        return None
    hash_and_rest = run_name[_RUN_NAME_PREFIX_LEN :]
    if "-" not in hash_and_rest:
        return None
    return hash_and_rest.split("-", 1)[0]


def _resolve_resume_ckpt_dir(run_settings: MolGenRunConfig) -> Path:
    if not run_settings.resume_run_path_tag:
        raise ValueError(
            "resume_run_path_tag must be set when init_mode == 'resume'."
        )
    resume_path = _resolve_tag_or_path(run_settings.resume_run_path_tag)
    if not resume_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint path {resume_path} does not exist."
        )
    if resume_path.name.startswith("step"):
        return resume_path.parent
    return resume_path


def _generate_run_name(description: str) -> str:
    stamp = datetime.now().strftime("%y%m%d-%H%M")
    run_hash = secrets.token_hex(2)
    return f"{stamp}-{run_hash}-{_sanitize_description(description) or 'run'}"


def _plan_run_layout(
    description: str,
    run_settings: MolGenRunConfig,
    job_section,
) -> RunLayout:
    dump_folder = getattr(job_section, "dump_folder", None) or "pretrain_runs"
    # Keep Qwen-specific subfolder to avoid collisions with other model families.
    logs_root = paths.get_pretrain_logs_path(Path(dump_folder) / "qwen3_06b")
    ckpts_root = paths.get_root_path("ckpts_root", Path(dump_folder) / "qwen3_06b")
    wandb_root = paths.get_wandb_path(Path(dump_folder) / "qwen3_06b")

    if run_settings.init_mode == "resume":
        ckpts_dir = _resolve_resume_ckpt_dir(run_settings)
        run_name = run_settings.run_name or ckpts_dir.name
        run_settings.run_name = run_name
        if not run_name:
            raise ValueError(
                "Unable to derive run_name while resuming; "
                "set molgen_run.run_name explicitly."
            )
        run_hash = _extract_run_hash_from_name(run_name) or "resume"
        logs_dir = logs_root / run_name
        wandb_dir = wandb_root / run_name
        return RunLayout(
            run_name=run_name,
            run_hash=run_hash,
            logs_dir=logs_dir,
            ckpts_dir=ckpts_dir,
            wandb_dir=wandb_dir,
            reuse_existing_dirs=True,
        )

    if run_settings.run_name:
        run_name = run_settings.run_name
    else:
        run_name = _generate_run_name(description)
        run_settings.run_name = run_name

    run_hash = _extract_run_hash_from_name(run_name) or "0000"
    logs_dir = logs_root / run_name
    ckpts_dir = ckpts_root / run_name
    wandb_dir = wandb_root / run_name
    return RunLayout(
        run_name=run_name,
        run_hash=run_hash,
        logs_dir=logs_dir,
        ckpts_dir=ckpts_dir,
        wandb_dir=wandb_dir,
        reuse_existing_dirs=False,
    )


def _load_tokenizer_details(tokenizer_path: Path) -> TokenizerDetails:
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
        fix_mistral_regex=True,
    )
    ensure_tokenizer_pad_token(tokenizer)
    vocab_size = len(tokenizer)
    base_vocab = QWEN3_BASE_VOCAB
    added_tokens = vocab_size - base_vocab
    if added_tokens < 0:
        raise ValueError(
            f"Tokenizer vocab smaller than Qwen3 base: {vocab_size} < {base_vocab}"
        )
    if vocab_size > QWEN3_PADDED_VOCAB:
        raise ValueError(
            f"Tokenizer vocab ({vocab_size}) exceeds Qwen3 padded vocab "
            f"({QWEN3_PADDED_VOCAB})."
        )
    padded_vocab = QWEN3_PADDED_VOCAB

    return TokenizerDetails(
        path=tokenizer_path,
        vocab_size=vocab_size,
        added_tokens=int(added_tokens),
        base_vocab_size=int(base_vocab),
        padded_vocab_size=int(padded_vocab),
    )


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _apply_run_environment(
    run_name: str,
    run_hash: str,
    wandb_dir: Path,
    cfg: QwenPretrainRunConfig,
) -> None:
    os.environ["RUN_NAME"] = run_name
    os.environ["WANDB_RUN_ID"] = run_hash
    os.environ["WANDB_RUN_NAME"] = run_name
    os.environ["WANDB_DIR"] = str(wandb_dir)
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.wandb_entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity
    if cfg.wandb_group:
        os.environ["WANDB_GROUP"] = cfg.wandb_group


def _configure_initial_load(job_config, run_settings: MolGenRunConfig) -> None:
    job_config.checkpoint.initial_load_path = None
    job_config.checkpoint.initial_load_in_hf = False
    job_config.checkpoint.initial_load_model_only = True

    if run_settings.init_mode == "scratch":
        _log_rank("MolGen pretrain init mode: scratch (random initialization).")
        return

    if run_settings.init_mode == "hf_pretrain":
        if not run_settings.base_model_tag:
            raise ValueError("base_model_tag must be set for hf_pretrain mode.")
        init_path = _resolve_tag_or_path(run_settings.base_model_tag)
        init_path = _ensure_hf_checkpoint_has_lm_head(init_path)
        job_config.checkpoint.initial_load_path = str(init_path)
        job_config.checkpoint.initial_load_in_hf = True
        job_config.checkpoint.initial_load_model_only = True
        _log_rank("MolGen pretrain init mode: hf_pretrain from %s", init_path)
        return

    if run_settings.init_mode == "resume":
        if not run_settings.resume_run_path_tag:
            raise ValueError("resume_run_path_tag must be set for resume mode.")
        resume_path = _resolve_tag_or_path(run_settings.resume_run_path_tag)
        job_config.checkpoint.initial_load_path = str(resume_path)
        job_config.checkpoint.initial_load_in_hf = False
        job_config.checkpoint.initial_load_model_only = False
        _log_rank("MolGen pretrain init mode: resume from %s", resume_path)
        return

    raise ValueError(f"Unknown init_mode '{run_settings.init_mode}'.")


def _normalize_job_config_paths(job_config) -> None:
    if job_config.model.hf_assets_path:
        job_config.model.hf_assets_path = str(
            _resolve_tag_or_path(job_config.model.hf_assets_path)
        )

    dataset_path = getattr(job_config.training, "dataset_path", None)
    if dataset_path:
        job_config.training.dataset_path = str(_resolve_tag_or_path(dataset_path))

    validation_path = getattr(job_config.validation, "dataset_path", None)
    if validation_path:
        job_config.validation.dataset_path = str(
            _resolve_tag_or_path(validation_path)
        )

    init_load = getattr(job_config.checkpoint, "initial_load_path", None)
    if init_load:
        job_config.checkpoint.initial_load_path = str(
            _resolve_tag_or_path(init_load)
        )


def _prepare_job_config(
    job_config,
    cfg: QwenPretrainRunConfig,
    run_settings: MolGenRunConfig,
    layout: RunLayout,
) -> None:
    logs_dir = layout.logs_dir
    ckpts_dir = layout.ckpts_dir
    wandb_dir = layout.wandb_dir

    _ensure_dirs(logs_dir, wandb_dir)

    if layout.reuse_existing_dirs:
        if not ckpts_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory {ckpts_dir} does not exist for resume."
            )
    else:
        if run_settings.init_mode == "hf_pretrain":
            ckpts_dir.parent.mkdir(parents=True, exist_ok=True)
        else:
            ckpts_dir.mkdir(parents=True, exist_ok=True)

    _enable_runtime_log(
        logs_dir,
        rotate_existing=layout.reuse_existing_dirs,
    )

    job_config.job.description = getattr(job_config.job, "description", None) or "run"
    job_config.job.dump_folder = str(logs_dir.parent)
    job_config.checkpoint.folder = str(ckpts_dir)
    if hasattr(job_config.checkpoint, "last_save_in_hf"):
        job_config.checkpoint.last_save_in_hf = False
    job_config.metrics.save_for_all_ranks = False
    if not job_config.metrics.save_tb_folder:
        job_config.metrics.save_tb_folder = "tb"

    tokenizer_path = _resolve_tag_or_path(run_settings.tokenizer_tag)
    tokenizer_info = _load_tokenizer_details(tokenizer_path)
    vocab_size = tokenizer_info.vocab_size

    data_cfg = getattr(job_config, "molgen_data", None)
    if data_cfg is not None:
        data_cfg.tokenizer_override = str(tokenizer_path)

    setattr(job_config.model, "tokenizer_override", str(tokenizer_path))
    setattr(job_config.model, "vocab_size_override", int(vocab_size))
    setattr(job_config.model, "tokenizer_added_tokens", tokenizer_info.added_tokens)
    setattr(
        job_config.model,
        "tokenizer_base_vocab_size",
        tokenizer_info.base_vocab_size,
    )
    setattr(
        job_config.model,
        "padded_vocab_size",
        tokenizer_info.padded_vocab_size,
    )

    tokenizer_cfg = getattr(job_config, "tokenizer", None)
    if tokenizer_cfg is None:

        class _JobTokenizerConfig:
            def __init__(self):
                self.base_vocab_size: int = 0
                self.added_tokens: int = 0
                self.padded_vocab_size: int = 0
                self.total_vocab_size: int = 0
                self.num_new_tokens: int = 0
                self.path: str = ""

        tokenizer_cfg = _JobTokenizerConfig()
        job_config.tokenizer = tokenizer_cfg

    tokenizer_cfg.base_vocab_size = QWEN3_BASE_VOCAB
    tokenizer_cfg.added_tokens = tokenizer_info.added_tokens
    tokenizer_cfg.padded_vocab_size = QWEN3_PADDED_VOCAB
    tokenizer_cfg.total_vocab_size = tokenizer_info.vocab_size
    tokenizer_cfg.num_new_tokens = tokenizer_info.added_tokens
    tokenizer_cfg.path = str(tokenizer_path)

    if tokenizer_cfg.total_vocab_size != QWEN3_BASE_VOCAB + tokenizer_cfg.added_tokens:
        raise ValueError(
            "Tokenizer total vocab inconsistent: "
            f"{tokenizer_cfg.total_vocab_size} != {QWEN3_BASE_VOCAB} + {tokenizer_cfg.added_tokens}"
        )

    _normalize_job_config_paths(job_config)
    _configure_initial_load(job_config, run_settings)

    _log_rank(
        "MolGen run '%s': model=%s flavor=%s seq_len=%d batch(local)=%d dtype=%s",
        layout.run_name,
        job_config.model.name,
        job_config.model.flavor,
        job_config.training.seq_len,
        job_config.training.local_batch_size,
        job_config.training.dtype,
    )
    _log_rank(
        "Directories: logs=%s ckpts=%s wandb=%s",
        logs_dir,
        ckpts_dir,
        wandb_dir,
    )
    _log_rank(
        "Tokenizer resolved: %s (base=%d, added=%d, total=%d tokens)",
        tokenizer_info.path,
        tokenizer_info.base_vocab_size,
        tokenizer_info.added_tokens,
        tokenizer_info.vocab_size,
    )
    train_path = getattr(job_config.training, "dataset_path", None)
    if train_path:
        _log_rank("Training dataset: %s", train_path)

    config_payload = job_config.to_dict()

    serialized_config = json.dumps(config_payload, indent=2, sort_keys=True)

    config_snapshot = logs_dir / "job_config.json"
    try:
        config_snapshot.write_text(serialized_config)
    except Exception as exc:  # pragma: no cover - diagnostic only
        _log_rank("Failed to write config snapshot: %s", exc)

    _schedule_checkpoint_config_write(ckpts_dir, serialized_config)

    return wandb_dir


def _enable_runtime_log(logs_dir: Path, rotate_existing: bool = False) -> None:
    global _TEE_ENABLED
    if _TEE_ENABLED or not _is_log_rank():
        return
    log_path = logs_dir / "runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if rotate_existing and log_path.exists():
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        backup_path = log_path.with_name(f"runtime.{timestamp}.log")
        log_path.rename(backup_path)
    log_file = open(log_path, "a", buffering=1)

    class _Tee:
        def __init__(self, stream, log_stream):
            self._stream = stream
            self._log = log_stream

        def write(self, data):
            self._stream.write(data)
            if data:
                self._log.write(data)

        def flush(self):
            self._stream.flush()
            self._log.flush()

        def isatty(self):
            return getattr(self._stream, "isatty", lambda: False)()

        def fileno(self):
            return self._stream.fileno()

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    _TEE_ENABLED = True


def _ensure_hf_checkpoint_has_lm_head(hf_dir: Path) -> Path:
    """
    TorchTitan expects `lm_head.weight` inside the HF checkpoint. Official
    Qwen3 releases omit it because the embedding matrix is tied. Rather than
    mutating the original download, clone it under `patched_for_titan/` once
    and inject the missing tensor so repeated launches can reuse the patched
    copy.
    """
    model_file = hf_dir / "model.safetensors"
    patched_dir = hf_dir / "patched_for_titan"
    patched_model = patched_dir / "model.safetensors"

    if not patched_model.exists():
        patched_dir.mkdir(parents=True, exist_ok=True)
        for item in hf_dir.iterdir():
            if item.is_file():
                shutil.copy(item, patched_dir / item.name)
            elif item.is_dir() and item.name != "patched_for_titan":
                target = patched_dir / item.name
                if not target.exists():
                    shutil.copytree(item, target)

    tensors = load_safetensor(patched_model)
    if "lm_head.weight" not in tensors and "model.embed_tokens.weight" in tensors:
        _log_rank(
            "Adding lm_head.weight to HF checkpoint at %s by copying model.embed_tokens.weight",
            patched_model,
        )
        embed = tensors["model.embed_tokens.weight"]
        tensors["lm_head.weight"] = embed.clone()
        save_safetensor(tensors, patched_model)
    return patched_dir


def _schedule_checkpoint_config_write(ckpt_dir: Path, payload: str) -> None:
    target = ckpt_dir / "config.json"
    if target.exists():
        return

    def _writer():
        while True:
            if target.exists():
                return
            if ckpt_dir.exists():
                try:
                    entries = [
                        p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step-")
                    ]
                except FileNotFoundError:
                    entries = []
                if entries:
                    try:
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        target.write_text(payload)
                    except Exception as exc:
                        _log_rank("Failed to write config to checkpoint dir: %s", exc)
                    return
            time.sleep(5)

    threading.Thread(target=_writer, daemon=True).start()


def launch_qwen3_pretrain(cfg: QwenPretrainRunConfig) -> None:
    toml_path = _resolve_config_path(cfg.train_toml)
    config_manager = ConfigManager()
    job_config = config_manager.parse_args(
        [f"--job.config_file={toml_path}"]
    )

    run_settings: MolGenRunConfig = getattr(
        job_config, "molgen_run", MolGenRunConfig()
    )
    description = getattr(job_config.job, "description", None) or "run"
    if cfg.run_desc:
        # Backward compat: allow CLI/env to override, but prefer job.description.
        description = cfg.run_desc
        job_config.job.description = description
    layout = _plan_run_layout(description, run_settings, job_config.job)
    _prepare_job_config(job_config, cfg, run_settings, layout)
    _apply_run_environment(
        layout.run_name,
        layout.run_hash,
        layout.wandb_dir,
        cfg,
    )

    set_active_job_config(job_config)
    init_logger()

    trainer: Optional[Trainer] = None
    try:
        trainer = Trainer(job_config)

        if job_config.checkpoint.create_seed_checkpoint:
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise RuntimeError(
                    "Seed checkpoints must be created with a single process."
                )
            if not job_config.checkpoint.enable:
                raise RuntimeError(
                    "Checkpointing must be enabled to create a seed checkpoint."
                )
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        if trainer:
            trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
