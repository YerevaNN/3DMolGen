import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from secrets import token_hex
from typing import Callable, Optional

from torchtitan.config.job_config import JobConfig as TorchTitanJobConfig

from molgen3D.config.paths import (
    get_data_path,
    get_pretrain_logs_path,
    get_root_path,
    get_tokenizer_path,
    get_wandb_path,
)


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_via(value: Optional[str], resolver: Callable[[str], str]) -> Optional[str]:
    normalized = _strip_or_none(value)
    if not normalized:
        return normalized
    return str(resolver(normalized))


def _resolve_if_alias(value: Optional[str], resolver: Callable[[str], str]) -> Optional[str]:
    normalized = _strip_or_none(value)
    if not normalized:
        return normalized
    try:
        return str(resolver(normalized))
    except KeyError:
        # Treat value as literal path; downstream components will surface errors if it is invalid.
        return normalized


@dataclass
class MolGenDataConfig:
    """
    TorchTitan extension for configuring the custom MolGen JSONL dataloader.

    These options map directly to the arguments exposed by
    `molgen3D.training.pretraining.dataprocessing.dataloader.build_dataloader`.
    Either provide concrete paths (`train_path`, `tokenizer_override`) or refer to
    entries in `paths.yaml` via `train_path_key` / `tokenizer_key`.
    """

    train_path: str = ""
    tokenizer_override: Optional[str] = None
    train_path_key: Optional[str] = None
    tokenizer_key: Optional[str] = None
    min_emb_len: int = 16
    shuffle_lines: bool = True
    infinite: bool = True
    seed: Optional[int] = None
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Resolve any aliases declared in paths.yaml so the dataloader only
        ever receives absolute paths.
        """

        train_path = (
            _resolve_via(self.train_path_key, get_data_path)
            if self.train_path_key
            else _resolve_if_alias(self.train_path, get_data_path)
        )
        self.train_path = train_path or ""

        tokenizer_override = (
            _resolve_via(self.tokenizer_key, get_tokenizer_path)
            if self.tokenizer_key
            else _resolve_if_alias(
                _strip_or_none(self.tokenizer_override), get_tokenizer_path
            )
        )
        self.tokenizer_override = tokenizer_override

@dataclass
class JobConfig(TorchTitanJobConfig):
    """
    Custom JobConfig that surfaces MolGen-specific dataloader settings.

    Register this dataclass via `job.custom_config_module` so TorchTitan parses
    the extra TOML section automatically.
    """

    molgen_data: MolGenDataConfig = field(default_factory=MolGenDataConfig)


def _resolve_job_paths(job_config: TorchTitanJobConfig) -> None:
    resolved_dump = _resolve_if_alias(
        job_config.job.dump_folder,
        lambda value: str(get_root_path("ckpts_root", value)),
    )
    if resolved_dump:
        job_config.job.dump_folder = resolved_dump


def _resolve_training_paths(job_config: TorchTitanJobConfig) -> None:
    dataset_path = getattr(job_config.training, "dataset_path", None)
    if dataset_path:
        resolved = _resolve_if_alias(dataset_path, get_data_path)
        if resolved:
            job_config.training.dataset_path = resolved


def _resolve_validation_paths(job_config: TorchTitanJobConfig) -> None:
    dataset_path = getattr(job_config.validation, "dataset_path", None)
    if dataset_path:
        resolved = _resolve_if_alias(dataset_path, get_data_path)
        if resolved:
            job_config.validation.dataset_path = resolved


def _resolve_model_paths(job_config: TorchTitanJobConfig) -> None:
    if getattr(job_config.model, "hf_assets_path", None):
        resolved = _resolve_if_alias(job_config.model.hf_assets_path, get_tokenizer_path)
        if resolved:
            job_config.model.hf_assets_path = resolved
    else:
        data_cfg = getattr(job_config, "molgen_data", None)
        if data_cfg and data_cfg.tokenizer_override:
            job_config.model.hf_assets_path = data_cfg.tokenizer_override

    if getattr(job_config.model, "tokenizer_path", None):
        resolved = _resolve_if_alias(job_config.model.tokenizer_path, get_tokenizer_path)
        if resolved:
            job_config.model.tokenizer_path = resolved


def _sanitize_description(description: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", description)
    safe = safe.strip("-_")
    return safe or "run"


def _ensure_wandb_run_name(job_config: TorchTitanJobConfig) -> str:
    existing = os.getenv("WANDB_RUN_NAME")
    if existing:
        return existing

    description = job_config.job.description or ""
    safe_description = _sanitize_description(description)

    timestamp = datetime.utcnow().strftime("%y%m%d_%H%M")
    base_name = f"{timestamp}-{safe_description}"

    user_name = os.getenv("WANDB_NAME")
    if user_name:
        base_name = f"{base_name}-{user_name}"

    os.environ["WANDB_RUN_NAME"] = base_name
    return base_name


def _dump_args_snapshot(logs_path: Path, job_config: TorchTitanJobConfig) -> None:
    args_path = logs_path / "torchtitan_args.json"
    try:
        args_dict = job_config.to_dict()
    except AttributeError:
        args_dict = asdict(job_config)
    args_path.write_text(json.dumps(args_dict, default=str, indent=2))


def _configure_log_env(job_config: TorchTitanJobConfig) -> None:
    run_name = _ensure_wandb_run_name(job_config)

    base_dump = Path(job_config.job.dump_folder)
    base_dump.mkdir(parents=True, exist_ok=True)
    if base_dump.name == run_name:
        run_dump = base_dump
    else:
        run_dump = base_dump / run_name
    run_dump.mkdir(parents=True, exist_ok=True)
    job_config.job.dump_folder = str(run_dump)

    logs_path = get_pretrain_logs_path(run_name)
    wandb_path = get_wandb_path(run_name)

    logs_path.mkdir(parents=True, exist_ok=True)
    wandb_path.mkdir(parents=True, exist_ok=True)

    log_file = logs_path / "runtime.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_stream = open(log_file, "a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
            self._last_logged: tuple[str, float] = ("", 0.0)

        def write(self, data):
            for stream in self._streams:
                stream.write(data)
            now = time.monotonic()
            if log_stream and data and data == self._last_logged[0] and (now - self._last_logged[1]) < 0.05:
                return
            if log_stream:
                log_stream.write(data)
            self._last_logged = (data, now)

        def flush(self):
            for stream in self._streams:
                stream.flush()
            if log_stream:
                log_stream.flush()

        def fileno(self):
            for stream in self._streams:
                if hasattr(stream, "fileno"):
                    return stream.fileno()
            raise AttributeError("Underlying streams have no fileno()")

        def isatty(self):
            for stream in self._streams:
                if hasattr(stream, "isatty") and stream.isatty():
                    return True
            return False

    sys.stdout = _Tee(sys.stdout, log_stream)
    sys.stderr = _Tee(sys.stderr, log_stream)

    os.environ["TORCHTITAN_LOG_DIR"] = str(logs_path)
    os.environ["WANDB_DIR"] = str(wandb_path)

    _dump_args_snapshot(logs_path, job_config)


_ORIGINAL_POST_INIT = getattr(TorchTitanJobConfig, "__post_init__", None)
_PATCHED = False


def _apply_post_init_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    def _patched_post_init(self, *args, **kwargs):
        if _ORIGINAL_POST_INIT:
            _ORIGINAL_POST_INIT(self, *args, **kwargs)
        _resolve_job_paths(self)
        _resolve_training_paths(self)
        _resolve_validation_paths(self)
        _resolve_model_paths(self)
        _configure_log_env(self)

    TorchTitanJobConfig.__post_init__ = _patched_post_init
    _PATCHED = True


_apply_post_init_patch()


