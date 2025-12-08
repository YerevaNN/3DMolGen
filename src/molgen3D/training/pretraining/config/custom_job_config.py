from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from torchtitan.config.job_config import (
    JobConfig as TorchTitanJobConfig,
    Checkpoint as TorchTitanCheckpoint,
)

from molgen3D.config.paths import (
    get_data_path,
    get_tokenizer_path,
    resolve_tag,
)
from molgen3D.training.pretraining.helpers.wsds_scheduler import (
    WSDSSchedulerConfig,
)


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_via(
    value: Optional[str], resolver: Callable[[str], Optional[str] | str]
) -> Optional[str]:
    normalized = _strip_or_none(value)
    if not normalized:
        return normalized

    if ":" in normalized:
        return str(resolve_tag(normalized))

    resolved = resolver(normalized)
    return str(resolved) if resolved is not None else None


def _resolve_if_alias(
    value: Optional[str], resolver: Callable[[str], Optional[str] | str]
) -> Optional[str]:
    normalized = _strip_or_none(value)
    if not normalized:
        return normalized
    try:
        return _resolve_via(normalized, resolver)
    except KeyError:
        # Treat the value as a literal path if it is not a known alias.
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
    lookahead_limit: Optional[int] = None

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
        if self.lookahead_limit is None:
            self.lookahead_limit = 100


@dataclass
class MolGenRunConfig:
    """
    Encodes how a Qwen3 run should initialize (scratch, HF weights, resume) and
    which tokenizer or checkpoint tags to use. This section intentionally lives
    in the TOML so it serves as the single source of truth for launch scripts.
    """

    init_mode: Literal["scratch", "hf_pretrain", "resume"] = "scratch"
    tokenizer_tag: str = "tokenizers:qwen3_0.6b_custom"
    base_model_tag: Optional[str] = "base_paths:qwen3_0.6b_base_model"
    resume_run_path_tag: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class JobConfig(TorchTitanJobConfig):
    """
    Custom JobConfig that surfaces MolGen-specific dataloader settings and WSDS
    scheduler knobs.
    """

    molgen_data: MolGenDataConfig = field(default_factory=MolGenDataConfig)
    molgen_run: MolGenRunConfig = field(default_factory=MolGenRunConfig)
    wsds_scheduler: WSDSSchedulerConfig = field(default_factory=WSDSSchedulerConfig)

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        if self.wsds_scheduler.enable:
            lr = getattr(self.optimizer, "lr", None)
            if lr is None:
                raise ValueError("optimizer.lr must be set when WSDS scheduler is enabled.")
            if self.wsds_scheduler.base_lr is None:
                self.wsds_scheduler.base_lr = lr
            lr_max = getattr(self.wsds_scheduler, "lr_max", None)
            if lr_max is None or lr_max <= 0:
                self.wsds_scheduler.lr_max = lr
        
