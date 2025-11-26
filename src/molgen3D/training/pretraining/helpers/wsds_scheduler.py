from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from torch.optim import Optimizer
from torchtitan.components import lr_scheduler as tt_lr_scheduler


@dataclass
class WSDSSchedulerConfig:
    enable: bool = False
    warmup_steps: int = 0
    checkpoints: List[int] = field(default_factory=list)
    lr_max: float = 0.0
    lr_min: float = 0.0
    decay_frac: Optional[float] = 0.1
    decay_steps: Optional[int] = None
    base_lr: Optional[float] = None

    def validate(self) -> List[int]:
        ckpts = [int(x) for x in self.checkpoints]
        if ckpts != sorted(ckpts):
            raise ValueError("Checkpoints must be strictly increasing")
        if any(c <= 0 for c in ckpts):
            raise ValueError("Checkpoints must be positive")
        if self.decay_frac is not None and not (0 <= self.decay_frac <= 1):
            raise ValueError("decay_frac must be within [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.lr_max <= 0:
            raise ValueError("lr_max must be positive")
        if self.lr_min < 0 or self.lr_min > self.lr_max:
            raise ValueError("lr_min must be between 0 and lr_max")
        if self.decay_steps is not None and self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        if self.decay_steps is None and self.decay_frac is None:
            raise ValueError("Set decay_steps or decay_frac so decay length is defined")
        if not ckpts:
            raise ValueError("At least one checkpoint is required for WSD-S")
        return ckpts


class WSDSSchedule:

    def __init__(
        self,
        optimizer: Optimizer,
        checkpoints: List[int],
        warmup_steps: int,
        lr_max: float,
        lr_min: float = 0.0,
        decay_frac: Optional[float] = 0.1,
        decay_steps: Optional[int] = None,
        base_lr: Optional[float] = None,
    ) -> None:
        self.optimizer = optimizer
        self.checkpoints = checkpoints
        self.warmup_steps = max(0, warmup_steps)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.decay_frac = decay_frac
        self.decay_steps = decay_steps
        self.base_lr = (
            float(base_lr)
            if base_lr is not None
            else float(optimizer.param_groups[0]["lr"])
        )
        if self.base_lr <= 0:
            raise ValueError("Optimizer base lr must be positive")
        self.lr_lambda = self.__call__

    def _find_target(self, step: int) -> Optional[int]:
        for target in self.checkpoints:
            if step <= target:
                return target
        return None

    def _decay_length(self, target: int) -> int:
        if self.decay_steps is not None:
            return max(1, min(target, self.decay_steps))
        if self.decay_frac is None:
            raise ValueError("decay_frac must be set when decay_steps is None")
        return max(1, int(target * self.decay_frac))

    def _decay_start(self, target: int) -> int:
        return max(self.warmup_steps, target - self._decay_length(target))

    def get_lr(self, step: int) -> float:
        if step < 0:
            return 0.0
        if step < self.warmup_steps:
            warmup_progress = step / max(1, self.warmup_steps)
            return self.lr_max * warmup_progress

        target = self._find_target(step)
        if target is None:
            return self.lr_min

        decay_start = self._decay_start(target)
        if step < decay_start:
            return self.lr_max
        if step > target:
            return self.lr_min

        progress = (step - decay_start) / max(1, target - decay_start)
        progress = min(1.0, max(0.0, progress))
        return self.lr_max - (self.lr_max - self.lr_min) * progress

    def __call__(self, step: int) -> float:
        lr = self.get_lr(step)
        return lr / self.base_lr


_active_job_config = None


def set_active_job_config(job_config):
    global _active_job_config
    _active_job_config = job_config


def _get_wsds_config():
    if _active_job_config is None:
        return None
    cfg = getattr(_active_job_config, "wsds_scheduler", None)
    return cfg if cfg and cfg.enable else None


def build_wsds_lr_schedulers(optimizers, lr_scheduler_config, training_steps):
    cfg = _get_wsds_config()
    if not cfg:
        return tt_lr_scheduler.build_lr_schedulers(
            optimizers, lr_scheduler_config, training_steps
        )
    checkpoints = cfg.validate()
    scheduler = WSDSSchedule(
        optimizers.optimizers[0],
        checkpoints=checkpoints,
        warmup_steps=cfg.warmup_steps,
        lr_max=cfg.lr_max,
        lr_min=cfg.lr_min,
        decay_frac=cfg.decay_frac,
        decay_steps=cfg.decay_steps,
        base_lr=cfg.base_lr,
    )
    return tt_lr_scheduler.LRSchedulersContainer(
        optimizers, scheduler.lr_lambda
    )


