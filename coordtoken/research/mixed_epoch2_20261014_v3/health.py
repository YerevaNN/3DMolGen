"""Training-only safety checks. Never selects checkpoints using test results.

Call on globally reduced loss every 100 updates on every rank. Persist the
state alongside the optimizer so that resuming cannot reset deterioration.
This detects gross failure; passing it does not establish model quality.
"""
from dataclasses import asdict, dataclass, field
import math
import statistics


@dataclass
class LossGuard:
    window: int = 10
    min_step: int = 1000
    relative_limit: float = 10.0
    absolute_floor: float = 0.25
    patience: int = 3
    recent: list[float] = field(default_factory=list)
    best_median: float | None = None
    consecutive_bad: int = 0

    def observe(self, step: int, loss: float, gradient_norm: float) -> dict:
        if not math.isfinite(loss) or not math.isfinite(gradient_norm):
            raise FloatingPointError('Nonfinite training loss or gradient')
        if loss < 0 or gradient_norm < 0:
            raise ValueError('Invalid training metrics')
        if step < self.min_step:
            return {'healthy': True, 'reason': 'initialization period'}
        self.recent = (self.recent + [loss])[-self.window:]
        if len(self.recent) < self.window:
            return {'healthy': True, 'reason': 'collecting baseline'}
        median = statistics.median(self.recent)
        self.best_median = median if self.best_median is None else min(self.best_median, median)
        limit = max(self.absolute_floor, self.relative_limit * self.best_median)
        self.consecutive_bad = self.consecutive_bad + 1 if median > limit else 0
        return {'healthy': self.consecutive_bad < self.patience,
                'median_loss': median, 'best_median_loss': self.best_median,
                'limit': limit, 'consecutive_bad': self.consecutive_bad,
                'reason': 'sustained loss increase' if self.consecutive_bad >= self.patience else 'within guard limits'}

    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict):
        return cls(**state)
