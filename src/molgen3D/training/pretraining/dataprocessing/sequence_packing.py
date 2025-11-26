from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass
class PendingUnit:
    tokens: List[int]
    total_len: int


class SequenceState:
    def __init__(
        self,
        seq_len: int,
        separator_id: int,
        pad_id: int,
        ignore_index: int,
    ):
        self.seq_len = int(seq_len)
        self.separator_id = int(separator_id)
        self.pad_id = int(pad_id)
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self) -> None:
        self.tokens: List[int] = []
        self.used_len = 0

    def can_fit(self, unit_len: int) -> bool:
        return self.used_len + unit_len <= self.seq_len

    def append_unit(self, unit: PendingUnit) -> None:
        self.tokens.extend(unit.tokens)
        self.tokens.append(self.separator_id)
        self.used_len += unit.total_len

    def has_content(self) -> bool:
        return self.used_len > 0

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.tokens:
            raise ValueError("Cannot finalize an empty sequence.")
        seq = list(self.tokens)
        pad_needed = self.seq_len - len(seq)
        if pad_needed < 0:
            raise ValueError("Sequence longer than configured seq_len.")
        if pad_needed:
            seq.extend([self.pad_id] * pad_needed)
        inp = torch.tensor(seq, dtype=torch.long)
        lab = inp.clone()
        if pad_needed:
            lab[-pad_needed:] = self.ignore_index
        self.reset()
        return inp, lab

    def export_tokens(self) -> List[int]:
        return list(self.tokens)

    def load_tokens(self, tokens: Sequence[int]) -> None:
        tokens = list(tokens)
        if tokens:
            self.tokens = tokens
            self.used_len = len(tokens)
        else:
            self.reset()
