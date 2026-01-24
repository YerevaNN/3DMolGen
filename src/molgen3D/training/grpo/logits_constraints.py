"""Generation-time constraints for conformer-only decoding."""

from __future__ import annotations

import types
from typing import Iterable, List, Optional, Sequence

import torch
from loguru import logger
from transformers import LogitsProcessor, StoppingCriteria


def _to_id_list(token_ids: Optional[Iterable[int]]) -> List[int]:
    if token_ids is None:
        return []
    return [int(tok) for tok in token_ids if tok is not None]


def _ends_with_sequence(seq: Sequence[int], suffix: Sequence[int]) -> bool:
    if not suffix or len(seq) < len(suffix):
        return False
    return list(seq[-len(suffix):]) == list(suffix)


def _suffix_prefix_overlap(seq: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern:
        return 0
    max_len = min(len(seq), len(pattern) - 1)
    for k in range(max_len, 0, -1):
        if list(seq[-k:]) == list(pattern[:k]):
            return k
    return 0


class ConformerControlLogitsProcessor(LogitsProcessor):
    """Force [CONFORMER] tags after end tags and ban disallowed tokens."""

    def __init__(
        self,
        conformer_start_ids: Sequence[int],
        conformer_end_ids: Sequence[int],
        banned_token_ids: Optional[Iterable[int]] = None,
        target_k: int = 8,
        force_hard: bool = True,
    ) -> None:
        self.conformer_start_ids = list(conformer_start_ids)
        self.conformer_end_ids = list(conformer_end_ids)
        self.banned_token_ids = _to_id_list(banned_token_ids)
        self.target_k = max(int(target_k), 1)
        self.force_hard = bool(force_hard)
        self._end_counts: Optional[List[int]] = None
        self._start_counts: Optional[List[int]] = None
        self._cached_lens: Optional[List[int]] = None

    def _init_state(self, input_ids: torch.Tensor) -> None:
        batch_size = int(input_ids.shape[0])
        self._end_counts = [0] * batch_size
        self._start_counts = [0] * batch_size
        self._cached_lens = [0] * batch_size
        for row in range(batch_size):
            seq = input_ids[row].tolist()
            self._end_counts[row] = self._count_occurrences(seq, self.conformer_end_ids)
            self._start_counts[row] = self._count_occurrences(seq, self.conformer_start_ids)
            self._cached_lens[row] = len(seq)

    @staticmethod
    def _count_occurrences(seq: Sequence[int], pattern: Sequence[int]) -> int:
        if not pattern or len(seq) < len(pattern):
            return 0
        count = 0
        pat_len = len(pattern)
        for idx in range(len(seq) - pat_len + 1):
            if list(seq[idx:idx + pat_len]) == list(pattern):
                count += 1
        return count

    def _update_counts(self, input_ids: torch.Tensor) -> None:
        if self._end_counts is None or self._start_counts is None or self._cached_lens is None:
            self._init_state(input_ids)
            return
        for row in range(int(input_ids.shape[0])):
            seq = input_ids[row].tolist()
            old_len = self._cached_lens[row]
            new_len = len(seq)
            if new_len <= old_len:
                continue
            for end_pos in range(old_len, new_len):
                if (
                    end_pos + 1 >= len(self.conformer_end_ids)
                    and list(seq[end_pos + 1 - len(self.conformer_end_ids):end_pos + 1]) == self.conformer_end_ids
                ):
                    self._end_counts[row] += 1
                if (
                    end_pos + 1 >= len(self.conformer_start_ids)
                    and list(seq[end_pos + 1 - len(self.conformer_start_ids):end_pos + 1]) == self.conformer_start_ids
                ):
                    self._start_counts[row] += 1
            self._cached_lens[row] = new_len

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self._update_counts(input_ids)
        if self._end_counts is None or self._start_counts is None:
            return scores

        for row in range(int(input_ids.shape[0])):
            seq = input_ids[row].tolist()
            end_count = self._end_counts[row]
            start_count = self._start_counts[row]
            at_start = start_count == 0 and end_count == 0
            ended = _ends_with_sequence(seq, self.conformer_end_ids)
            should_force = (end_count < self.target_k) and (at_start or ended)

            if should_force and self.conformer_start_ids:
                if _ends_with_sequence(seq, self.conformer_start_ids):
                    continue
                prefix_len = _suffix_prefix_overlap(seq, self.conformer_start_ids)
                next_id = self.conformer_start_ids[prefix_len]
                if self.force_hard:
                    scores[row].fill_(float("-inf"))
                    scores[row, next_id] = 0.0
                else:
                    scores[row, next_id] = scores[row, next_id] + 1e4

        for token_id in self.banned_token_ids:
            if 0 <= token_id < scores.shape[-1]:
                scores[:, token_id] = float("-inf")
        return scores


class ConformerCountStoppingCriteria(StoppingCriteria):
    """Stop generation once all sequences reach the target number of [/CONFORMER] tags."""

    def __init__(self, conformer_end_ids: Sequence[int], target_k: int) -> None:
        self.conformer_end_ids = list(conformer_end_ids)
        self.target_k = max(int(target_k), 1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        if not self.conformer_end_ids:
            return False
        for row in range(int(input_ids.shape[0])):
            seq = input_ids[row].tolist()
            count = 0
            pat_len = len(self.conformer_end_ids)
            for idx in range(len(seq) - pat_len + 1):
                if list(seq[idx:idx + pat_len]) == self.conformer_end_ids:
                    count += 1
            if count < self.target_k:
                return False
        return True


def attach_conformer_controls(model, tokenizer, config) -> None:
    target_k = int(getattr(config.grpo, "target_conformers", 8))
    conf_tags = getattr(config.model, "conf_tags", ["[CONFORMER]", "[/CONFORMER]"])
    mol_tags = getattr(config.model, "mol_tags", ["[SMILES]", "[/SMILES]"])

    conformer_start_ids = tokenizer.encode(conf_tags[0], add_special_tokens=False)
    conformer_end_ids = tokenizer.encode(conf_tags[1], add_special_tokens=False)
    smiles_start_ids = tokenizer.encode(mol_tags[0], add_special_tokens=False)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids(getattr(config.model, "pad_token", ""))

    banned_ids = set(smiles_start_ids)
    if pad_id is not None:
        banned_ids.add(int(pad_id))

    if not conformer_start_ids or not conformer_end_ids:
        logger.warning("Conformer tags missing; skipping conformer logits processor.")
        return

    processor_factory = lambda: ConformerControlLogitsProcessor(
        conformer_start_ids=conformer_start_ids,
        conformer_end_ids=conformer_end_ids,
        banned_token_ids=banned_ids,
        target_k=target_k,
        force_hard=True,
    )
    stopping_factory = lambda: ConformerCountStoppingCriteria(conformer_end_ids, target_k)

    if hasattr(model, "_get_logits_processor"):
        original = model._get_logits_processor

        def _get_logits_processor(self, *args, **kwargs):
            processors = original(*args, **kwargs)
            processors.append(processor_factory())
            return processors

        model._get_logits_processor = types.MethodType(_get_logits_processor, model)
    else:
        logger.warning("Model does not expose _get_logits_processor; cannot attach conformer constraints.")

    if hasattr(model, "_get_stopping_criteria"):
        original_stop = model._get_stopping_criteria

        def _get_stopping_criteria(self, *args, **kwargs):
            criteria = original_stop(*args, **kwargs)
            criteria.append(stopping_factory())
            return criteria

        model._get_stopping_criteria = types.MethodType(_get_stopping_criteria, model)
    else:
        logger.warning("Model does not expose _get_stopping_criteria; skipping conformer stopping criterion.")

