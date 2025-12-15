"""Multi-component reward shaping for GRPO conformer generation.

This module implements the reward design described in the user specification.
For each prompt we score K generated conformers against up to M reference
conformers, balancing precision, coverage, identity, validity and (optionally)
diversity.  The implementation keeps the input/output semantics of the legacy
reward so it can drop-in replace the existing hook in the GRPO trainer.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger
from rdkit import Chem

import wandb

from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.evaluation.utils import extract_between
from molgen3D.training.grpo.config import AdvancedRewardConfig, Config
from molgen3D.training.grpo.stats import RunStatistics
from molgen3D.training.grpo.utils import load_ground_truths
from molgen3D.utils.utils import get_best_rmsd


PoseBustersChecker = Callable[[Chem.Mol], bool]
MatchScorer = Callable[[Optional[Chem.Mol], str, bool], float]
RMSDFunc = Callable[[Chem.Mol, Chem.Mol], float]

_POSEBUSTERS_INSTANCE = None
_POSEBUSTERS_AVAILABLE = None


def _default_posebusters_checker(config_name: str) -> PoseBustersChecker:
    """Return a callable that runs PoseBusters if available.

    We cache the PoseBusters instance per process to avoid repeated expensive
    initialisation.  When PoseBusters cannot be imported we degrade gracefully
    and mark all checks as failed (reward=0) so the agent learns from the
    penalty signal without crashing the loop.
    """

    def _load_posebusters():
        global _POSEBUSTERS_INSTANCE, _POSEBUSTERS_AVAILABLE
        if _POSEBUSTERS_AVAILABLE is False:
            return None
        if _POSEBUSTERS_INSTANCE is None:
            try:
                from posebusters import PoseBusters  # type: ignore

                _POSEBUSTERS_INSTANCE = PoseBusters(config=config_name)
                _POSEBUSTERS_AVAILABLE = True
            except Exception as exc:  # pragma: no cover - depends on optional dep
                logger.warning(f"PoseBusters unavailable; validity reward disabled: {exc}")
                _POSEBUSTERS_AVAILABLE = False
                return None
        return _POSEBUSTERS_INSTANCE

    def _checker(mol: Chem.Mol) -> bool:
        posebusters = _load_posebusters()
        if posebusters is None:
            return False
        try:
            df = posebusters.bust([mol], None, None, full_report=False)
            # PoseBusters returns a DataFrame with boolean flags per check.
            return bool(df.all(axis=1).iloc[0])
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(f"PoseBusters check failed: {exc}")
            return False

    return _checker


def _exp_decay(distance: float, scale: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    scale = max(scale, 1e-8)
    return float(np.exp(-distance / scale))


def _one_minus_decay(distance: float, scale: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    scale = max(scale, 1e-8)
    return float(1.0 - np.exp(-distance / scale))


def _normalize(values: Iterable[float], eps: float) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    mean = float(arr[finite_mask].mean())
    std = float(arr[finite_mask].std())
    if std < eps:
        return np.zeros_like(arr)
    return (arr - mean) / (std + eps)


def _default_rmsd_fn(probe: Chem.Mol, ref: Chem.Mol) -> float:
    try:
        rmsd = get_best_rmsd(probe, ref, use_alignmol=False)
        if rmsd is None or np.isnan(rmsd):
            return float("inf")
        return float(rmsd)
    except Exception as exc:
        logger.debug(f"RMSD computation failed: {exc}")
        return float("inf")


def _default_match_score(mol: Optional[Chem.Mol], target_smiles: str, allow_partial: bool) -> float:
    if mol is None:
        return 0.0
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    except Exception:
        return 0.0
    if smiles == target_smiles:
        return 1.0
    if not allow_partial:
        return 0.0
    matcher = SequenceMatcher(None, target_smiles, smiles)
    return float(matcher.ratio())


def _decode_generated_conformer(raw_text: str) -> Optional[Chem.Mol]:
    if not raw_text:
        return None
    try:
        return decode_cartesian_raw(raw_text)
    except Exception as exc:
        logger.debug(f"Failed to decode generated conformer: {exc}")
        return None


@dataclass
class _GroupRewards:
    precision: List[float]
    coverage: List[float]
    match: List[float]
    validity: List[float]
    diversity: List[float]
    final: List[float]
    rmsd_values: List[float]
    coverage_claims: int
    reference_count: int


class MultiComponentRewardCalculator:
    """Stateful reward calculator keeping shared utilities and caches."""

    def __init__(
        self,
        config: Config,
        stats: RunStatistics,
        tokenizer,
        posebusters_checker: Optional[PoseBustersChecker] = None,
        match_scorer: Optional[MatchScorer] = None,
        rmsd_fn: Optional[RMSDFunc] = None,
    ) -> None:
        self.config = config
        self.stats = stats
        self.tokenizer = tokenizer
        self.settings: AdvancedRewardConfig = config.grpo.advanced_reward
        self.posebusters_checker = posebusters_checker
        if self.posebusters_checker is None and self.settings.enable_posebusters:
            self.posebusters_checker = _default_posebusters_checker(self.settings.posebusters_config)
        self.match_scorer = match_scorer or _default_match_score
        self.rmsd_fn = rmsd_fn or _default_rmsd_fn
        self._seen_prompts: set[str] = set()

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        grouped = self._group_by_prompt(prompts, completions)
        final_rewards = [0.0] * len(completions)

        all_precision: List[float] = []
        all_coverage: List[float] = []
        all_match: List[float] = []
        all_validity: List[float] = []
        all_diversity: List[float] = []
        all_final: List[float] = []
        all_rmsd: List[float] = []

        total_coverage_claims = 0
        total_coverage_possible = 0
        posebusters_passes = 0
        posebusters_checks = 0

        for group in grouped.values():
            rewards = self._compute_group_rewards(**group)
            indices = group["indices"]
            for local_idx, global_idx in enumerate(indices):
                final_rewards[global_idx] = rewards.final[local_idx]

            all_precision.extend(rewards.precision)
            all_coverage.extend(rewards.coverage)
            all_match.extend(rewards.match)
            all_validity.extend(rewards.validity)
            all_diversity.extend(rewards.diversity)
            all_final.extend(rewards.final)
            all_rmsd.extend(rewards.rmsd_values)
            total_coverage_claims += rewards.coverage_claims
            total_coverage_possible += rewards.reference_count

            posebusters_passes += sum(1 for val in rewards.validity if val > 0.5)
            posebusters_checks += len(rewards.validity)

        self._log_metrics(
            all_precision,
            all_coverage,
            all_match,
            all_validity,
            all_diversity,
            all_final,
            all_rmsd,
            total_coverage_claims,
            total_coverage_possible,
            posebusters_passes,
            posebusters_checks,
        )

        return final_rewards

    def _group_by_prompt(self, prompts: List[str], completions: List[str]):
        groups: "OrderedDict[int, dict]" = OrderedDict()
        active_groups: dict[Tuple[str, str], dict] = {}
        expected_k = max(1, self.config.grpo.num_generations)
        group_counter = 0

        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            canonical_smiles = extract_between(prompt, "[SMILES]", "[/SMILES]")
            if canonical_smiles is None:
                logger.warning(f"Prompt missing SMILES tag; treating as invalid.\nPrompt: {prompt}")
                canonical_smiles = ""
            key = (prompt, canonical_smiles)
            group = active_groups.get(key)
            if group is None or len(group["completions"]) >= expected_k:
                group = {
                    "prompt": prompt,
                    "canonical_smiles": canonical_smiles,
                    "completions": [],
                    "indices": [],
                }
                groups[group_counter] = group
                active_groups[key] = group
                group_counter += 1
            group["completions"].append(completion)
            group["indices"].append(idx)
            if len(group["completions"]) >= expected_k:
                active_groups.pop(key, None)
        return groups

    def _compute_group_rewards(
        self,
        prompt: str,
        canonical_smiles: str,
        completions: List[str],
        indices: List[int],
    ) -> _GroupRewards:
        delta = self.settings.delta
        self.stats.processed_prompts += len(completions)

        ground_truths = load_ground_truths(
            canonical_smiles,
            num_gt=self.settings.max_reference_conformers or self.config.grpo.max_ground_truths,
        )
        if not ground_truths:
            self.stats.failed_ground_truth += len(completions)
            logger.info(f"No ground truth conformers for prompt: {canonical_smiles}")
            return _GroupRewards(
                precision=[0.0] * len(completions),
                coverage=[0.0] * len(completions),
                match=[0.0] * len(completions),
                validity=[0.0] * len(completions),
                diversity=[0.0] * len(completions),
                final=[0.0] * len(completions),
                rmsd_values=[float("nan")] * len(completions),
                coverage_claims=0,
                reference_count=0,
            )

        if canonical_smiles and canonical_smiles not in self._seen_prompts:
            self._seen_prompts.add(canonical_smiles)
            self.stats.distinct_prompts += 1

        generated_mols: List[Optional[Chem.Mol]] = []

        for completion in completions:
            generated_conformer = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
            if generated_conformer is None:
                self.stats.failed_conformer_generation += 1
                generated_mols.append(None)
                continue
            mol = _decode_generated_conformer(generated_conformer)
            generated_mols.append(mol)
            if mol is None:
                self.stats.failed_conformer_generation += 1

        distance_matrix = self._pairwise_rmsd(generated_mols, ground_truths)

        precision_scores, min_distances = self._precision_scores(distance_matrix)
        coverage_scores, coverage_claims = self._coverage_scores(distance_matrix, delta)
        match_scores = self._match_scores(generated_mols, canonical_smiles)
        validity_scores = self._validity_scores(generated_mols)
        diversity_scores = self._diversity_scores(generated_mols)

        final_scores = self._combine_components(
            precision_scores,
            coverage_scores,
            match_scores,
            validity_scores,
            diversity_scores,
        )

        for rmsd in min_distances:
            if np.isfinite(rmsd):
                self.stats.add_rmsd(rmsd)
            else:
                self.stats.failed_rmsd += 1

        for score in final_scores:
            if score > 0:
                self.stats.successful_generations += 1

        return _GroupRewards(
            precision=precision_scores,
            coverage=coverage_scores,
            match=match_scores,
            validity=validity_scores,
            diversity=diversity_scores,
            final=final_scores,
            rmsd_values=min_distances,
            coverage_claims=coverage_claims,
            reference_count=len(ground_truths),
        )

    def _pairwise_rmsd(
        self,
        generated: List[Optional[Chem.Mol]],
        references: List[Chem.Mol],
    ) -> np.ndarray:
        if not generated or not references:
            return np.full((len(generated), len(references)), float("inf"), dtype=np.float32)

        matrix = np.full((len(generated), len(references)), float("inf"), dtype=np.float32)
        for i, candidate in enumerate(generated):
            if candidate is None:
                continue
            for j, reference in enumerate(references):
                matrix[i, j] = self.rmsd_fn(candidate, reference)
        return matrix

    def _precision_scores(self, distances: np.ndarray) -> Tuple[List[float], List[float]]:
        if distances.size == 0:
            return [0.0] * len(distances), []
        min_dists = np.min(distances, axis=1) if distances.shape[1] > 0 else np.full(distances.shape[0], float("inf"))
        precision_scores = [
            _exp_decay(dist, self.settings.precision_scale) if np.isfinite(dist) else 0.0 for dist in min_dists
        ]
        return precision_scores, min_dists.tolist()

    def _coverage_scores(self, distances: np.ndarray, delta: float) -> Tuple[List[float], int]:
        if distances.size == 0 or distances.shape[1] == 0:
            return [0.0] * len(distances), 0
        uncovered = set(range(distances.shape[1]))
        scores = [0.0] * distances.shape[0]
        claims = 0

        for idx in range(distances.shape[0]):
            if not uncovered:
                break
            remaining = {ref_idx: distances[idx, ref_idx] for ref_idx in uncovered}
            if not remaining:
                continue
            best_ref = min(remaining, key=remaining.get)
            best_dist = remaining[best_ref]
            if np.isfinite(best_dist) and best_dist < delta:
                scores[idx] = _exp_decay(best_dist, self.settings.coverage_scale)
                uncovered.remove(best_ref)
                claims += 1

        return scores, claims

    def _match_scores(self, generated: List[Optional[Chem.Mol]], canonical_smiles: str) -> List[float]:
        scores: List[float] = []
        for mol in generated:
            score = self.match_scorer(mol, canonical_smiles, self.settings.match_partial_credit)
            if score == 0.0 and canonical_smiles:
                self.stats.failed_matching_smiles += 1
            scores.append(score)
        return scores

    def _validity_scores(self, generated: List[Optional[Chem.Mol]]) -> List[float]:
        if not self.settings.enable_posebusters or self.posebusters_checker is None:
            return [0.0] * len(generated)

        scores: List[float] = []
        for mol in generated:
            if mol is None:
                scores.append(0.0)
                continue
            passed = self.posebusters_checker(mol)
            scores.append(1.0 if passed else 0.0)
            if passed:
                self.stats.posebusters_successes += 1
            else:
                self.stats.posebusters_failures += 1
        return scores

    def _diversity_scores(self, generated: List[Optional[Chem.Mol]]) -> List[float]:
        if not self.settings.enable_diversity or len(generated) == 0:
            return [0.0] * len(generated)
        scores = []
        previous: List[Chem.Mol] = []
        for mol in generated:
            if mol is None or not previous:
                scores.append(0.0)
            else:
                min_dist = min(self.rmsd_fn(mol, prev) for prev in previous)
                scores.append(_one_minus_decay(min_dist, self.settings.diversity_scale))
            if mol is not None:
                previous.append(mol)
        return scores

    def _combine_components(
        self,
        precision: List[float],
        coverage: List[float],
        match: List[float],
        validity: List[float],
        diversity: List[float],
    ) -> List[float]:
        eps = self.settings.normalization_epsilon

        prec_norm = _normalize(precision, eps) if precision else np.zeros(0, dtype=np.float32)
        cov_norm = _normalize(coverage, eps) if coverage else np.zeros(0, dtype=np.float32)
        match_norm = _normalize(match, eps) if match else np.zeros(0, dtype=np.float32)
        div_norm = _normalize(diversity, eps) if diversity else np.zeros(0, dtype=np.float32)

        weights = self.settings.weights
        final_scores = []
        for idx in range(len(precision)):
            score = (
                weights.precision * float(prec_norm[idx])
                + weights.coverage * float(cov_norm[idx])
                + weights.match * float(match_norm[idx])
                + weights.validity * float(validity[idx])
            )
            if self.settings.enable_diversity:
                score += weights.diversity * float(div_norm[idx])
            final_scores.append(score)
        return final_scores

    def _log_metrics(
        self,
        precision: List[float],
        coverage: List[float],
        match: List[float],
        validity: List[float],
        diversity: List[float],
        final_scores: List[float],
        rmsd_values: List[float],
        coverage_claims: int,
        coverage_possible: int,
        posebusters_passes: int,
        posebusters_checks: int,
    ) -> None:
        self.stats.precision_rewards.extend(precision)
        self.stats.coverage_rewards.extend(coverage)
        self.stats.match_rewards.extend(match)
        self.stats.validity_rewards.extend(validity)
        self.stats.diversity_rewards.extend(diversity)
        self.stats.final_rewards.extend(final_scores)
        self.stats.coverage_claims += coverage_claims
        self.stats.coverage_opportunities += coverage_possible

        if wandb.run is None:
            return

        def _safe_mean(values: List[float]) -> float:
            return float(np.nanmean(values)) if values else 0.0

        log_data = {
            "reward_mc/precision_mean": _safe_mean(precision),
            "reward_mc/coverage_mean": _safe_mean(coverage),
            "reward_mc/match_mean": _safe_mean(match),
            "reward_mc/validity_rate": _safe_mean(validity),
            "reward_mc/diversity_mean": _safe_mean(diversity),
            "reward_mc/final_mean": _safe_mean(final_scores),
            "reward_mc/rmsd_mean": float(np.nanmean(rmsd_values)) if rmsd_values else 0.0,
            "reward_mc/coverage_rate": (
                coverage_claims / coverage_possible if coverage_possible > 0 else 0.0
            ),
            "reward_mc/posebusters_pass_rate": (
                posebusters_passes / posebusters_checks if posebusters_checks > 0 else 0.0
            ),
        }
        wandb.log(log_data)


def multi_component_reward_function(
    prompts,
    completions,
    stats: RunStatistics,
    tokenizer,
    config: Config,
    **kwargs,
):
    """Entry point compatible with the GRPO trainer."""
    calculator = MultiComponentRewardCalculator(config=config, stats=stats, tokenizer=tokenizer)
    return calculator(prompts, completions, **kwargs)
