#!/usr/bin/env python3
"""Deterministic harness for grpo_reward_v3."""

from __future__ import annotations

import sys
import types

import numpy as np
from types import SimpleNamespace
from typing import List, Sequence
from unittest import mock


def fake_strip_smiles(text: str | None) -> str:
    return (text or "").replace("[", "").replace("]", "")


def fake_decode_cartesian(text: str | None):
    return text


def fake_same_graph(canonical: str, generated: str) -> bool:
    return bool(canonical) and canonical in generated


def fake_extract_between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0]


if "rdkit" not in sys.modules:
    rdkit_stub = types.ModuleType("rdkit")
    rdkit_stub.Chem = types.SimpleNamespace()
    sys.modules["rdkit"] = rdkit_stub
    sys.modules["rdkit.Chem"] = rdkit_stub.Chem

smiles_stub = types.ModuleType("molgen3D.data_processing.smiles_encoder_decoder")
smiles_stub.decode_cartesian_v2 = fake_decode_cartesian
smiles_stub.strip_smiles = fake_strip_smiles
sys.modules["molgen3D.data_processing.smiles_encoder_decoder"] = smiles_stub

eval_utils_stub = types.ModuleType("molgen3D.evaluation.utils")
eval_utils_stub.extract_between = fake_extract_between
eval_utils_stub.same_molecular_graph = fake_same_graph
sys.modules["molgen3D.evaluation.utils"] = eval_utils_stub

utils_stub = types.ModuleType("molgen3D.utils.utils")
utils_stub.get_best_rmsd = lambda *args, **_kwargs: 0.0
utils_stub.load_json = lambda *args, **_kwargs: {}
utils_stub.load_pkl = lambda *args, **_kwargs: {}
sys.modules["molgen3D.utils.utils"] = utils_stub

wandb_stub = types.SimpleNamespace(run=None, log=lambda *args, **kwargs: None)
sys.modules["wandb"] = wandb_stub

from molgen3D.training.grpo import grpo_reward_v3 as reward_mod


FAKE_DISTANCE = np.array(
    [
        [0.18, 0.42],
        [0.51, 0.22],
    ],
    dtype=np.float32,
)

EXPECTED_REWARDS = np.array([1.4536245, 1.2474134], dtype=np.float32)


class DummyStats:
    """Minimal stats object compatible with reward_v3."""

    def __init__(self) -> None:
        self.failed_ground_truth = 0
        self.failed_matching_smiles = 0
        self.failed_conformer_generation = 0
        self.failed_rmsd = 0
        self.posebusters_checked = 0
        self.posebusters_failed = 0
        self.posebusters_errors = 0
        self.posebusters_time_ms = 0.0
        self.posebusters_successes = 0
        self.posebusters_failures = 0
        self.processed_prompts = 0
        self.distinct_prompts = 0
        self.global_step = 0
        self.logged_rmsd: List[float] = []

    def add_rmsd(self, value: float) -> None:
        self.logged_rmsd.append(float(value))


def build_config():
    grpo = SimpleNamespace(
        delta=0.5,
        sigma=0.3,
        rho=0.6,
        lambda_qual=1.0,
        lambda_smcov=0.5,
        lambda_match=1.2,
        r_floor=-1.0,
        hard_rmsd_gate=True,
        max_ground_truths=2,
        num_generations=2,
        log_distance_samples_per_group=1,
        profile_rewards=True,
        log_every_steps=1,
        pairwise_rmsd_log_every=50,
        enable_pairwise_rmsd_logging=False,
        reward_seed=777,
        rmsd_workers=0,
    )
    config = SimpleNamespace(grpo=grpo)
    config.seed = 17
    return config


def fake_ground_truths(*_args, **_kwargs):
    return ["ref_a", "ref_b"]


def fake_compute_distance_matrix(
    rollout_mols,
    reference_mols,
    validity,
    **_kwargs,
):
    matrix = np.full(
        (len(rollout_mols), len(reference_mols)),
        np.inf,
        dtype=np.float32,
    )
    valid_indices = np.where(np.asarray(validity).astype(bool))[0]
    for offset, idx in enumerate(valid_indices):
        template_row = FAKE_DISTANCE[min(offset, FAKE_DISTANCE.shape[0] - 1)]
        matrix[idx, :] = template_row
    return matrix


def make_batch() -> tuple[Sequence[str], Sequence[str]]:
    prompts = ["[SMILES]CCO[/SMILES]", "[SMILES]CCO[/SMILES]"]
    completions = [
        "[CONFORMER]CCO_sample_A[/CONFORMER]",
        "[CONFORMER]CCO_sample_B[/CONFORMER]",
    ]
    return prompts, completions


def run_harness() -> None:
    config = build_config()
    stats = DummyStats()
    prompts, completions = make_batch()

    patches = [
        mock.patch.object(reward_mod, "get_cached_ground_truths", side_effect=fake_ground_truths),
        mock.patch.object(reward_mod, "compute_distance_matrix", side_effect=fake_compute_distance_matrix),
    ]

    for patch in patches:
        patch.start()

    try:
        rewards = reward_mod.reward_function(
            prompts=prompts,
            completions=completions,
            stats=stats,
            tokenizer=None,
            config=config,
        )
        np.testing.assert_allclose(rewards, EXPECTED_REWARDS, atol=1e-6)

        profiler = reward_mod.RewardProfiler(enabled=True)
        reward_mod.compute_group_reward(
            canonical_smiles="CCO",
            completions=completions,
            config=config,
            stats=stats,
            profiler=profiler,
            distance_sample_limit=1,
            enable_pairwise_logging=False,
            pairwise_sample_limit=None,
            rng=np.random.default_rng(0),
        )
        print("Profiler breakdown (s):", {k: round(v, 6) for k, v in profiler.sections.items()})
        print("Harness assertions passed.")
    finally:
        for patch in reversed(patches):
            patch.stop()


if __name__ == "__main__":
    run_harness()

