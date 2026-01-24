#!/usr/bin/env python3
"""Deterministic harness for grpo_reward_fbeta."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np


class DummyMol:
    def __init__(self, name: str):
        self.name = name

    def GetNumConformers(self):
        return 1


def fake_strip_smiles(text: str | None) -> str:
    return (text or "").replace("[", "").replace("]", "")


def fake_decode_cartesian(text: str | None):
    if text is None or "BAD" in text:
        raise ValueError("decode failed")
    return DummyMol(text)


def fake_same_graph(canonical: str, generated: str) -> bool:
    return canonical in generated


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
smiles_stub.decode_cartesian_binned = fake_decode_cartesian
smiles_stub.strip_smiles = fake_strip_smiles
smiles_stub.get_bins_for_coords = lambda *args, **_kwargs: None
sys.modules["molgen3D.data_processing.smiles_encoder_decoder"] = smiles_stub

eval_utils_stub = types.ModuleType("molgen3D.evaluation.utils")
eval_utils_stub.extract_between = fake_extract_between
eval_utils_stub.same_molecular_graph = fake_same_graph
sys.modules["molgen3D.evaluation.utils"] = eval_utils_stub

utils_stub = types.ModuleType("molgen3D.utils.utils")
utils_stub.get_best_rmsd = lambda *args, **_kwargs: 0.5
utils_stub.load_json = lambda *args, **_kwargs: {}
utils_stub.load_pkl = lambda *args, **_kwargs: {}
sys.modules["molgen3D.utils.utils"] = utils_stub

wandb_stub = types.SimpleNamespace(run=None, log=lambda *args, **kwargs: None)
sys.modules["wandb"] = wandb_stub

from molgen3D.training.grpo import grpo_reward_fbeta as reward_mod


class DummyStats:
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
        self.logged_rmsd = []
        self.successful_generations = 0

    def add_rmsd(self, value: float) -> None:
        self.logged_rmsd.append(float(value))


def build_config():
    grpo = SimpleNamespace(
        fbeta_delta=0.75,
        fbeta_beta=1.5,
        fbeta_gamma=1.0,
        fbeta_dup_rmsd_tau=0.3,
        fbeta_recall_ref_sample=2,
        fbeta_min_valid_to_score=0,
        fbeta_drop_if_valid_lt=1,
        max_ground_truths=3,
        num_generations=2,
        target_conformers=2,
        log_every_steps=1,
        rmsd_workers=0,
        posebusters=None,
    )
    config = SimpleNamespace(grpo=grpo)
    config.seed = 17
    return config


def fake_ground_truths(*_args, **_kwargs):
    return [DummyMol("ref_a"), DummyMol("ref_b"), DummyMol("ref_c")]


def fake_compute_distance_matrix(rollout_mols, reference_mols, validity, **_kwargs):
    K = len(rollout_mols)
    M = len(reference_mols)
    base = np.array([[0.2, 0.9, 1.2], [0.6, 0.3, 0.8]], dtype=np.float32)
    return base[:K, :M]


def fake_posebusters(rollout_mols, base_valid_mask, *_args, **_kwargs):
    summary = {"checked": float(np.count_nonzero(base_valid_mask)), "passed": float(np.count_nonzero(base_valid_mask)), "failed": 0.0, "errors": 0.0, "time_ms": 0.0}
    return base_valid_mask.astype(bool, copy=False), summary


def run_harness() -> None:
    config = build_config()
    stats = DummyStats()
    prompts = ["[SMILES]CCO[/SMILES]", "[SMILES]CCO[/SMILES]"]
    completions = [
        "[CONFORMER]CCO_A[/CONFORMER][CONFORMER]CCO_B[/CONFORMER]",
        "[CONFORMER]CCO_BAD[/CONFORMER]",
    ]

    patches = [
        mock.patch.object(reward_mod, "get_cached_ground_truths", side_effect=fake_ground_truths),
        mock.patch.object(reward_mod, "compute_distance_matrix", side_effect=fake_compute_distance_matrix),
        mock.patch.object(reward_mod, "apply_posebusters_gate", side_effect=fake_posebusters),
        mock.patch.object(reward_mod, "compute_rmsd_safe", side_effect=lambda *args, **_kwargs: 1.0),
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
        assert rewards[0] > 0.0
        assert rewards[1] == 0.0
        print("fbeta harness assertions passed.")
    finally:
        for patch in reversed(patches):
            patch.stop()


if __name__ == "__main__":
    run_harness()
