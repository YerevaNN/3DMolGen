import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from molgen3D.training.grpo import grpo_reward_v3 as reward_mod
from molgen3D.training.grpo.grpo_reward_v3 import compute_matching_reward
from molgen3D.training.grpo.stats import RunStatistics


class DummyMol:
    def __init__(self, name: str):
        self.name = name


@pytest.fixture(autouse=True)
def clear_ground_truth_cache():
    reward_mod._GROUND_TRUTH_CACHE.clear()


@pytest.fixture
def reward_harness(monkeypatch):
    class Harness:
        def __init__(self):
            self.refs = ["ref_a", "ref_b"]
            self.rmsd_map = {}

        def make_config(self, **overrides):
            base = dict(
                delta=0.75,
                sigma=0.25,
                rho=0.75,
                lambda_qual=1.0,
                lambda_smcov=1.0,
                lambda_match=1.0,
                r_floor=-1.0,
                hard_rmsd_gate=True,
                max_ground_truths=len(self.refs),
                num_generations=1,
                profile_rewards=False,
                log_distance_samples_per_group=0,
                enable_pairwise_rmsd_logging=False,
                pairwise_rmsd_log_every=50,
                log_every_steps=1,
                posebusters=SimpleNamespace(mode="off", max_workers=0, chunk_size=100, energy_num_threads=1),
            )
            base.update(overrides)
            return SimpleNamespace(grpo=SimpleNamespace(**base))

        def run(self, tmp_path, prompts, completions, **config_overrides):
            config = self.make_config(**config_overrides)
            stats = RunStatistics(output_dir=str(tmp_path))
            return reward_mod.reward_function(
                prompts=prompts,
                completions=completions,
                stats=stats,
                tokenizer=None,
                config=config,
            )

    harness = Harness()

    def fake_decode(text: str):
        return DummyMol(text) if text else None

    def fake_strip(text: str):
        return "CC" if text else ""

    def fake_graph(canonical: str, generated: str):
        return bool(canonical) and canonical == generated

    def fake_load_gt(smiles: str, num_gt: int):
        return [DummyMol(name) for name in harness.refs[:num_gt]]

    def fake_get_best_rmsd(probe: DummyMol, ref: DummyMol, use_alignmol=True):
        return harness.rmsd_map.get((probe.name, ref.name), float('inf'))

    monkeypatch.setattr(reward_mod, "decode_cartesian_v2", fake_decode)
    monkeypatch.setattr(reward_mod, "strip_smiles", fake_strip)
    monkeypatch.setattr(reward_mod, "same_molecular_graph", fake_graph)
    monkeypatch.setattr(reward_mod, "load_ground_truths", fake_load_gt)
    monkeypatch.setattr(reward_mod, "get_best_rmsd", fake_get_best_rmsd)

    return harness


def test_compute_matching_reward_returns_pairs_and_distances():
    """Sanity-check that matched pairs drive the logged distance stats."""
    D = np.array(
        [
            [0.10, 0.70, 0.95],
            [0.20, 0.60, 0.85],
            [0.55, 0.40, 0.30],
        ],
        dtype=np.float32,
    )
    validity = np.ones(3, dtype=np.int32)
    delta = 0.8

    r_match, num_matched, num_eligible, matched_pairs = compute_matching_reward(
        D, validity, delta
    )

    eligible_mask = validity.astype(bool)[:, None] & (D < delta)
    eligible_dists = D[eligible_mask]

    assert num_matched == len(matched_pairs) == 3
    assert num_eligible == int(np.count_nonzero(eligible_mask))
    assert matched_pairs == [(0, 0), (1, 1), (2, 2)]

    matched_dists = np.array([D[i, j] for (i, j) in matched_pairs], dtype=np.float32)

    # All matched edges must also be eligible, but the reverse should not hold here.
    assert eligible_dists.size > matched_dists.size
    assert not np.array_equal(
        np.sort(eligible_dists),
        np.sort(matched_dists),
    )

    # Sanity: rewards follow delta shaping for each matched edge.
    assert np.all(matched_dists < delta)
    expected_rewards = np.zeros_like(r_match)
    for (i, _), dist in zip(matched_pairs, matched_dists):
        expected_rewards[i] = max(0.0, 1.0 - dist / delta)
    assert np.allclose(r_match, expected_rewards, atol=1e-6)


def test_invalid_geometry_gets_floor_even_without_hard_gate(tmp_path, reward_harness):
    prompts = ["[SMILES]CC[/SMILES]"] * 2
    completions = [
        "[CONFORMER]mol_good[/CONFORMER]",
        "[CONFORMER]mol_bad[/CONFORMER]",
    ]
    reward_harness.rmsd_map = {
        ("mol_good", "ref_a"): 0.25,
        ("mol_good", "ref_b"): 0.4,
    }

    rewards = reward_harness.run(
        tmp_path,
        prompts,
        completions,
        hard_rmsd_gate=False,
        num_generations=len(completions),
    )

    assert rewards[0] > -1.0
    assert rewards[1] == -1.0


def test_reward_function_matches_golden_batch(tmp_path, reward_harness):
    prompts = ["[SMILES]CC[/SMILES]"] * 3
    completions = [
        "[CONFORMER]mol_good[/CONFORMER]",
        "[CONFORMER]mol_mid[/CONFORMER]",
        "[CONFORMER]mol_far[/CONFORMER]",
    ]
    reward_harness.rmsd_map = {
        ("mol_good", "ref_a"): 0.2,
        ("mol_good", "ref_b"): 0.5,
        ("mol_mid", "ref_a"): 0.6,
        ("mol_mid", "ref_b"): 0.4,
        ("mol_far", "ref_a"): 0.95,
        ("mol_far", "ref_b"): 0.9,
    }

    rewards = reward_harness.run(
        tmp_path,
        prompts,
        completions,
        hard_rmsd_gate=True,
        num_generations=len(completions),
    )

    expected = [
        1.4191093444824219,
        0.7860327363014221,
        0.04110811650753021,
    ]
    assert np.allclose(rewards, expected, atol=1e-6)


def test_posebusters_mode_off_skips_runner(tmp_path, reward_harness, monkeypatch):
    """Ensure PoseBusters is not touched when the mode is 'off'."""
    def boom(*_args, **_kwargs):
        raise AssertionError("PoseBusters runner should not be requested when mode=off")

    monkeypatch.setattr(reward_mod, "_get_posebusters_runner", boom)
    prompts = ["[SMILES]CC[/SMILES]"]
    completions = ["[CONFORMER]mol_good[/CONFORMER]"]

    rewards = reward_harness.run(
        tmp_path,
        prompts,
        completions,
        posebusters={"mode": "off"},
        num_generations=1,
    )
    assert len(rewards) == 1


def test_posebusters_basic_masks_only_valid_entries(monkeypatch):
    """PoseBusters should evaluate only base-valid mols and update mask accordingly."""
    captured = {}

    class StubRunner:
        def bust(self, mol_pred, full_report=False):
            captured["mol_pred"] = list(mol_pred)
            return pd.DataFrame(
                {
                    "mol_pred_loaded": [True, False],
                    "sanitization": [True, True],
                }
            )

    monkeypatch.setattr(reward_mod, "_get_posebusters_runner", lambda *_: StubRunner())
    rollout_mols = [DummyMol("good"), None, DummyMol("bad")]
    base_mask = np.array([True, False, True], dtype=bool)
    settings = reward_mod.PoseBustersRuntimeConfig(mode="basic")

    updated_mask, summary = reward_mod.apply_posebusters_gate(rollout_mols, base_mask, settings)

    assert len(captured["mol_pred"]) == 2  # only the True entries
    assert captured["mol_pred"][0].name == "good"
    assert captured["mol_pred"][1].name == "bad"
    assert updated_mask.tolist() == [True, False, False]
    assert summary.checked == 2
    assert summary.passed == 1
    assert summary.failed == 1


def test_posebusters_full_clamps_energy_threads():
    """Full mode with multiprocessing should force energy module threads to 1."""
    settings = reward_mod._normalize_posebusters_config(
        {"mode": "full", "max_workers": 4, "energy_num_threads": 8}
    )
    assert settings.energy_num_threads == 1
    config_dict = reward_mod._build_posebusters_config("full", settings.energy_num_threads)
    energy_module = config_dict["modules"][-1]
    assert energy_module["function"] == "energy_ratio"
    assert energy_module["parameters"]["num_threads"] == 1


def test_posebusters_exception_marks_invalid(monkeypatch):
    """If PoseBusters raises, all checked samples should be flagged as errors."""
    class ExplodingRunner:
        def bust(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(reward_mod, "_get_posebusters_runner", lambda *_: ExplodingRunner())
    rollout_mols = [DummyMol("a"), DummyMol("b")]
    base_mask = np.array([True, True], dtype=bool)
    settings = reward_mod.PoseBustersRuntimeConfig(mode="full")

    updated_mask, summary = reward_mod.apply_posebusters_gate(rollout_mols, base_mask, settings)
    assert updated_mask.tolist() == [False, False]
    assert summary.checked == 2
    assert summary.errors == 2
    assert summary.failed == 0

