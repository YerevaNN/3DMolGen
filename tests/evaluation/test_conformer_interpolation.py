from argparse import Namespace
import pickle

import numpy as np

from molgen3D.data_processing.smiles_encoder_decoder_fsq import MolFSQModel
from molgen3D.evaluation import conformer_interpolation as interpolation


def test_select_anchors_and_pairs_is_reproducible_and_distinct():
    first = interpolation.select_anchors_and_pairs(7, seed=123)
    second = interpolation.select_anchors_and_pairs(7, seed=123)

    assert first == second
    anchors, pairs = first
    assert len(anchors) == 3
    assert len(pairs) == 14
    assert all(a in anchors and b in anchors and a != b for a, b in pairs)
    assert interpolation.select_anchors_and_pairs(3, seed=123) == ([], [])


def test_balanced_local_pairs_prioritize_neighbors_without_repeats():
    class Representation:
        def __init__(self, value):
            self.latents = np.array([[value]], dtype=float)

    anchors = [10, 20, 30, 40]
    representations = [Representation(value) for value in (0.0, 1.0, 4.0, 10.0)]

    pairs = interpolation.select_balanced_local_pairs(
        anchors, representations, target_count=6, seed=123
    )
    repeated = interpolation.select_balanced_local_pairs(
        anchors, representations, target_count=6, seed=123
    )

    assert pairs == repeated
    assert len(pairs) == 6
    assert len({tuple(sorted(pair)) for pair in pairs}) == 6
    assert {tuple(sorted(pair)) for pair in pairs[:3]} == {
        (10, 20),
        (20, 30),
        (30, 40),
    }


def test_balanced_local_pairs_repeat_only_when_unavoidable():
    class Representation:
        def __init__(self, value):
            self.latents = np.array([[value]], dtype=float)

    pairs = interpolation.select_balanced_local_pairs(
        [0, 1, 2],
        [Representation(0), Representation(1), Representation(2)],
        target_count=7,
        seed=5,
    )

    assert len(pairs) == 7
    assert len({tuple(sorted(pair)) for pair in pairs[:3]}) == 3


def test_random_bit_interpolation_is_seeded_and_balanced():
    first = np.array([0b0000, 0b1111, 0b1010])
    second = np.array([0b1111, 0b0000, 0b0101])

    result = interpolation.random_balanced_bit_interpolation(
        first, second, rng=np.random.default_rng(123), width=4
    )
    repeated = interpolation.random_balanced_bit_interpolation(
        first, second, rng=np.random.default_rng(123), width=4
    )
    other_seed = interpolation.random_balanced_bit_interpolation(
        first, second, rng=np.random.default_rng(124), width=4
    )

    np.testing.assert_array_equal(result, repeated)
    assert not np.array_equal(result, other_seed)
    assert all(int(value).bit_count() == 2 for value in (result ^ first))


def test_deterministic_bit_interpolation_alternates_differences():
    first = np.array([0b0000, 0b1111, 0b1010])
    second = np.array([0b1111, 0b0000, 0b0101])

    result = interpolation.alternating_bit_interpolation(first, second, width=4)

    np.testing.assert_array_equal(result, [0b0101, 0b1010, 0b1111])


def test_fsq_model_exposes_continuous_and_discrete_batch_paths():
    model = MolFSQModel(d_model=64, n_layers=1, levels=[2] * 12, max_tokens=20)
    model.eval()
    enriched = [
        "[C]<0.0,0.0,0.0>[O]<1.0,0.0,0.0>",
        "[C]<0.0,0.1,0.0>[O]<1.0,-0.1,0.0>",
    ]

    representations = model.encode_representations(enriched)

    assert len(representations) == 2
    assert representations[0].tokens == representations[1].tokens
    assert representations[0].latents.shape == (2, 12)
    assert representations[0].codes.shape == (2,)

    midpoint = np.stack([0.5 * representations[0].latents + 0.5 * representations[1].latents])
    codes = np.stack(
        [
            interpolation.random_balanced_bit_interpolation(
                representations[0].codes,
                representations[1].codes,
                rng=np.random.default_rng(7),
            )
        ]
    )
    assert len(model.decode_tokens_with_latent_batch(representations[0].tokens, midpoint)) == 1
    assert len(model.decode_tokens_with_code_batch(representations[0].tokens, codes)) == 1


def _aggregate(cov_r: float, cov_p: float):
    return {
        "thresholds": np.array([0.75]),
        "CoverageR": np.array([[cov_r]]),
        "CoverageP": np.array([[cov_p]]),
        "MatchingR": np.array([0.5]),
        "MatchingP": np.array([0.5]),
    }


def test_stage_one_gate_requires_all_quality_checks():
    decision = interpolation.decide_stage_one(
        aggregates={
            "anchors": _aggregate(0.50, 0.80),
            "latent": _aggregate(0.60, 0.77),
            "token": _aggregate(0.70, 0.60),
            "random_token": _aggregate(0.65, 0.74),
        },
        rates={
            "latent": {"validity_rate": 0.95, "decode_success_rate": 0.96},
            "token": {"validity_rate": 0.99, "decode_success_rate": 0.99},
            "random_token": {
                "validity_rate": 0.99,
                "decode_success_rate": 0.99,
            },
        },
        threshold=0.75,
        max_precision_drop=0.05,
        min_validity_rate=0.90,
        min_decode_success_rate=0.90,
    )

    assert decision["passed"] is True
    assert decision["methods"]["latent"]["passed"] is True
    assert decision["methods"]["token"]["passed"] is False
    assert decision["methods"]["token"]["checks"]["precision_acceptable"] is False


def test_chunked_decode_preserves_pair_alignment_after_failure():
    values = np.arange(5)

    def decode(batch):
        if 2 in batch:
            raise RuntimeError("synthetic batch failure")
        return [str(value) for value in batch]

    decoded = interpolation._decode_in_chunks(values, batch_size=2, decode=decode)

    assert decoded == ["0", "1", None, None, "4"]


def test_stage_one_resume_reuses_incremental_record(tmp_path, monkeypatch):
    input_path = tmp_path / "ground_truth.pkl"
    output_dir = tmp_path / "out"
    molecule_key = "C"
    with input_path.open("wb") as handle:
        pickle.dump({molecule_key: {"confs": []}}, handle)

    metric = {
        "n_true": 1,
        "n_model": 1,
        "coverage_recall": np.ones_like(interpolation.DEFAULT_THRESHOLDS),
        "coverage_precision": np.ones_like(interpolation.DEFAULT_THRESHOLDS),
        "matching_recall": 0.0,
        "matching_precision": 0.0,
    }
    record = {
        "molecule_key": molecule_key,
        "status": "complete",
        "reference_count": 1,
        "anchor_indices": [0],
        "metrics": {condition: metric for condition in interpolation.CONDITIONS},
        "statistics": {
            condition: {
                "attempted": 1,
                "decoded": 1,
                "valid": 1,
                "decode_failures": 0,
                "decode_success_rate": 1.0,
                "validity_rate": 1.0,
                "duplicate_rate": 0.0,
            }
            for condition in interpolation.INTERPOLATION_CONDITIONS
        },
    }
    interpolation._atomic_pickle_dump(record, interpolation._record_path(output_dir, molecule_key))

    class FakeModel:
        device = "cpu"

    monkeypatch.setattr(interpolation, "configure_fsq_encoder", lambda **_: FakeModel())

    def should_not_run(**_):
        raise AssertionError("process_molecule should not run for a resumed record")

    monkeypatch.setattr(interpolation, "process_molecule", should_not_run)
    args = Namespace(
        input=str(input_path),
        output_dir=str(output_dir),
        fsq_checkpoint=str(tmp_path / "model.ckpt"),
        device="cpu",
        max_tokens=218,
        max_molecules=None,
        phase="all",
        model_batch_size=8,
        num_workers=1,
        seed=5,
        duplicate_rmsd_threshold=1e-3,
        gate_threshold=0.75,
        max_precision_drop=0.05,
        min_validity_rate=0.90,
        min_decode_success_rate=0.90,
        resume=True,
        log_every=1,
    )

    summary = interpolation.run_stage_one(args)

    assert summary["status_counts"] == {"complete": 1}
