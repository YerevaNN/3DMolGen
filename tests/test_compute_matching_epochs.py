import pytest

from molgen3D.training.pretraining.helpers.compute_matching_epochs import (
    compute_matching_epochs,
    measure_character_fraction,
)


def test_compute_matching_epochs_hand_checkable_values():
    # D/x = 6 - 4(1/2) = 4, so:
    # e = (9*100*2 + 50) / (3*100*4) = 1850/1200 = 37/24.
    result = compute_matching_epochs(
        qwen_parameters=100,
        tokenizer_parameters=50,
        our_epochs=2,
        character_fraction=0.5,
    )

    assert result.D_over_x == 4
    assert result.D_prime_over_x == 3
    assert result.e == pytest.approx(37 / 24)


@pytest.mark.parametrize(
    ("N", "N_prime", "e_prime", "a"),
    [
        (0, 1, 1, 0.5),
        (1, -1, 1, 0.5),
        (1, 1, 0, 0.5),
        (1, 1, 1, -0.01),
        (1, 1, 1, 1.01),
    ],
)
def test_compute_matching_epochs_rejects_invalid_inputs(N, N_prime, e_prime, a):
    with pytest.raises(ValueError):
        compute_matching_epochs(N, N_prime, e_prime, a)


def test_character_fraction_uses_coordinate_free_source_tokens(tmp_path):
    dataset = tmp_path / "train.jsonl"
    # Source tokens are [C], =, [O], and 1. The two bracketed atom tokens are
    # excluded, so a = 2/4.
    dataset.write_text(
        '{"canonical_smiles":"C=O1","embedded_smiles":'
        '"[C]<1>=<2>[O]<3>1<4>"}\n',
        encoding="utf-8",
    )

    stats = measure_character_fraction([dataset])

    assert stats.character_tokens == 2
    assert stats.source_tokens == 4
    assert stats.fraction == 0.5
