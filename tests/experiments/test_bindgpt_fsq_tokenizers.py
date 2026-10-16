"""Tokenizer contracts for the BindGPT-Coord vs BindGPT-FSQ experiment.

The coordinate spelling and the SMILES tokenization are checked against the
*public* BindGPT tokenizer, not against a remembered specification.
"""

from __future__ import annotations

import numpy as np
import pytest

from molgen3D.experiments.bindgpt_fsq import config, coord_tokens, vocabulary
from molgen3D.experiments.bindgpt_fsq import public_tokenizer as pt
from molgen3D.experiments.bindgpt_fsq.sequences import (
    Arm,
    SequenceFormatError,
    build_coord_stream,
    build_fsq_stream,
    build_sequence,
    parse_coord_stream,
    parse_enriched,
    parse_fsq_stream,
    split_sequence,
)

# Only atom tokens carry a coordinate block; structural tokens stand bare.
ENRICHED = (
    "[C]<1.2345,-0.5,2.0>[C]<-1.0,0.25,-3.75>([O]<0.001,-0.0004,99.9994>)"
    "[N]<-12.0,3.5,-0.125>"
)

pytestmark = pytest.mark.skipif(
    not (config.PUBLIC_TOKENIZER_DIR / "tokenizer.json").exists(),
    reason="run `cli prepare-tokenizers` to pin the public BindGPT tokenizer",
)


@pytest.fixture(scope="module")
def public_files():
    return pt.PublicTokenizerFiles(config.PUBLIC_TOKENIZER_DIR)


@pytest.fixture(scope="module")
def reference(public_files):
    return pt.load_reference_tokenizer(public_files)


@pytest.fixture(scope="module")
def smiles_tokens(public_files):
    return pt.derive_smiles_tokens(public_files)


def test_public_architecture_is_the_pinned_one(public_files):
    architecture = pt.verify_architecture(public_files)
    assert architecture["num_hidden_layers"] == 15
    assert architecture["hidden_size"] == 768
    assert architecture["num_attention_heads"] == 12
    assert architecture["intermediate_size"] == 3072
    assert architecture["max_position_embeddings"] == 2048
    assert architecture["tie_word_embeddings"] is False


def test_every_coordinate_token_exists_upstream(public_files):
    pt.verify_coordinate_tokens(public_files)
    assert len(coord_tokens.integer_tokens()) == 200
    assert len(coord_tokens.fraction_tokens()) == 1000


@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, 1.2345, -1.2345, 0.0004, -0.0004, 99.9994, -99.9994, 12.5, -0.5],
)
def test_scalar_spelling_matches_the_public_tokenizer(reference, value):
    int_token, frac_token = coord_tokens.scalar_to_tokens(value)
    theirs = pt.reference_tokens(reference, f"{int_token}{frac_token}")
    assert theirs == [int_token, frac_token]


@pytest.mark.parametrize(
    "value", [0.0, 1.2345, -1.2345, 0.4999, -0.4999, 42.0006, -42.0006]
)
def test_scalar_round_trip_within_half_a_milliangstrom(value):
    int_token, frac_token = coord_tokens.scalar_to_tokens(value)
    recovered = coord_tokens.tokens_to_scalar(int_token, frac_token)
    assert abs(recovered - value) <= 0.0005 + 1e-12


def test_negative_zero_keeps_its_sign():
    assert coord_tokens.scalar_to_tokens(-0.0004)[0] == "-0"
    assert coord_tokens.scalar_to_tokens(0.0004)[0] == "0"


def test_out_of_range_coordinate_is_refused():
    with pytest.raises(coord_tokens.CoordinateRangeError):
        coord_tokens.scalar_to_tokens(100.0)
    with pytest.raises(coord_tokens.CoordinateRangeError):
        coord_tokens.scalar_to_tokens(float("nan"))


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "Clc1ccccc1Br",
        "COc1ccc(/C=N/NC(=O)[C@H]2[C@@H]3CCCC[C@@H]32)cc1OC",
        "O=C(COn1nnc2ccccc21)NCC1CCCCC1",
        "CS(=O)(=O)C1=[S+]C(C#N)=C2C[C@@H](c3ccccc3)CC([O-])=C12",
        "c1ccc2c(c1)[nH]c1ccccc12",
    ],
)
def test_smiles_tokenization_matches_the_public_tokenizer(reference, smiles_tokens, smiles):
    assert pt.greedy_tokenize(smiles, smiles_tokens) == pt.reference_tokens(
        reference, smiles
    )


def test_adjacent_ring_digits_merge_like_upstream(reference, smiles_tokens):
    # The public trie is longest-match, so "...c21)" yields the integer token
    # "21" rather than "2" then "1"; our tokenizer must do the same.
    tokens = pt.greedy_tokenize("O=C(COn1nnc2ccccc21)NCC1CCCCC1", smiles_tokens)
    assert "21" in tokens


def test_coord_stream_round_trips_the_graph_and_coordinates():
    parsed = parse_enriched(ENRICHED)
    stream = build_coord_stream(parsed)
    graph, coordinates = parse_coord_stream(stream)

    assert graph == [token.text for token in parsed]
    assert coordinates.shape == (4, 3)
    original = np.asarray(
        [token.coords for token in parsed if token.is_atom], dtype=float
    )
    assert np.allclose(coordinates, original, atol=5e-4)
    # Six tokens per atom, none for the parenthesis.
    assert len(stream) == len(parsed) + 6 * 4


def test_fsq_stream_round_trips_graph_and_codes():
    graph = [token.text for token in parse_enriched(ENRICHED)]
    codes = list(range(len(graph)))
    stream = build_fsq_stream(graph, codes)
    recovered_graph, recovered_codes = parse_fsq_stream(stream)

    assert recovered_graph == graph
    assert recovered_codes.tolist() == codes
    # One code per graph token, atom or not.
    assert len(stream) == 2 * len(graph)


def test_fsq_stream_refuses_a_code_count_mismatch():
    with pytest.raises(SequenceFormatError):
        build_fsq_stream(["[C]", "[O]"], [1])


def test_sequence_split_is_the_inverse_of_build():
    smiles = ["C", "C", "O"]
    stream = build_fsq_stream(["[C]", "[C]", "[O]"], [1, 2, 3])
    sequence = build_sequence(smiles, stream)
    assert split_sequence(sequence) == (smiles, stream)


@pytest.mark.parametrize(
    "tokens",
    [
        ["C", "C", "O"],                       # no specials at all
        ["<LIGAND>", "C", "<XYZ>", "<XYZ>"],   # two <XYZ>
        ["<XYZ>", "C", "<LIGAND>"],            # wrong order
    ],
)
def test_malformed_sequences_report_a_missing_special_token(tokens):
    with pytest.raises(SequenceFormatError) as raised:
        split_sequence(tokens)
    assert raised.value.reason.value == "missing_special_token"


def test_union_vocabulary_is_shared_and_masked_per_arm(smiles_tokens):
    union = vocabulary.build_vocabulary(smiles_tokens)
    assert len(set(union.tokens)) == len(union.tokens)
    assert len(union.blocks["fsq_codes"]) == vocabulary.FSQ_CODEBOOK_SIZE
    assert "." in union.blocks["smiles"]

    coord_forbidden = set(union.forbidden_ids(Arm.COORD))
    fsq_forbidden = set(union.forbidden_ids(Arm.FSQ))
    table = union.token_to_id

    # Each arm is denied the other's geometry tokens.
    assert table["<0>"] in coord_forbidden
    assert table[".500"] in fsq_forbidden
    assert table["<0>"] not in fsq_forbidden
    assert table[".500"] not in coord_forbidden

    # Digits and graph tokens stay legal in both arms: SMILES needs the digits,
    # and the repeated graph is identical across arms.
    for token in list("0123456789") + ["21", ".", "[C]", "(", "="]:
        assert table[token] not in coord_forbidden
        assert table[token] not in fsq_forbidden
