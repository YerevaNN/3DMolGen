"""Matching, batching and failure-attribution contracts.

These are the invariants the experiment's causal claim rests on: both arms see
the same examples in the same optimizer step, and every rejected generation is
attributed a reason from the closed vocabulary.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from molgen3D.experiments.bindgpt_fsq import dataset, vocabulary
from molgen3D.experiments.bindgpt_fsq.data.identity import (
    Exclusion,
    identity_from_graph,
)
from molgen3D.experiments.bindgpt_fsq.decode import decode_generation
from molgen3D.experiments.bindgpt_fsq.sequences import (
    PAD,
    UNK,
    XYZ,
    Arm,
    build_coord_stream,
    build_fsq_stream,
    build_sequence,
    parse_enriched,
    render,
)

ETHANOL = "[C]<0.0,0.0,0.0>[C]<1.5,0.0,0.0>[O]<2.0,1.4,0.0>"
BENZENE_FRAGMENT = "[c]<0.0,1.4,0.0>1[c]<1.2,0.7,0.0>[c]<1.2,-0.7,0.0>"


def _write_pair(tmp_path, enriched_list, smiles_tokens):
    """Write matched coord/fsq example shards, as the builder would."""
    coord_path = tmp_path / "coord_000.jsonl"
    fsq_path = tmp_path / "fsq_000.jsonl"
    with open(coord_path, "w") as coord, open(fsq_path, "w") as fsq:
        for row, enriched in enumerate(enriched_list):
            parsed = parse_enriched(enriched)
            graph = [token.text for token in parsed]
            codes = [(row * 7 + index) % 4096 for index in range(len(graph))]
            payloads = {
                "coord": build_sequence(smiles_tokens, build_coord_stream(parsed)),
                "fsq": build_sequence(smiles_tokens, build_fsq_stream(graph, codes)),
            }
            for handle, arm in ((coord, "coord"), (fsq, "fsq")):
                handle.write(
                    json.dumps(
                        {
                            "example_id": f"train:000:{row:07d}",
                            "shard": 0,
                            "row": row,
                            "molecule_id": f"train:{row:07d}",
                            "tokens": render(payloads[arm]),
                            "n_tokens": len(payloads[arm]),
                        }
                    )
                    + "\n"
                )
    return coord_path, fsq_path


@pytest.fixture(scope="module")
def union():
    return vocabulary.build_vocabulary()


@pytest.fixture
def packed(tmp_path, union):
    enriched_list = [ETHANOL, BENZENE_FRAGMENT] * 16
    coord_path, fsq_path = _write_pair(tmp_path, enriched_list, ["C", "C", "O"])
    table = union.token_to_id
    for path, stem in ((coord_path, "coord"), (fsq_path, "fsq")):
        dataset.pack_binary([path], tmp_path / "packed", table, table[UNK], stem)
    return (
        dataset.PackedExamples(tmp_path / "packed", "coord"),
        dataset.PackedExamples(tmp_path / "packed", "fsq"),
    )


def test_packing_is_lossless_and_paired(packed, union):
    coord, fsq = packed
    assert len(coord) == len(fsq) == 32
    assert coord.example_ids() == fsq.example_ids()
    # The FSQ arm is the shorter representation for the same molecules.
    assert fsq.lengths.sum() < coord.lengths.sum()
    report = json.loads(
        (coord.directory / "coord.pack_report.json").read_text()
    )
    assert report["unknown_tokens"] == 0


def test_epoch_order_ignores_sequence_length(packed):
    coord, fsq = packed
    left = dataset.epoch_order(len(coord), seed=17, epoch=0)
    right = dataset.epoch_order(len(fsq), seed=17, epoch=0)
    assert np.array_equal(left, right)
    # A different epoch must reshuffle, a different seed must differ.
    assert not np.array_equal(left, dataset.epoch_order(len(coord), 17, 1))
    assert not np.array_equal(left, dataset.epoch_order(len(coord), 23, 0))


def test_both_arms_see_the_same_examples_in_the_same_step(packed, union):
    coord, fsq = packed
    table = union.token_to_id
    order = dataset.epoch_order(len(coord), seed=17, epoch=0)

    def membership(packed_arm):
        steps = {}
        for step, micro_batches in dataset.iter_global_batches(
            packed_arm, order, 8, 3, table[PAD], table[XYZ]
        ):
            steps[step] = sorted(
                int(index)
                for micro in micro_batches
                for index in micro.example_indices
            )
        return steps

    assert membership(coord) == membership(fsq)


def test_micro_batches_are_length_sorted_but_membership_is_not(packed, union):
    coord, _ = packed
    table = union.token_to_id
    order = dataset.epoch_order(len(coord), seed=17, epoch=0)
    _, micro_batches = next(
        iter(dataset.iter_global_batches(coord, order, 8, 3, table[PAD], table[XYZ]))
    )
    lengths = [
        int(coord.lengths[index])
        for micro in micro_batches
        for index in micro.example_indices
    ]
    assert lengths == sorted(lengths)


def test_labels_ignore_padding_and_geometry_mask_starts_after_xyz(packed, union):
    coord, _ = packed
    table = union.token_to_id
    order = np.arange(len(coord))
    _, micro_batches = next(
        iter(dataset.iter_global_batches(coord, order, 6, 6, table[PAD], table[XYZ]))
    )
    micro = micro_batches[0]
    padded = micro.attention_mask == 0
    assert bool((micro.labels[padded] == dataset.LABEL_IGNORE).all())
    # The <XYZ> separator itself is prefix, not geometry.
    assert not bool(micro.geometry_mask[micro.input_ids == table[XYZ]].any())
    # Nothing before <XYZ> is geometry.
    for row in range(micro.input_ids.size(0)):
        cut = int((micro.input_ids[row] == table[XYZ]).nonzero()[0])
        assert not bool(micro.geometry_mask[row, : cut + 1].any())
        assert bool(micro.geometry_mask[row, cut + 1 :].any())


@pytest.mark.parametrize(
    "graph, reason",
    [
        # Existing SFT data keeps disconnected SMILES, so overlap/fragment
        # metadata must never become a training exclusion.
        ("[C]<0,0,0>.[O]<1,1,1>", Exclusion.NONE.value),
        ("[C]<0,0,0>[C]<1,0,0>[O]<2,1,0>", Exclusion.NONE.value),
    ],
)
def test_identity_marks_disconnected_molecules(graph, reason):
    from molgen3D.experiments.bindgpt_fsq.data.extract_source import strip_coordinates

    identity = identity_from_graph(strip_coordinates(graph))
    assert identity.exclusion_reason == reason
    assert identity.standardized_isomeric_smiles


def test_disconnected_identity_is_usable_and_preserves_fragment_count():
    from molgen3D.experiments.bindgpt_fsq.data.extract_source import strip_coordinates

    identity = identity_from_graph(strip_coordinates("[C]<0,0,0>.[O]<1,1,1>"))
    assert identity.usable
    assert identity.num_fragments == 2


def test_coord_generation_decodes_to_a_molecule_with_a_conformer():
    parsed = parse_enriched(ETHANOL)
    tokens = build_sequence(["C", "C", "O"], build_coord_stream(parsed))
    result = decode_generation(tokens, Arm.COORD, layout="interleaved")
    assert result.ok, result.failure_reason
    assert result.canonical_smiles == "CCO"
    assert result.connected is True
    assert result.mol.GetNumConformers() == 1
    assert result.coordinates.shape == (3, 3)


def test_graph_mismatch_between_the_two_halves_is_reported():
    parsed = parse_enriched(ETHANOL)
    # Left says propane, right says ethanol.
    tokens = build_sequence(["C", "C", "C"], build_coord_stream(parsed))
    result = decode_generation(tokens, Arm.COORD, layout="interleaved")
    assert result.failure_reason == "graph_mismatch"
    assert result.graph_consistent is False


def test_prompt_mismatch_is_reported_for_conditional_generation():
    parsed = parse_enriched(ETHANOL)
    tokens = build_sequence(["C", "C", "O"], build_coord_stream(parsed))
    result = decode_generation(
        tokens, Arm.COORD, prompt_smiles="CCC", layout="interleaved"
    )
    assert result.matches_prompt is False
    assert result.failure_reason == "graph_mismatch"


def test_truncated_coordinate_triplet_is_a_decode_failure():
    parsed = parse_enriched(ETHANOL)
    stream = build_coord_stream(parsed)
    result = decode_generation(
        build_sequence(["C", "C", "O"], stream[:-3]), Arm.COORD, layout="interleaved"
    )
    assert result.failure_reason == "coordinate_decode_failure"


def test_length_limit_is_distinguished_from_a_malformed_sequence():
    result = decode_generation(
        ["<LIGAND>", "C", "C"], Arm.COORD, hit_length_limit=True, layout="interleaved"
    )
    assert result.failure_reason == "sequence_length_limit"


def test_fsq_arm_without_the_frozen_tokenizer_fails_cleanly():
    graph = [token.text for token in parse_enriched(ETHANOL)]
    tokens = build_sequence(["C", "C", "O"], build_fsq_stream(graph, [1, 2, 3]))
    result = decode_generation(
        tokens, Arm.FSQ, fsq_tokenizer=None, layout="interleaved"
    )
    assert result.failure_reason == "coordinate_decode_failure"


# --------------------------------------------------------------------------- #
# Suffix layout -- published BindGPT's structure
# --------------------------------------------------------------------------- #


def test_rebracket_and_unbracket_are_inverse():
    from molgen3D.experiments.bindgpt_fsq.sequences import rebracket, unbracket

    for smiles in ("CCO", "CN(CCO)c1cnccn1", "Clc1ccccc1Br", "C[C@@H](N)[O-]", "c1cc[nH]c1"):
        graph = rebracket(smiles)
        assert unbracket(graph) == smiles
        assert all(token for token in graph)


def test_rebracket_matches_the_enriched_graph_tokens():
    """The suffix layout is only possible because this holds."""
    from molgen3D.experiments.bindgpt_fsq.sequences import rebracket
    from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles

    for enriched in (ETHANOL, BENZENE_FRAGMENT):
        parsed = parse_enriched(enriched)
        graph = [token.text for token in parsed]
        plain = strip_smiles("".join(graph))
        assert rebracket(plain) == graph


def test_coord_suffix_round_trips_without_transmitting_the_graph():
    from molgen3D.experiments.bindgpt_fsq.sequences import (
        atom_coordinates, build_coord_suffix, parse_coord_suffix, rebracket, unbracket,
    )

    parsed = parse_enriched(ETHANOL)
    graph = [token.text for token in parsed]
    suffix = build_coord_suffix(parsed)
    # six tokens per atom, and nothing else -- no graph in the suffix
    assert len(suffix) == 6 * sum(token.is_atom for token in parsed)
    recovered = parse_coord_suffix(suffix, rebracket(unbracket(graph)))
    assert np.allclose(recovered, atom_coordinates(parsed), atol=5e-4)


def test_fsq_suffix_round_trips_and_is_one_code_per_graph_token():
    from molgen3D.experiments.bindgpt_fsq.sequences import (
        build_fsq_suffix, parse_fsq_suffix, rebracket, unbracket,
    )

    parsed = parse_enriched(BENZENE_FRAGMENT)
    graph = [token.text for token in parsed]
    codes = list(range(len(graph)))
    suffix = build_fsq_suffix(graph, codes)
    assert len(suffix) == len(graph)          # includes non-atom tokens
    assert all(t.startswith("<") for t in suffix)   # bare codes, no graph
    assert parse_fsq_suffix(suffix, rebracket(unbracket(graph))).tolist() == codes


def test_suffix_layout_is_shorter_than_interleaved():
    from molgen3D.experiments.bindgpt_fsq.sequences import (
        build_coord_stream, build_coord_suffix, build_fsq_stream, build_fsq_suffix,
        build_sequence,
    )

    parsed = parse_enriched(ETHANOL)
    graph = [token.text for token in parsed]
    codes = [1, 2, 3]
    smiles = ["C", "C", "O"]
    for suffix, stream in (
        (build_coord_suffix(parsed), build_coord_stream(parsed)),
        (build_fsq_suffix(graph, codes), build_fsq_stream(graph, codes)),
    ):
        assert len(build_sequence(smiles, suffix)) < len(build_sequence(smiles, stream))


@pytest.mark.parametrize("arm", [Arm.COORD, Arm.FSQ])
def test_suffix_layout_reports_atom_count_mismatch_on_a_short_suffix(arm):
    from molgen3D.experiments.bindgpt_fsq.sequences import (
        build_coord_suffix, build_fsq_suffix, build_sequence,
    )

    parsed = parse_enriched(ETHANOL)
    graph = [token.text for token in parsed]
    suffix = (
        build_coord_suffix(parsed) if arm is Arm.COORD
        else build_fsq_suffix(graph, [1, 2, 3])
    )
    result = decode_generation(
        build_sequence(["C", "C", "O"], suffix[:-1]), arm, layout="suffix"
    )
    assert result.failure_reason == "atom_count_mismatch"


def test_suffix_layout_cannot_produce_a_graph_mismatch():
    """The graph is derived from the SMILES, so the failure mode is gone."""
    from molgen3D.experiments.bindgpt_fsq.sequences import build_coord_suffix, build_sequence

    parsed = parse_enriched(ETHANOL)
    result = decode_generation(
        build_sequence(["C", "C", "O"], build_coord_suffix(parsed)),
        Arm.COORD, layout="suffix",
    )
    assert result.ok and result.graph_consistent is True
