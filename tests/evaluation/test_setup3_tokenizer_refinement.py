"""Unit tests for the Setup 3 tokenizer-refinement pass.

The FSQ model is stubbed, so these exercise the batching, token grouping and
failure-fall-through logic without a checkpoint or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from molgen3D.data_processing.smiles_encoder_decoder import encode_cartesian_v2
from molgen3D.data_processing.smiles_encoder_decoder_fsq import FSQRepresentation
from molgen3D.evaluation.setup3_tokenizer_refinement import (
    gate_interpolants,
    gated_condition_names,
    process_record,
    refine_molecules,
    refined_condition_names,
)


def _conformer(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, parameters) == 0
    return Chem.RemoveHs(mol)


class StubModel:
    """Emits fixed coordinates per token, and fails on nominated inputs."""

    def __init__(self, *, shift: float = 0.05, fail_texts: set[str] | None = None):
        self.shift = shift
        self.fail_texts = fail_texts or set()
        self.encode_calls: list[int] = []
        self.decode_calls: list[int] = []

    def encode_representations(self, texts):
        self.encode_calls.append(len(texts))
        out = []
        for text in texts:
            if text in self.fail_texts:
                raise RuntimeError("stub encode failure")
            tokens = _enriched_tokens(text)
            out.append(
                FSQRepresentation(
                    tokens=tokens,
                    latents=np.zeros((len(tokens), 12), dtype=np.float32),
                    codes=np.zeros(len(tokens), dtype=np.int64),
                )
            )
        return out

    def decode_tokens_with_code_batch(self, tokens, codes, precision: int = 4):
        self.decode_calls.append(len(codes))
        atom_positions = [i for i, token in enumerate(tokens) if token.startswith("[")]
        text = []
        for row in range(len(codes)):
            parts = []
            offset = 0
            for token in tokens:
                if token.startswith("["):
                    value = self.shift * (offset + 1)
                    parts.append(f"{token}<{value:.4f},{value:.4f},{value:.4f}>")
                    offset += 1
                else:
                    parts.append(token)
            assert offset == len(atom_positions)
            text.append("".join(parts))
        return text


def _enriched_tokens(text: str) -> list[str]:
    from molgen3D.data_processing.smiles_encoder_decoder_fsq import tokenize_enriched

    return [
        str(token.get("atom_desc", token.get("text")))
        for token in tokenize_enriched(text)
    ]


def test_refine_molecules_replaces_every_conformer():
    molecules = [_conformer("CCO", seed) for seed in (1, 2, 3)]
    model = StubModel()

    refined, stats = refine_molecules(model, molecules, batch_size=2)

    assert stats == {"n": 3, "encode_failures": 0, "decode_failures": 0, "token_groups": 1}
    # One token group, so encode and decode chunk the same way.
    assert model.encode_calls == [2, 1]
    assert model.decode_calls == [2, 1]
    for original, result in zip(molecules, refined):
        assert result is not original
        assert result.GetNumAtoms() == original.GetNumAtoms()


def test_encode_failure_keeps_the_original_conformer():
    molecules = [_conformer("CCO", seed) for seed in (1, 2, 3)]
    doomed, _ = encode_cartesian_v2(molecules[1])
    model = StubModel(fail_texts={doomed})

    refined, stats = refine_molecules(model, molecules, batch_size=3)

    # The whole batch fails first, then the retry isolates the one bad input.
    assert model.encode_calls == [3, 1, 1, 1]
    assert stats["encode_failures"] == 1
    assert stats["n"] == 3
    assert refined[1] is molecules[1]
    assert refined[0] is not molecules[0]
    assert refined[2] is not molecules[2]


def test_separate_token_sequences_decode_separately():
    molecules = [_conformer("CCO", 1), _conformer("CCC", 2), _conformer("CCO", 3)]
    model = StubModel()

    _, stats = refine_molecules(model, molecules, batch_size=8)

    assert stats["token_groups"] == 2
    assert sorted(model.decode_calls) == [1, 2]


def test_empty_input_is_a_no_op():
    model = StubModel()
    refined, stats = refine_molecules(model, [], batch_size=4)
    assert refined == []
    assert stats["n"] == 0
    assert model.encode_calls == []


def test_process_record_builds_both_policies():
    anchors = [_conformer("CCO", seed) for seed in (1, 2)]
    interpolants = [_conformer("CCO", seed) for seed in (3, 4)]
    record = {
        "molecule_key": "mol",
        "k": 2,
        "n_anchor": 2,
        "variant": "full",
        "conformers": {
            "anchors": anchors,
            "latent": anchors + interpolants,
            "cartesian": anchors + interpolants[:1],
        },
        "metrics": {"latent": None},
    }

    refined = process_record(
        StubModel(), record, source_conditions=("latent", "cartesian"), batch_size=4
    )

    assert set(refined["conformers"]) == set(
        refined_condition_names(("latent", "cartesian"))
    )
    # rt_interp keeps the original anchor objects; rt_all replaces them.
    assert refined["conformers"]["latent_rt_interp"][:2] == anchors
    assert all(
        molecule not in anchors for molecule in refined["conformers"]["latent_rt_all"][:2]
    )
    assert refined["conformers"]["latent_rt_all"][2:] == (
        refined["conformers"]["latent_rt_interp"][2:]
    )
    assert len(refined["conformers"]["cartesian_rt_all"]) == 3
    assert refined["source_metrics"] == {"latent": None}


def test_gate_keeps_originals_that_already_passed():
    before = ["a0", "a1", "a2", "a3"]
    after = ["b0", "b1", "b2", "b3"]

    assert gate_interpolants(before, after, "1010") == ["a0", "b1", "a2", "b3"]
    assert gate_interpolants(before, after, "1111") == before
    assert gate_interpolants(before, after, "0000") == after


def test_gate_rejects_misaligned_flags():
    with pytest.raises(ValueError, match="Gate inputs disagree"):
        gate_interpolants(["a", "b"], ["c", "d"], "1")


def test_gated_condition_names():
    assert gated_condition_names(("latent", "cartesian")) == (
        "latent_gated",
        "cartesian_gated",
    )


def test_process_record_rejects_an_anchor_count_mismatch():
    anchors = [_conformer("CCO", 1)]
    record = {
        "molecule_key": "mol",
        "k": 2,
        "n_anchor": 2,
        "variant": "full",
        "conformers": {"anchors": anchors, "latent": anchors},
    }
    with pytest.raises(ValueError, match="expected 2 anchors"):
        process_record(StubModel(), record, source_conditions=("latent",), batch_size=4)
