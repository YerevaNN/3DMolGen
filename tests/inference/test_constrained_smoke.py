import pytest

from molgen3D.evaluation.constrained_smoke import (
    build_prompts,
    sample_smiles,
    validate_smoke_outputs,
)


def test_build_prompts_wraps_smiles():
    """Test that the build_prompts function wraps SMILES strings in [SMILES] and [/SMILES] blocks."""
    prompts = build_prompts(["CC", "O"])
    assert prompts == ["[SMILES]CC[/SMILES]", "[SMILES]O[/SMILES]"]


def test_sample_smiles_returns_unique_subset():
    """Test that the sample_smiles function returns a unique subset of SMILES strings."""
    fake_gt = {f"SMI{i}": {} for i in range(10)}
    subset = sample_smiles(fake_gt, 5, seed=123)
    assert len(subset) == 5
    assert len(set(subset)) == 5
    assert subset == sample_smiles(fake_gt, 5, seed=123)


def _fake_conformer(smiles: str) -> str:
    """Create a fake conformer string for a given SMILES string."""
    blocks = "".join(f"[{atom}]<0,0,0>" for atom in smiles)
    return f"[SMILES]{smiles}[/SMILES][CONFORMER]{blocks}[/CONFORMER]"


def test_validate_smoke_outputs_passes_when_strings_match():
    """Test that the validate_smoke_outputs function passes when the SMILES strings match."""
    smiles = ["CC"]
    decoded = [_fake_conformer("CC")]
    result = validate_smoke_outputs(smiles, decoded)
    assert result.total == 1
    assert result.num_passed == 1
    result.raise_for_failures()


def test_validate_smoke_outputs_flags_smiles_mismatch():
    """Test that the validate_smoke_outputs function flags when the SMILES strings mismatch."""
    smiles = ["CC"]
    decoded = [_fake_conformer("CN")]
    result = validate_smoke_outputs(smiles, decoded)
    assert result.failures
    assert "SMILES block mismatch" in result.failures[0].issues
    with pytest.raises(AssertionError):
        result.raise_for_failures()


def test_validate_smoke_outputs_flags_missing_conformer():
    """Test that the validate_smoke_outputs function flags when the conformer block is missing."""
    smiles = ["CC"]
    decoded = ["[SMILES]CC[/SMILES]"]
    result = validate_smoke_outputs(smiles, decoded)
    assert "missing [CONFORMER] block" in result.failures[0].issues
