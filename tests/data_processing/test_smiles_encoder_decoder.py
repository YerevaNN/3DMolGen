import pytest

from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Legacy enriched: bare atoms followed by coordinate payloads.
        ("C<0,0,0>N<1.0,2.0,3.0>", "CN"),
        # Current enriched: bracketed atoms, coordinate payloads should be removed.
        ("[C]<0,0,0>[N]<1,2,3>", "CN"),
        # Non-trivial descriptor should remain bracketed after stripping.
        ("[NH2+]<0.1,0.2,0.3>[C]", "[NH2+]C"),
        # Decorative aromatic carbon should collapse to lowercase.
        ("[cH]<-0.1,0.0,0.1>C", "cC"),
        ("[C][N][C@H]<0.1,0.2,0.3>[c]", "CN[C@H]c"),
        ("[C]<000,000,000>[N]<214,321,123.0>", "CN"),
    ],
)
def test_strip_smiles_handles_legacy_and_new_formats(raw: str, expected: str) -> None:
    actual = strip_smiles(raw)
    assert actual == expected

