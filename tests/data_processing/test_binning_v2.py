import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from molgen3D.data_processing.smiles_encoder_decoder import (
    encode_cartesian_binned_v2,
    decode_cartesian_binned_v2,
    get_bins_for_coords,
    coords_rmsd
)

def test_v2_roundtrip_basic():
    """Test that encode_cartesian_binned_v2 and decode_cartesian_binned_v2 work together."""
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    
    bin_size = 0.1
    ranges = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    # 1. Encode
    enriched, canon_smiles = encode_cartesian_binned_v2(mol, bin_size=bin_size, ranges=ranges)
    
    # Verify format: [C]000000000; (no angle brackets, no commas)
    assert "<" not in enriched
    assert ">" not in enriched
    assert "," not in enriched
    assert ";" in enriched
    
    # 2. Decode
    bins = get_bins_for_coords(ranges, bin_size=bin_size)
    decoded_mol = decode_cartesian_binned_v2(enriched, bins, use_bin_center=True)
    
    # 3. Verify
    assert decoded_mol.GetNumAtoms() == mol.GetNumAtoms()
    assert Chem.MolToSmiles(decoded_mol) == Chem.MolToSmiles(mol)
    
    # RMSD should be small (within bin size limits)
    rmsd = coords_rmsd(mol, decoded_mol)
    # Max error per axis is bin_size/2, so max 3D distance is sqrt(3*(bin_size/2)^2) = bin_size * sqrt(3)/2
    # For bin_size=0.1, max error is ~0.086.
    assert rmsd < bin_size

def test_v2_format_details():
    """Verify the specific string format of v2 encoding."""
    smiles = "C"
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (1.0, 2.0, 3.0))
    mol.AddConformer(conf)
    
    bin_size = 1.0
    ranges = [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)]
    # bins: [0, 1, 2, 3, 4]
    # np.digitize(1.0, [0,1,2,3,4]) -> 2 (since 1.0 is the edge of bin 1 and 2, digitize returns 2 for bins[i-1] <= x < bins[i])
    # Wait, np.digitize(1.0, [0,1,2,3,4]):
    # bins[0]=0, bins[1]=1, bins[2]=2...
    # 1.0 is not < bins[1], but is < bins[2]. So it returns 2.
    
    enriched, _ = encode_cartesian_binned_v2(mol, bin_size=bin_size, ranges=ranges)
    
    # bins are [0,1,2,3,4], length 5. digit_width = max(3, len(str(5))) = 3.
    # indices for (1.0, 2.0, 3.0) with bins [0,1,2,3,4] are (2, 3, 4)
    # So we expect [C]002003004;
    assert enriched == "[C]002003004;"

def test_v2_complex_molecule():
    """Test with a more complex molecule and ring closures."""
    smiles = "c1ccccc1" # Benzene
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    
    bin_size = 0.05
    ranges = [(-5.0, 5.0)] * 3
    
    enriched, _ = encode_cartesian_binned_v2(mol, bin_size=bin_size, ranges=ranges)
    bins = get_bins_for_coords(ranges, bin_size=bin_size)
    decoded_mol = decode_cartesian_binned_v2(enriched, bins)
    
    assert Chem.MolToSmiles(decoded_mol) == Chem.MolToSmiles(mol)
    assert coords_rmsd(mol, decoded_mol) < bin_size

def test_v2_tokenizer_error():
    """Test that the v2 tokenizer raises error on malformed strings."""
    from molgen3D.data_processing.smiles_encoder_decoder import tokenize_enriched_v2
    
    # Missing semicolon
    with pytest.raises(ValueError):
        tokenize_enriched_v2("[C]001002003")
    
    # Unrecognized character
    with pytest.raises(ValueError):
        tokenize_enriched_v2("[C]001002003;?")
    
    # Bad length (not multiple of 3)
    with pytest.raises(ValueError):
        tokenize_enriched_v2("[C]00100200;")
