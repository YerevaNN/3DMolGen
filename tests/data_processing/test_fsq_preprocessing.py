import numpy as np

from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
    MolFSQModel,
    V,
    build_fsq_string,
    parse_fsq_text,
    tokenize_and_encode,
)


def test_build_fsq_string_adds_one_code_per_enriched_token():
    enriched = "[C]<0.0,0.0,0.0>([O]<1.0,0.0,0.0>)"
    fsq = build_fsq_string(enriched, [11, 22, 33, 44])

    assert fsq == "[C]<11>(<22>[O]<33>)<44>"
    tokens, codes = parse_fsq_text(fsq)
    assert tokens == ["[C]", "(", "[O]", ")"]
    np.testing.assert_array_equal(codes, np.array([11, 22, 33, 44]))


def test_tokenize_and_encode_matches_fsq_vocab_shape():
    enriched = "[C]<0.0,1.0,2.0>=[O]<3.0,4.0,5.0>"
    feats, tok_ids = tokenize_and_encode(enriched)

    assert feats.shape == (3, V + 3)
    assert tok_ids.shape == (3,)
    assert feats[:, :V].sum() == 3
    np.testing.assert_array_equal(feats[0, V:], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(feats[1, V:], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(feats[2, V:], np.array([3.0, 4.0, 5.0], dtype=np.float32))


def test_batched_fsq_encoder_emits_one_code_per_input_token():
    model = MolFSQModel(d_model=64, n_layers=1, levels=[2, 2], max_tokens=20)
    model.eval()
    enriched = [
        "[C]<0.0,0.0,0.0>=[O]<1.0,0.0,0.0>",
        "[C]<0.0,0.0,0.0>([O]<1.0,0.0,0.0>)",
    ]

    fsq_texts = model.encode_texts(enriched)

    assert len(fsq_texts) == 2
    for source, fsq_text in zip(enriched, fsq_texts):
        source_tokens = tokenize_and_encode(source)[1]
        fsq_tokens, codes = parse_fsq_text(fsq_text)
        assert len(fsq_tokens) == len(source_tokens)
        assert len(codes) == len(source_tokens)
