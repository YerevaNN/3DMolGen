import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pytest
import torch
from rdkit import Chem
from molgen3D.data_processing.coordtoken import (
    CoordTokenCodec,
    parse,
    graph_mol,
    FORMAT,
    SELECTED_CHECKPOINT,
)
from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2
from molgen3D.data_processing.preprocess_coordtoken_csv import validate_tokenizer
from molgen3D.data_processing.coordtoken_contract import validate_record, load_contract

ROOT = Path(__file__).resolve().parents[2]
TEXTS = [
    "[C]<0,0,0>[O]<1.4,0.2,-0.1>",
    "[C]<0,0,0>(=[O]<1.2,0,0>)[O]<-1.3,0.1,0>",
    "[C]<0,0,0>1[C]<1,0,0>[C]<1,1,0>1",
    "[Na]<0,0,0>.[Cl]<2,0,0>",
]
TOKENIZER = ROOT / "src/molgen3D/training/tokenizers/Qwen3_tokenizer_coordtoken_atom12_reg1"


def test_layout_zero_punctuation_and_stereo():
    s = parse("<0>[C@H]<0>([F]<1>)[C]<4095>1=[C]<2>[C]<3>1")
    assert s.molecular_code == 0 and s.atom_codes == (0, 1, 4095, 2, 3)
    assert s.serialize() == "<0>[C@H]<0>([F]<1>)[C]<4095>1=[C]<2>[C]<3>1"


@pytest.mark.parametrize(
    "s",
    [
        "[C]<1>",
        "<0><1>[C]<2>",
        "<4096>[C]<1>",
        "<0>[C]<4096>",
        "<0>[C]",
        "<0>[C]<1>(<2>[O]<3>)",
        "<0>[C]<1><2>",
        "<0>[C]<-1>",
        "<0>[C]<1>garbage",
        "<0>",
        "<0> C<1>",
        "<0>[C]<1.2>",
        "<00>[C]<1>",
    ],
)
def test_malformed_streams(s):
    with pytest.raises(ValueError):
        parse(s)


def test_stereo_and_explicit_h():
    mol = decode_cartesian_v2("[F]<0,0,0>/[C]<1,0,0>=[C]<2,0,0>/[F]<3,0,0>")
    assert any(b.GetStereo() == Chem.BondStereo.STEREOE for b in mol.GetBonds())
    mol = decode_cartesian_v2("[H]<0,0,0>[O]<1,0,0>[H]<2,0,0>")
    assert mol.GetNumAtoms() == 3 and mol.GetConformer().GetAtomPosition(2).x == 2
    with pytest.raises(ValueError):
        decode_cartesian_v2("[C]<nan,0,0>")
    with pytest.raises(ValueError):
        decode_cartesian_v2("[C]<0,0,0>garbage")


@pytest.fixture(scope="module")
def codec():
    if not Path(SELECTED_CHECKPOINT).exists():
        pytest.skip("Selected shared-storage checkpoint unavailable")
    torch.set_num_threads(4)
    return CoordTokenCodec(device="cpu")


def test_strict_checkpoint(codec, tmp_path):
    with pytest.raises(FileNotFoundError):
        CoordTokenCodec(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="SHA256"):
        CoordTokenCodec(SELECTED_CHECKPOINT, expected_sha256="0" * 64)


def test_encode_decode_dense_parity_and_batch_layout(codec):
    batch, records = codec.prepare(TEXTS)
    with torch.inference_mode():
        xyz, indices = codec.model(**batch)
    streams = codec.encode_batch(TEXTS)
    decoded = codec.decode_batch(streams)
    for i, (s, got) in enumerate(decoded):
        assert s.molecular_code == int(indices[i, -1])
        positions = list(s.atom_positions)
        np.testing.assert_allclose(got[positions], xyz[i, positions].numpy(), rtol=1e-5, atol=1e-5)
        # Register slot follows padded width, never the final atom or punctuation.
        assert len(s.atom_codes) == int(batch["atoms"][i].sum())
        assert len(s.atom_codes) + 1 == len(__import__("re").findall(r"<\d+>", streams[i]))
        mol = codec.decode_mol(streams[i])
        assert mol.GetNumAtoms() == len(positions)
        np.testing.assert_allclose(
            mol.GetConformer().GetPositions(), got[positions], atol=1e-4, rtol=1e-4
        )
    # Different padding lengths do not move the molecular register into graph slots.
    for text, stream in zip(TEXTS, streams):
        assert codec.encode_text(text) == stream
    with pytest.raises(ValueError):
        codec.encode_text("[C]<inf,0,0>")


def test_port_matches_frozen_source(codec):
    frozen = ROOT / "coordtoken/research/mixed_epoch2_20261014_v3"
    sys.path.insert(0, str(frozen))
    try:
        spec = importlib.util.spec_from_file_location(
            "frozen_coordtoken_model", frozen / "model.py"
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        original = m.Tokenizer(m.Config(**codec.model.config.__dict__))
        original.load_state_dict(codec.model.state_dict(), strict=True)
        original.eval()
        batch, _ = codec.prepare(TEXTS[:2])
        with torch.inference_mode():
            a, ia = original(**batch)
            b, ib = codec.model(**batch)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.equal(ia, ib)
    finally:
        sys.path.pop(0)


def test_qwen_codes_and_label_targets():
    from transformers import AutoTokenizer
    from molgen3D.training.pretraining.dataprocessing.text_processing import build_unit
    from molgen3D.training.pretraining.dataprocessing.sequence_packing import (
        SequenceState,
        PendingUnit,
    )

    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    validate_tokenizer(tok)
    stream = "<317>[C]<42>(=[O]<901>)[O]<0>"
    unit = build_unit("O=CO", stream)
    ids = tok.encode(unit, add_special_tokens=False)
    state = SequenceState(128, tok.eos_token_id, tok.eos_token_id, -100)
    state.append_unit(PendingUnit(ids, len(ids) + 1))
    inp, labels = state.finalize()
    for code in [317, 42, 901, 0]:
        index = ids.index(tok.convert_tokens_to_ids(f"<{code}>"))
        assert index > 0 and labels[index - 1] == ids[index]
    assert tok.decode(ids, skip_special_tokens=False) == unit
    assert (
        tok.decode(tok.encode(stream, add_special_tokens=False), skip_special_tokens=True) == stream
    )
    contract = load_contract(TOKENIZER)
    validate_record(
        {
            "canonical_smiles": "O=CO",
            "embedded_smiles": stream,
            "coordtoken_format": FORMAT,
            "coordtoken_sha256": contract["checkpoint_sha256"],
        },
        contract,
    )
    with pytest.raises(ValueError):
        validate_record({"embedded_smiles": stream}, contract)


def test_csv_shards_and_assembly(tmp_path):
    import argparse, csv
    from molgen3D.data_processing.preprocess_coordtoken_csv import run
    from molgen3D.data_processing.assemble_coordtoken import assemble
    from molgen3D.data_processing.coordtoken import SELECTED_SHA256

    if not Path(SELECTED_CHECKPOINT).exists():
        pytest.skip("Shared checkpoint unavailable")
    source = tmp_path / "source.csv"
    with source.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "enriched_text", "dataset"])
        for i, text in enumerate(TEXTS):
            writer.writerow([f"row{i}", text, "fixture"])
    before = source.read_bytes()
    out = tmp_path / "encoded"
    for shard in range(2):
        a = argparse.Namespace(
            input=str(source),
            output=str(out),
            split="train",
            tokenizer=str(TOKENIZER),
            checkpoint=SELECTED_CHECKPOINT,
            expected_sha256=SELECTED_SHA256,
            device="cpu",
            precision="fp32",
            batch_size=2,
            seq_len=128,
            num_shards=2,
            shard_index=shard,
        )
        run(a)
    manifest = assemble(out, 4)
    assert manifest["rows"] == 4 and source.read_bytes() == before
    records = [
        json.loads(s) for p in sorted(out.glob("*.jsonl")) for s in p.read_text().splitlines()
    ]
    assert [x["name"] for x in records] == ["row0", "row1", "row2", "row3"]
    assert all(x["coordtoken_format"] == FORMAT and x["split"] == "train" for x in records)
    from molgen3D.data_processing.coordtoken_contract import validate_training_data

    assert validate_training_data(out, TOKENIZER, 128)["rows"] == 4
    # The actual launch-config generator must preserve batch/GPU and format paths.
    import subprocess, tomli

    base = tmp_path / "qwen-base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    config = tmp_path / "qwen.toml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/configure_coordtoken_qwen.py"),
            "--size",
            "06b",
            "--data",
            str(out),
            "--tokenizer",
            str(TOKENIZER),
            "--base-model",
            str(base),
            "--output",
            str(config),
            "--steps",
            "100",
            "--global-batch",
            "32",
            "--local-batch",
            "2",
            "--gpus",
            "8",
            "--seq-len",
            "128",
        ],
        check=True,
    )
    c = tomli.loads(config.read_text())
    assert (
        c["training"]["global_batch_size"] == 32
        and c["parallelism"]["data_parallel_shard_degree"] == 8
    )
    assert c["molgen_data"]["serialization_mode"] == "pairs" and c["molgen_data"][
        "tokenizer_override"
    ] == str(TOKENIZER)
    assert c["wsds_scheduler"]["checkpoints"] == [100]
    with pytest.raises(FileExistsError):
        run(a)
    with (out / "part-00000.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="changed"):
        validate_training_data(out, TOKENIZER, 128)


def test_small_qwen_training_resume_and_constrained_generation(tmp_path):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from molgen3D.training.pretraining.dataprocessing.text_processing import build_unit

    torch.set_num_threads(4)
    torch.manual_seed(123)
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    stream = "<317>[C]<42>(=[O]<901>)[O]<0>"
    unit = build_unit("O=CO", stream)
    ids = torch.tensor([tok.encode(unit, add_special_tokens=False)])
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        attention_dropout=0.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = Qwen3ForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step(m, o):
        m.train()
        o.zero_grad()
        loss = m(input_ids=ids, labels=ids, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        o.step()

    step(model, opt)
    save = tmp_path / "resume.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, save)
    step(model, opt)
    restored = Qwen3ForCausalLM(cfg)
    ro = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = torch.load(save, weights_only=True)
    restored.load_state_dict(state["model"])
    ro.load_state_dict(state["optimizer"])
    step(restored, ro)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Forced valid suffix checks the generate/tokenizer/parse interface, not model quality.
    prompt = tok.encode("[SMILES]O=CO[/SMILES][CONFORMER]", add_special_tokens=False)
    suffix = tok.encode(stream + "[/CONFORMER]", add_special_tokens=False)
    restored.eval()
    generated = restored.generate(
        torch.tensor([prompt]),
        attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
        max_new_tokens=len(suffix),
        do_sample=False,
        prefix_allowed_tokens_fn=lambda batch, seq: [suffix[len(seq) - len(prompt)]],
    )
    text = tok.decode(generated[0, len(prompt) :], skip_special_tokens=False)
    assert (
        text == stream + "[/CONFORMER]"
        and parse(text.removesuffix("[/CONFORMER]")).molecular_code == 317
    )


def test_rmsd_helper_does_not_move_input():
    from molgen3D.data_processing.smiles_encoder_decoder import coords_rmsd

    a = decode_cartesian_v2(TEXTS[0])
    b = Chem.Mol(a)
    b.GetConformer().SetAtomPosition(1, [1.0, 0.4, 0.2])
    before = a.GetConformer().GetPositions().copy()
    other = b.GetConformer().GetPositions().copy()
    assert np.isfinite(coords_rmsd(a, b))
    np.testing.assert_array_equal(a.GetConformer().GetPositions(), before)
    np.testing.assert_array_equal(b.GetConformer().GetPositions(), other)


def test_exported_lm_contract(tmp_path):
    from transformers import Qwen3Config, AutoTokenizer
    from molgen3D.data_processing.coordtoken_contract import (
        attach_model_contract,
        validate_model_contract,
    )

    cfg = Qwen3Config()
    contract = load_contract(TOKENIZER)
    with pytest.raises(ValueError):
        validate_model_contract(cfg, contract)
    attach_model_contract(cfg, TOKENIZER, tmp_path)
    cfg.save_pretrained(tmp_path)
    restored = Qwen3Config.from_pretrained(tmp_path)
    validate_model_contract(restored, contract)
    assert load_contract(tmp_path) == contract
    validate_tokenizer(AutoTokenizer.from_pretrained(tmp_path, local_files_only=True))
    restored.coordtoken_checkpoint_sha256 = "wrong"
    with pytest.raises(ValueError):
        validate_model_contract(restored, contract)
