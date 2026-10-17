"""Versioned atom-code stream with a single molecular code as its first token.

Example: <317>[C]<42>(=[O]<901>)[O]<0>. No code follows punctuation.
The prefix and atom codes use the same 4096-entry geometry vocabulary.
"""

from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import numpy as np
import torch
from .coordtoken_model import Config, Tokenizer
from .smiles_encoder_decoder_fsq import (
    VOCAB,
    tokenize_enriched,
    format_enriched_from_tokens_and_coords,
)

FORMAT = "atom12_reg1_prefix_v1"
SELECTED_SHA256 = "29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f"
SELECTED_CHECKPOINT = "/mnt/weka/fgeikyan/CoordToken/artifacts/releases/coordtoken-atom12-reg1-mixed-epoch2-20261017/model.pt"
_CODE = re.compile(r"<(0|[1-9][0-9]*)>")
_GRAPH = re.compile(r"\[[^\[\]<>\s]+\]|%[0-9]{2}|[1-9()=#:/.\\-]")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CodeStream:
    graph_tokens: tuple[str, ...]
    atom_positions: tuple[int, ...]
    atom_codes: tuple[int, ...]
    molecular_code: int

    def serialize(self):
        atoms = dict(zip(self.atom_positions, self.atom_codes))
        return f"<{self.molecular_code}>" + "".join(
            t + (f"<{atoms[i]}>" if i in atoms else "") for i, t in enumerate(self.graph_tokens)
        )


def parse(text: str) -> CodeStream:
    def code(pos):
        m = _CODE.match(text, pos)
        if m is None or int(m[1]) >= 4096:
            raise ValueError(f"Expected a code in [0,4095] at position {pos}")
        return int(m[1]), m.end()

    register, pos = code(0)
    tokens = []
    positions = []
    codes = []
    while pos < len(text):
        m = _GRAPH.match(text, pos)
        if m is None or m[0] not in VOCAB:
            raise ValueError(f"Unsupported graph token at position {pos}")
        token = m[0]
        tokens.append(token)
        pos = m.end()
        if token.startswith("["):
            value, pos = code(pos)
            positions.append(len(tokens) - 1)
            codes.append(value)
    if not positions:
        raise ValueError("A molecule must contain an atom")
    return CodeStream(tuple(tokens), tuple(positions), tuple(codes), register)


def graph_smiles(stream: CodeStream) -> str:
    # Same decorative-bracket convention as the original coordinate strings.
    # Ambiguous original radical brackets cannot be recovered from this format.
    from .smiles_encoder_decoder import _ORGANIC_SUBSET

    return "".join(
        t[1:-1] if t.startswith("[") and t[1:-1] in _ORGANIC_SUBSET else t
        for t in stream.graph_tokens
    )


def graph_mol(stream: CodeStream):
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.sanitize = True
    params.removeHs = False
    mol = Chem.MolFromSmiles(graph_smiles(stream), params)
    if mol is None or mol.GetNumAtoms() != len(stream.atom_positions):
        raise ValueError("Invalid graph or atom-order/count mismatch")
    return mol


class CoordTokenCodec:
    format = FORMAT

    def __init__(
        self,
        checkpoint=SELECTED_CHECKPOINT,
        *,
        expected_sha256=SELECTED_SHA256,
        device="cpu",
        precision="fp32",
    ):
        self.path = Path(checkpoint).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not expected_sha256 or sha256(self.path) != expected_sha256:
            raise ValueError("Coordinate checkpoint SHA256 mismatch")
        if precision not in {"fp32", "bf16"}:
            raise ValueError("Unsupported precision")
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        if precision == "bf16" and self.device.type != "cuda":
            raise ValueError("bf16 mode requires CUDA")
        self.precision = precision
        self.checkpoint_sha256 = expected_sha256
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if not {"model", "config", "vocab", "manifest"} <= state.keys():
            raise ValueError(
                "Expected a new CoordToken checkpoint, not a legacy Lightning checkpoint"
            )
        if state["vocab"] != VOCAB:
            raise ValueError("Checkpoint graph vocabulary differs")
        cfg = {k: v for k, v in state["config"].items() if k != "parameters"}
        if (
            cfg.get("bits") != 12
            or cfg.get("registers") != 1
            or cfg.get("mode") != "atom_only"
            or cfg.get("workspace") != 0
        ):
            raise ValueError("Expected atom-only 12-bit plus one transmitted molecular register")
        self.model = Tokenizer(Config(**cfg))
        self.model.load_state_dict(state["model"], strict=True)
        if (
            state["config"].get("parameters", self.model.architecture()["parameters"])
            != self.model.architecture()["parameters"]
        ):
            raise ValueError("Parameter count mismatch")
        self.model.to(self.device).eval()
        self.vocab = {t: i for i, t in enumerate(VOCAB)}

    def _autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )

    def prepare(self, texts):
        if not texts:
            raise ValueError("Empty batch")
        records = []
        for text in texts:
            parts = tokenize_enriched(text)
            tokens = []
            xyz = []
            atoms = []
            for p in parts:
                atom = p["type"] == "atom_with_coords"
                token = str(p["atom_desc"] if atom else p["text"])
                if token not in self.vocab:
                    raise ValueError("Unsupported graph token: " + token)
                tokens.append(token)
                atoms.append(atom)
                xyz.append(p["coords"] if atom else (0, 0, 0))
            atoms = np.asarray(atoms, dtype=bool)
            xyz = np.asarray(xyz, dtype=np.float32)
            if not atoms.any() or not np.isfinite(xyz).all():
                raise ValueError("Missing atoms or nonfinite coordinates")
            xyz[atoms] -= xyz[atoms].mean(0)  # Exact training packer convention.
            if not np.isfinite(xyz).all():
                raise ValueError("Nonfinite centered coordinates")
            records.append((tokens, xyz, atoms))
        width = (max(len(x[0]) for x in records) + 7) // 8 * 8
        ids = np.full((len(texts), width), -1, np.int64)
        coords = np.zeros((len(texts), width, 3), np.float32)
        mask = np.zeros((len(texts), width), bool)
        for i, (t, c, a) in enumerate(records):
            ids[i, : len(t)] = [self.vocab[x] for x in t]
            coords[i, : len(t)] = c
            mask[i, : len(t)] = a
        return {
            k: torch.from_numpy(v).to(self.device)
            for k, v in [("tokens", ids), ("coords", coords), ("atoms", mask)]
        }, records

    @torch.inference_mode()
    def encode_batch(self, texts):
        if not texts:
            return []
        batch, records = self.prepare(texts)
        from torch.nn.attention import sdpa_kernel, SDPBackend

        backend = (
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            if self.device.type == "cuda"
            else SDPBackend.MATH
        )
        with self._autocast(), sdpa_kernel(backend):
            _, indices = self.model.encode(**batch)
        indices = indices.cpu().numpy()
        out = []
        for i, (tokens, _, atoms) in enumerate(records):
            positions = tuple(np.flatnonzero(atoms).tolist())
            stream = CodeStream(
                tuple(tokens),
                positions,
                tuple(int(indices[i, p]) for p in positions),
                int(indices[i, -1]),
            )
            text = stream.serialize()
            parse(text)
            out.append(text)
        return out

    def encode_text(self, text):
        return self.encode_batch([text])[0]

    @torch.inference_mode()
    def decode_batch(self, texts):
        if not texts:
            return []
        streams = [parse(t) for t in texts]
        width = (max(len(s.graph_tokens) for s in streams) + 7) // 8 * 8
        ids = np.full((len(streams), width), -1, np.int64)
        codes = np.full((len(streams), width + 1), -1, np.int64)
        atoms = np.zeros((len(streams), width), bool)
        for i, s in enumerate(streams):
            ids[i, : len(s.graph_tokens)] = [self.vocab[t] for t in s.graph_tokens]
            atoms[i, list(s.atom_positions)] = True
            codes[i, list(s.atom_positions)] = s.atom_codes
            codes[i, width] = s.molecular_code
        ids, codes, atoms = [torch.from_numpy(x).to(self.device) for x in [ids, codes, atoms]]
        with self._autocast():
            xyz = self.model.decode(ids, codes, atoms)
        if not bool(torch.isfinite(xyz[atoms]).all()):
            raise ValueError("Nonfinite decoded coordinates")
        xyz = xyz.float().cpu().numpy()
        return [(s, xyz[i, : len(s.graph_tokens)]) for i, s in enumerate(streams)]

    def decode_text(self, text, precision=8):
        s, xyz = self.decode_batch([text])[0]
        return format_enriched_from_tokens_and_coords(list(s.graph_tokens), xyz, precision)

    def decode_mol(self, text):
        from rdkit import Chem

        s, xyz = self.decode_batch([text])[0]
        mol = graph_mol(s)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom, pos in enumerate(s.atom_positions):
            conf.SetAtomPosition(atom, xyz[pos].astype(float).tolist())
        mol.AddConformer(conf, assignId=True)
        return mol


def encode_mol(mol, codec, precision=4):
    from .smiles_encoder_decoder import encode_cartesian_v2

    text, smiles = encode_cartesian_v2(mol, precision=precision)
    return codec.encode_text(text), smiles
