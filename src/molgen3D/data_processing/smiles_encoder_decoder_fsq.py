"""FSQ coordinate-token serialization helpers.

The trained FSQ coordinate model from the old baseline consumes the normal
cartesian enriched string, centers atom coordinates internally, and emits one
integer FSQ code for every SMILES token:

    [C]<123>(<45>[O]<678>)<90>

This module keeps that behavior local to this repository so the big-data
preprocessing script can use ``--embedding_type fsq``.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from molgen3D.data_processing.fsq import FSQ


VOCAB = [
    "#", "%10", "%11", "%12", "%13", "%14", "%15", "%16", "%17", "%18", "%19",
    "%20", "(", ")", "-", ".", "/", "1", "2", "3", "4", "5", "6", "7", "8",
    "9", "=", "[Ag]", "[Al]", "[AsH2]", "[AsH3]", "[As]", "[B-]", "[B@@-]",
    "[BH-]", "[BH2-]", "[BH]", "[B]", "[Ba]", "[Bi]", "[Br+]", "[Br]", "[C+]",
    "[C-]", "[C@@H]", "[C@@]", "[C@H]", "[C@]", "[CH+]", "[CH-]", "[CH2+]",
    "[CH2-]", "[CH2]", "[CH3-]", "[CH3]", "[CH]", "[C]", "[CaH]", "[Ca]",
    "[Cl-]", "[Cl]", "[Cs]", "[Cu@SP]", "[F-]", "[F]", "[Ga]", "[H+]", "[H]",
    "[I+2]", "[I+]", "[I-]", "[IH]", "[I]", "[K]", "[Li]", "[MgH]", "[Mg]",
    "[Mn@SP]", "[Mn]", "[N+]", "[N-2]", "[N-]", "[N@+]", "[N@@+]",
    "[N@@H+]", "[N@H+]", "[NH+]", "[NH-]", "[NH2+]", "[NH3+]", "[NH4+]",
    "[NH]", "[N]", "[Na]", "[O+]", "[O-]", "[OH+]", "[OH2+]", "[OH]",
    "[O]", "[P+]", "[P-2]", "[P-]", "[P@+]", "[P@@+]", "[P@@H+]",
    "[P@@H2+]", "[P@@H2]", "[P@@H3+]", "[P@@H]", "[P@@]", "[P@H+]",
    "[P@H2+]", "[P@H2]", "[P@H3+]", "[P@H]", "[P@OH]", "[P@TB]", "[P@]",
    "[PH+]", "[PH2+]", "[PH2]", "[PH]", "[P]", "[Pb]", "[Rb]", "[S+2]",
    "[S+]", "[S-]", "[S@+]", "[S@@+]", "[S@@H]", "[S@@]", "[S@H]",
    "[S@OH]", "[S@SP]", "[S@TB]", "[S@]", "[SH+3]", "[SH+]", "[SH2+]",
    "[SH2]", "[SH3+2]", "[SH3]", "[SH4]", "[SH]", "[S]", "[Se-]", "[SeH]",
    "[Se]", "[Si+2]", "[Si+3]", "[Si+]", "[Si-]", "[Si@@H]", "[Si@@]",
    "[Si@H]", "[Si@]", "[SiH-]", "[SiH2]", "[SiH3]", "[SiH4]", "[SiH]",
    "[Si]", "[Sr]", "[Zn]", "[c+]", "[c-]", "[cH+]", "[cH-]", "[c]",
    "[n+]", "[n-]", "[nH+]", "[nH]", "[n]", "[o+]", "[oH+]", "[o]",
    "[p+]", "[p-]", "[pH]", "[p]", "[s+2]", "[s+]", "[sH+]", "[s]",
    "[se]", "[si+]", "[si-]", "[si]", "\\",
]

TOKEN_TO_IDX = {tok: i for i, tok in enumerate(VOCAB)}
V = len(VOCAB)
IS_ATOM_NP = np.array([tok.startswith("[") for tok in VOCAB], dtype=np.float32)

_ENRICHED_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\])<([^>]+)>|(%\d{2,})|(=|#|:|\/|\\|-)|(\()|(\))|(\d)|(\.)"
)
_FSQ_PAIR_RE = re.compile(r"(.*?)<(\d+)>")

DEFAULT_FSQ_CKPT_CANDIDATES = (
    "/mnt/weka/fgeikyan/fsq/new_checkpoints/"
    "full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt",
    "/auto/home/filya/fsq_remote/checkpoints/fsq/last-v2.ckpt",
)

_FSQ_ENCODER: Optional["MolFSQModel"] = None
_FSQ_ENCODER_CONFIG: Optional[dict[str, object]] = None


def _truncate(x: float, precision: int = 4) -> str:
    if precision < 0:
        raise ValueError("precision must be non-negative")
    value = float(x)
    if precision == 0:
        return str(int(math.trunc(value)))

    factor = 10 ** precision
    truncated = math.trunc(value * factor) / factor
    if abs(truncated) < 10 ** (-precision):
        truncated = 0.0
    return f"{truncated:.{precision}f}"


def _resolve_existing_path(*candidates: str | os.PathLike[str] | None) -> Path:
    fallback: Optional[Path] = None
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if fallback is None:
            fallback = path
        try:
            if path.exists():
                return path
        except PermissionError:
            continue
    if fallback is None:
        raise FileNotFoundError("No FSQ checkpoint path candidates were provided.")
    return fallback


def _resolve_fsq_ckpt_path(ckpt_path: str | os.PathLike[str] | None = None) -> Path:
    return _resolve_existing_path(
        ckpt_path,
        os.environ.get("FSQ_CKPT_PATH"),
        os.environ.get("VQ_CKPT_PATH"),
        *DEFAULT_FSQ_CKPT_CANDIDATES,
    )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def tokenize_enriched(enriched_smiles: str) -> list[dict[str, object]]:
    tokens: list[dict[str, object]] = []
    pos = 0
    for match in _ENRICHED_TOKEN_PATTERN.finditer(enriched_smiles):
        if match.start() != pos:
            raise ValueError(
                f"Unrecognized enriched fragment: "
                f"{enriched_smiles[pos:match.start()]} in {enriched_smiles}"
            )

        atom_desc = match.group(1)
        if atom_desc is not None:
            coord_str = match.group(2)
            parts = [p.strip() for p in coord_str.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Bad coord triplet: {coord_str}")
            tokens.append(
                {
                    "type": "atom_with_coords",
                    "atom_desc": atom_desc,
                    "coords": tuple(float(p) for p in parts),
                }
            )
        else:
            text = (
                match.group(3)
                or match.group(4)
                or match.group(5)
                or match.group(6)
                or match.group(7)
                or match.group(8)
            )
            tokens.append({"type": "nonatom", "text": text})
        pos = match.end()

    if pos != len(enriched_smiles):
        raise ValueError(
            f"Unparsed trailing enriched fragment: {enriched_smiles[pos:]} in {enriched_smiles}"
        )
    return tokens


def tokenize_and_encode(enriched_smiles: str) -> tuple[np.ndarray, np.ndarray]:
    tokens = tokenize_enriched(enriched_smiles)
    feats = np.zeros((len(tokens), V + 3), dtype=np.float32)
    tok_ids = np.empty(len(tokens), dtype=np.int64)

    for i, token in enumerate(tokens):
        if token["type"] == "atom_with_coords":
            text = str(token["atom_desc"])
            coords = token["coords"]
            idx = TOKEN_TO_IDX[text]
            tok_ids[i] = idx
            feats[i, idx] = 1.0
            feats[i, V : V + 3] = np.asarray(coords, dtype=np.float32)
        else:
            text = str(token["text"])
            idx = TOKEN_TO_IDX[text]
            tok_ids[i] = idx
            feats[i, idx] = 1.0

    return feats, tok_ids


def build_fsq_string(enriched_text: str, codes) -> str:
    tokens = tokenize_enriched(enriched_text)
    if len(tokens) != len(codes):
        raise ValueError(f"Mismatch: {len(tokens)} tokens != {len(codes)} FSQ codes")

    new_parts: list[str] = []
    for token, code in zip(tokens, codes):
        code_val = int(code)
        if token["type"] == "atom_with_coords":
            new_parts.append(f"{token['atom_desc']}<{code_val}>")
        else:
            new_parts.append(f"{token['text']}<{code_val}>")
    return "".join(new_parts)


def parse_fsq_text(fsq_text: str) -> tuple[list[str], np.ndarray]:
    pairs = _FSQ_PAIR_RE.findall(fsq_text)
    if not pairs:
        raise ValueError("FSQ text parsing failed: no token<code> pairs found.")
    tokens = [token for token, _ in pairs]
    codes = np.array([int(code) for _, code in pairs], dtype=np.int64)
    return tokens, codes


def tokens_to_vocab_onehot(tokens: list[str]) -> np.ndarray:
    x = np.zeros((len(tokens), V), dtype=np.float32)
    for i, token in enumerate(tokens):
        x[i, TOKEN_TO_IDX[token]] = 1.0
    return x


def format_enriched_from_tokens_and_coords(
    tokens: list[str],
    coords: np.ndarray,
    precision: int = 4,
) -> str:
    out_parts: list[str] = []
    for token, xyz in zip(tokens, coords):
        if token.startswith("["):
            x = _truncate(float(xyz[0]), precision)
            y = _truncate(float(xyz[1]), precision)
            z = _truncate(float(xyz[2]), precision)
            out_parts.append(f"{token}<{x},{y},{z}>")
        else:
            out_parts.append(token)
    return "".join(out_parts)


class MolFSQModel(nn.Module):
    """Checkpoint-compatible FSQ coordinate model from the old CoordToken run."""

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 8,
        levels: Optional[list[int]] = None,
        max_tokens: Optional[int] = 218,
    ) -> None:
        super().__init__()
        if levels is None:
            levels = [2] * 12

        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.levels = list(levels)
        self.max_tokens = max_tokens

        self.tok_emb = nn.Linear(V + 3, self.d_model)
        self.register_buffer("pos_emb_cache", torch.empty(1, 0, self.d_model), persistent=False)

        def block() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.d_model // 64,
                dim_feedforward=self.d_model * 4,
                batch_first=True,
                norm_first=True,
                bias=False,
            )
            return nn.TransformerEncoder(layer, self.n_layers)

        self.enc = block()
        self.dec = block()
        self.pre_q = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 8),
            nn.ReLU(),
            nn.Linear(self.d_model // 8, len(self.levels)),
            nn.Tanh(),
        )
        self.quant = FSQ(
            self.levels,
            return_indices=True,
            preserve_symmetry=True,
            keep_num_codebooks_dim=False,
        )
        self.post_q = nn.Sequential(
            nn.Linear(V + len(self.levels), 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model),
        )
        self.out = nn.Linear(self.d_model, 3)
        self.register_buffer("util", torch.zeros(math.prod(self.levels), dtype=torch.long))

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
        device: str | torch.device | None = None,
        max_tokens: Optional[int] = 218,
    ) -> "MolFSQModel":
        try:
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=map_location)

        hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
        model = cls(
            d_model=int(hparams.get("d_model", 1024)),
            n_layers=int(hparams.get("n_layers", 8)),
            levels=list(hparams.get("levels", [2] * 12)),
            max_tokens=max_tokens,
        )

        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unexpected FSQ checkpoint format: {type(checkpoint)!r}")
        if all(str(key).startswith("model.") for key in state_dict):
            state_dict = {str(key).removeprefix("model."): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.requires_grad_(False)
        model.to(_resolve_device(device))
        return model

    def _get_pos_emb(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pe = self.pos_emb_cache
        if pe.size(1) >= n and pe.device == device and pe.dtype == dtype:
            return pe[:, :n]

        p = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        v = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        new_pe = torch.zeros(1, n, self.d_model, device=device, dtype=torch.float32)
        new_pe[0, :, 0::2] = torch.sin(p * v)
        new_pe[0, :, 1::2] = torch.cos(p * v)
        self.pos_emb_cache = new_pe.to(dtype=dtype)
        return self.pos_emb_cache

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _featurize_texts(self, enriched_smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encoded: list[tuple[np.ndarray, np.ndarray]] = []
        lengths = np.empty(len(enriched_smiles_list), dtype=np.int64)
        max_len = 0

        for i, enriched_smiles in enumerate(enriched_smiles_list):
            feats_np, tok_ids = tokenize_and_encode(enriched_smiles)
            n_tokens = len(tok_ids)
            if n_tokens == 0:
                raise ValueError("FSQ input has no tokens.")
            if self.max_tokens is not None and n_tokens > self.max_tokens:
                raise ValueError(
                    f"FSQ input has {n_tokens} tokens, exceeding max_tokens={self.max_tokens}"
                )

            atom_mask = IS_ATOM_NP[tok_ids].astype(bool)
            if atom_mask.any():
                feats_np[atom_mask, V:] -= feats_np[atom_mask, V:].mean(axis=0)

            encoded.append((feats_np, tok_ids))
            lengths[i] = n_tokens
            max_len = max(max_len, n_tokens)

        features = np.zeros((len(encoded), max_len, V + 3), dtype=np.float32)
        for i, (feats_np, _tok_ids) in enumerate(encoded):
            features[i, : feats_np.shape[0]] = feats_np
        return features, lengths

    def encode_texts(self, enriched_smiles_list: list[str]) -> list[str]:
        if not enriched_smiles_list:
            return []

        features, lengths = self._featurize_texts(enriched_smiles_list)
        x = torch.from_numpy(features).to(self.device, dtype=self.dtype)
        max_len = x.size(1)
        pad_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= torch.as_tensor(
            lengths, device=x.device
        ).unsqueeze(1)

        with torch.inference_mode():
            emb = self.tok_emb(x)
            pos_emb = self._get_pos_emb(x.size(1), x.device, emb.dtype)
            z = self.enc(emb + pos_emb, src_key_padding_mask=pad_mask)
            _, indices = self.quant(self.pre_q(z))

        indices_np = indices.detach().cpu().numpy()
        return [
            build_fsq_string(enriched_smiles, indices_np[i, : int(length)])
            for i, (enriched_smiles, length) in enumerate(zip(enriched_smiles_list, lengths))
        ]

    def encode_text(self, enriched_smiles: str) -> str:
        return self.encode_texts([enriched_smiles])[0]

    def decode_text(self, fsq_text: str, precision: int = 4) -> str:
        tokens, codes_np = parse_fsq_text(fsq_text)
        indices = torch.from_numpy(codes_np).unsqueeze(0).to(self.device, dtype=torch.int32)
        xv = torch.from_numpy(tokens_to_vocab_onehot(tokens)).unsqueeze(0).to(
            self.device, dtype=self.dtype
        )

        with torch.inference_mode():
            quant = self.quant.indices_to_codes(indices).to(dtype=self.dtype, device=self.device)
            decoder_input = self.post_q(torch.cat([xv, quant], dim=-1))
            pos_emb = self._get_pos_emb(len(tokens), decoder_input.device, decoder_input.dtype)
            recon = self.out(self.dec(decoder_input + pos_emb))[0].detach().cpu().numpy()
        return format_enriched_from_tokens_and_coords(tokens, recon, precision=precision)


def configure_fsq_encoder(
    ckpt_path: str | os.PathLike[str] | None = None,
    device: str | torch.device | None = "auto",
    max_tokens: Optional[int] = 218,
) -> MolFSQModel:
    global _FSQ_ENCODER, _FSQ_ENCODER_CONFIG

    resolved_ckpt = _resolve_fsq_ckpt_path(ckpt_path)
    resolved_device = _resolve_device(device)
    config = {
        "ckpt_path": str(resolved_ckpt),
        "device": str(resolved_device),
        "max_tokens": max_tokens,
    }
    if _FSQ_ENCODER is not None and _FSQ_ENCODER_CONFIG == config:
        return _FSQ_ENCODER

    _FSQ_ENCODER = MolFSQModel.load_from_checkpoint(
        resolved_ckpt,
        map_location="cpu",
        device=resolved_device,
        max_tokens=max_tokens,
    )
    _FSQ_ENCODER_CONFIG = config
    return _FSQ_ENCODER


def get_fsq_encoder() -> MolFSQModel:
    if _FSQ_ENCODER is None:
        if _FSQ_ENCODER_CONFIG is None:
            raise RuntimeError(
                "FSQ encoder is not configured. Call configure_fsq_encoder() before encoding."
            )
        return configure_fsq_encoder(**_FSQ_ENCODER_CONFIG)
    return _FSQ_ENCODER


def encode_cartesian_fsq(mol, precision: int = 4) -> tuple[str, str]:
    from molgen3D.data_processing.smiles_encoder_decoder import encode_cartesian_v2

    enriched, iso_smiles = encode_cartesian_v2(mol, precision=precision)
    fsq_text = get_fsq_encoder().encode_text(enriched)
    return fsq_text, iso_smiles
