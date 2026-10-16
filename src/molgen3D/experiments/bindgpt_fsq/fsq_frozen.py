"""The frozen FSQ coordinate tokenizer.

Wraps the project's ``MolFSQModel`` -- the exact module that encoded the
project's FSQ corpora -- pinned to one immutable checkpoint, held in eval mode
with gradients disabled, and fingerprinted so a training run can prove no FSQ
parameter moved.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
    MolFSQModel,
    format_enriched_from_tokens_and_coords,
)
from molgen3D.experiments.bindgpt_fsq import config

#: Ceiling on graph tokens per molecule imposed by the FSQ model. GEOM-DRUGS
#: does not approach it (observed maximum 150 over a 200k-conformer sample), but
#: any exceedance is recorded as an exclusion rather than silently dropped.
MAX_GRAPH_TOKENS = 218

#: Parameter count of the pinned checkpoint, asserted at load time.
EXPECTED_PARAMETERS = 202_011_535


@dataclass(frozen=True)
class FSQFingerprint:
    """Provenance and integrity record for the frozen tokenizer."""

    checkpoint: str
    checkpoint_sha256: str
    parameter_sha256: str
    parameter_count: int
    d_model: int
    n_layers: int
    levels: Sequence[int]
    codebook_size: int
    epoch: Optional[int]
    global_step: Optional[int]

    def as_dict(self) -> Dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameter_sha256": self.parameter_sha256,
            "parameter_count": self.parameter_count,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "levels": list(self.levels),
            "codebook_size": self.codebook_size,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }


def parameter_checksum(model: torch.nn.Module) -> str:
    """Return a SHA-256 over every parameter, in sorted-name order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach().to(dtype=torch.float32, device="cpu").numpy().tobytes()
        )
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class FrozenFSQTokenizer:
    """Deterministic encode/decode against one pinned FSQ checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = "auto",
        *,
        verify_checkpoint_hash: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint or config.FSQ_CHECKPOINT)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"FSQ checkpoint not found: {self.checkpoint}")

        self.model = MolFSQModel.load_from_checkpoint(
            self.checkpoint,
            map_location="cpu",
            device=device,
            max_tokens=MAX_GRAPH_TOKENS,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        count = sum(p.numel() for p in self.model.parameters())
        if count != EXPECTED_PARAMETERS:
            raise ValueError(
                f"pinned FSQ checkpoint has {count:,} parameters, expected "
                f"{EXPECTED_PARAMETERS:,}"
            )

        self._raw = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._checkpoint_sha256 = (
            sha256_file(self.checkpoint) if verify_checkpoint_hash else "not-computed"
        )
        self._parameter_sha256 = parameter_checksum(self.model)

    # ------------------------------------------------------------------ #

    @property
    def codebook_size(self) -> int:
        return int(np.prod(self.model.levels))

    def fingerprint(self) -> FSQFingerprint:
        return FSQFingerprint(
            checkpoint=str(self.checkpoint),
            checkpoint_sha256=self._checkpoint_sha256,
            parameter_sha256=self._parameter_sha256,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            d_model=self.model.d_model,
            n_layers=self.model.n_layers,
            levels=tuple(self.model.levels),
            codebook_size=self.codebook_size,
            epoch=self._raw.get("epoch") if isinstance(self._raw, dict) else None,
            global_step=(
                self._raw.get("global_step") if isinstance(self._raw, dict) else None
            ),
        )

    def assert_unchanged(self) -> None:
        """Raise if any FSQ parameter has moved since load."""
        current = parameter_checksum(self.model)
        if current != self._parameter_sha256:
            raise RuntimeError(
                "frozen FSQ parameters changed during the run: "
                f"{self._parameter_sha256} -> {current}"
            )

    def assert_no_gradients(self) -> None:
        """Raise if any FSQ parameter is collecting gradient."""
        offenders = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad or parameter.grad is not None
        ]
        if offenders:
            raise RuntimeError(f"FSQ parameters are not frozen: {offenders[:8]}")

    # ------------------------------------------------------------------ #

    def encode(self, enriched_texts: Sequence[str]) -> List[np.ndarray]:
        """Return one FSQ code array per input enriched string.

        The model centres each conformer on its own heavy-atom centroid
        internally, which is the single deterministic centring rule used by both
        arms.
        """
        if not enriched_texts:
            return []
        with torch.inference_mode():
            representations = self.model.encode_representations(list(enriched_texts))
        return [representation.codes for representation in representations]

    def decode(
        self, graph_tokens: Sequence[str], codes: Sequence[int], precision: int = 4
    ) -> np.ndarray:
        """Decode codes back to one coordinate triplet per atom-bearing token."""
        if len(graph_tokens) != len(codes):
            raise ValueError(
                f"{len(graph_tokens)} graph tokens but {len(codes)} FSQ codes"
            )
        with torch.inference_mode():
            enriched = self.model.decode_tokens_with_codes(
                list(graph_tokens), np.asarray(codes, dtype=np.int64), precision=precision
            )
        from molgen3D.experiments.bindgpt_fsq.sequences import (
            atom_coordinates,
            parse_enriched,
        )

        return atom_coordinates(parse_enriched(enriched))

    def decode_batch(
        self,
        graph_tokens: Sequence[str],
        code_batch: np.ndarray,
        precision: int = 4,
    ) -> List[np.ndarray]:
        """Decode a batch of code arrays that share one graph token sequence."""
        from molgen3D.experiments.bindgpt_fsq.sequences import (
            atom_coordinates,
            parse_enriched,
        )

        with torch.inference_mode():
            enriched_batch = self.model.decode_tokens_with_code_batch(
                list(graph_tokens), np.asarray(code_batch, dtype=np.int64), precision
            )
        return [atom_coordinates(parse_enriched(text)) for text in enriched_batch]

    def round_trip(self, enriched_text: str, precision: int = 4) -> np.ndarray:
        """Encode then decode one conformer, returning reconstructed coordinates."""
        from molgen3D.experiments.bindgpt_fsq.sequences import parse_enriched

        tokens = parse_enriched(enriched_text)
        codes = self.encode([enriched_text])[0]
        return self.decode([token.text for token in tokens], codes, precision=precision)


__all__ = [
    "FrozenFSQTokenizer",
    "FSQFingerprint",
    "MAX_GRAPH_TOKENS",
    "format_enriched_from_tokens_and_coords",
    "parameter_checksum",
    "sha256_file",
]
