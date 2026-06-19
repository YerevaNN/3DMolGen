"""Finite scalar quantization used by the trained coordinate tokenizer.

This is the small subset of the old CoordToken FSQ implementation needed for
offline encoding/decoding of molecule coordinate strings.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _round_ste(z: torch.Tensor) -> torch.Tensor:
    zhat = z.round()
    return z + (zhat - z).detach()


def _floor_ste(z: torch.Tensor) -> torch.Tensor:
    zhat = z.floor()
    return z + (zhat - z).detach()


class FSQ(nn.Module):
    def __init__(
        self,
        levels: List[int],
        return_indices: bool = True,
        num_codebooks: int = 1,
        dim: int | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        force_quantization_f32: bool = True,
        projection_has_bias: bool = True,
        keep_num_codebooks_dim: bool | None = None,
        preserve_symmetry: bool = False,
        noise_dropout: float = 0.0,
        scale: float | None = None,
        channel_first: bool = False,
    ) -> None:
        super().__init__()

        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        self.register_buffer(
            "_basis",
            torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0),
            persistent=False,
        )

        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout
        self.codebook_dim = len(levels)
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = self.codebook_dim * num_codebooks
        self.keep_num_codebooks_dim = (
            num_codebooks > 1 if keep_num_codebooks_dim is None else keep_num_codebooks_dim
        )
        if num_codebooks > 1 and not self.keep_num_codebooks_dim:
            raise ValueError("num_codebooks > 1 requires keep_num_codebooks_dim=True")

        self.dim = dim or self.effective_codebook_dim
        self.channel_first = channel_first
        has_projections = self.dim != self.effective_codebook_dim
        self.has_projections = has_projections
        self.project_in = (
            nn.Linear(self.dim, self.effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = int(self._levels.prod().item())
            self.register_buffer(
                "implicit_codebook",
                self._indices_to_codes(torch.arange(self.codebook_size)),
                persistent=False,
            )

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        levels = self._levels.to(device=z.device, dtype=z.dtype)
        half_l = (levels - 1) * (1 + eps) / 2
        offset = torch.where(levels.remainder(2) == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = levels // 2
        return _round_ste(bounded_z) / half_width

    def symmetry_preserving_bound(self, z: torch.Tensor) -> torch.Tensor:
        levels = self._levels.to(device=z.device, dtype=z.dtype)
        levels_minus_1 = levels - 1
        scale = 2.0 / levels_minus_1
        act = 2.0 * torch.sigmoid(1.6 * z) - 1.0
        bracket = (levels_minus_1 * (act + 1) / 2.0) + 0.5
        bracket = _floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        bound_fn = self.symmetry_preserving_bound if self.preserve_symmetry else self.bound
        bounded_z = bound_fn(z)
        if not self.training or self.noise_dropout == 0.0:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, self.noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        return torch.where(offset_mask, bounded_z + offset, bounded_z)

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        levels = self._levels.to(device=zhat_normalized.device, dtype=zhat_normalized.dtype)
        if self.preserve_symmetry:
            return (zhat_normalized + 1.0) / (2.0 / (levels - 1))
        half_width = levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        levels = self._levels.to(device=zhat.device, dtype=zhat.dtype)
        if self.preserve_symmetry:
            return zhat * (2.0 / (levels - 1)) - 1.0
        half_width = levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        if zhat.shape[-1] != self.codebook_dim:
            raise ValueError(
                f"expected codebook dim {self.codebook_dim}, got {zhat.shape[-1]}"
            )
        zhat = self._scale_and_shift(zhat)
        basis = self._basis.to(device=zhat.device, dtype=zhat.dtype)
        return (zhat * basis).sum(dim=-1).to(torch.int32)

    def indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.unsqueeze(-1)
        basis = self._basis.to(device=indices.device)
        levels = self._levels.to(device=indices.device)
        return (indices // basis) % levels

    def _indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        level_indices = self.indices_to_level_indices(indices)
        return self._scale_and_shift_inverse(level_indices.to(torch.float32))

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self._indices_to_codes(indices)
        return self.project_out(codes)

    def forward(self, z: torch.Tensor):
        if z.shape[-1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {z.shape[-1]}")

        z = self.project_in(z)
        z = z.reshape(*z.shape[:-1], self.num_codebooks, self.codebook_dim)

        force_f32 = self.force_quantization_f32
        orig_dtype = z.dtype
        if force_f32 and orig_dtype not in self.allowed_dtypes:
            z = z.float()

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = codes.reshape(*codes.shape[:-2], self.effective_codebook_dim)
        out = self.project_out(codes).to(orig_dtype)

        if not self.return_indices:
            return out
        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)
        return out, indices
