"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

# from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import int32
from torch.amp.autocast_mode import autocast

from einops import rearrange, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn): # Wraps a function fn(x,…) so that if x is None, it simply returns None instead of calling fn.
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def round_ste(z): # Quantization (rounding) is non-differentiable. STE (straight-through estimator) lets gradients pass unchanged
    """ round with straight through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    """ floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: List[int], # Number of options for each dimension
        return_indices = True, # So we get discrete tokens back
        num_codebooks = 1, # Number of codebooks for each token, let's keep it 1
        dim: int | None = None, # Optimally codebook_dim * num_codebooks, if no, we'll use projection
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        force_quantization_f32 = True,
        projection_has_bias: bool = True, # Whether to have bias in the projection (if there is one)
        keep_num_codebooks_dim: bool | None = None, # Output one joint index per position (False) or a separate index for each sub‐codebook (True). If you leave keep_num_codebooks_dim=None, it will be treated as True when you have more than one codebook
        preserve_symmetry: bool = False, # Guarantees that for any number of levels(odd or even), the set of quantized values is symmetric about 0 
        noise_dropout = 0.0,
        scale: float | None = None, # Not using it now
        channel_first: bool = False, # No need: it's for forcing image-style axis shuffles
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False) # The last arg. means that it won't be saved into the state_dict(): it still exists as a buffer during runtime, but it’s ignored when saving/loading

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32) # For mixed-radix encoding, which treats each coordinate as a "digit" whose place-value is the product of the sizes of all less-significant digits. (for uniqueness)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.has_projections = has_projections
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections and self.dim else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections and self.dim else nn.Identity()


        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3): # if not preserve_symmetry
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2 # The small eps prevents edge cases where tanh saturates exactly at +-1 and rounding might push out of bounds.
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0) # For even levels, the mapping is assymetrical, that's why we need 0.5 offset to center the quantization
        shift = (offset / half_l).atanh() # Recenters the tanh output so that quantization bins align symmetrically around zero. Notice, that for z=0, the output will be 0
        bounded_z = (z + shift).tanh() * half_l - offset # In [−half_l - offset, +half_l - offset], for example, L=5 -> −2,−1,0,1,2; but for L=4 => −2,−1,0,1
        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width # [-1,1]
    
    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    
    def symmetry_preserving_bound(self, z):
        """
        QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1
        """
        levels_minus_1 = (self._levels - 1)
        scale = 2.0 / levels_minus_1
        act = 2.0 * torch.sigmoid(1.6 * z) - 1.0 # iFSQ
        bracket = (levels_minus_1 * (act + 1) / 2.0) + 0.5
        # bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.0) + 0.5  # 0.5...(L-0.5)
        bracket = floor_ste(bracket) #0...(L-1)
        return scale * bracket - 1.0 #-1...1

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        shape, device, noise_dropout, preserve_symmetry = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        if not self.training or noise_dropout == 0.: # training is a built‑in flag of nn.Module (consequence of model.train())
            return bounded_z
            
        # determine where to add a random offset elementwise, if using noise dropout
        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)
        return bounded_z

    def _scale_and_shift(self, zhat_normalized): # “normalized” codes in [−1,1] -> “level indices” in {0,1,…,L−1}
        if self.preserve_symmetry: # get bracket from symmetry_preserving_bound, which was [0,L-1]
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat): # “level indices” in {0,1,…,L−1} -> “normalized” codes in [−1,1] 
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1. #zhat is bracket from symmetry_preserving_bound
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat):
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32) # For mixed-radix encoding, which treats each coordinate as a "digit" whose place-value is the product of the sizes of all less-significant digits. (for uniqueness)
    
    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1') # So we can broadcast against _basis which is (d,)
        codes_non_centered = (indices // self._basis) % self._levels # Just like in standart basis2 to basis10 conversion we need: 1.number%basis[1], 2.(number//basis[1])%basis[2], 3.(number//basis[2])%basis[3]
        return codes_non_centered
    
    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)
        codes = self._indices_to_codes(indices)
        codes = self.project_out(codes)
        return codes
    
    # All of the 3functions above into one function:
    # def indices_to_codes(self, indices):
    #     assert exists(indices)
    #     indices = rearrange(indices, '... -> ... 1') # So we can broadcast against _basis which is (d,)
    #     codes_non_centered = (indices // self._basis) % self._levels # Just like in standart basis2 to basis10 conversion we need: 1.number%basis[1], 2.(number//basis[1])%basis[2], 3.(number//basis[2])%basis[3]
    #     codes = self._scale_and_shift_inverse(codes_non_centered)
    #     codes = self.project_out(codes)
    #     return codes
    
    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.to(orig_dtype)

        # project out

        out = self.project_out(codes)

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices