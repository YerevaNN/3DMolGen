"""Atom-only FSQ with optional transmitted molecular registers.

The decoder receives graph identities and discrete codes only. Non-atom geometry
vectors are zero placeholders, not code ID zero. No orientation side channel.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from quantizer import FSQ


@dataclass
class Config:
    vocab_size: int = 192
    width: int = 1024
    layers: int = 8
    workspace: int = 0
    bits: int = 12
    dropout: float = 0.1
    mode: str = 'atom_only'
    registers: int = 0


class Tokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        if c.mode not in {'atom_only', 'all_tokens'} or c.workspace:
            raise ValueError('This study uses atom codes, optional transmitted registers, and no workspace')
        if c.bits not in {12, 13} or c.registers not in {0, 1}:
            raise ValueError('Unregistered bottleneck')
        self.tok_emb = nn.Linear(c.vocab_size + 3, c.width)
        def block():
            layer = nn.TransformerEncoderLayer(c.width, c.width // 64, c.width * 4,
                dropout=c.dropout, batch_first=True, norm_first=True, bias=False)
            return nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.enc, self.dec = block(), block()
        self.pre_q = nn.Sequential(nn.Linear(c.width, c.width // 8), nn.ReLU(),
                                   nn.Linear(c.width // 8, c.bits), nn.Tanh())
        self.quant = FSQ([2] * c.bits, return_indices=True, preserve_symmetry=True,
                         keep_num_codebooks_dim=False)
        self.post_q = nn.Sequential(nn.Linear(c.vocab_size + c.bits, 256), nn.ReLU(),
                                    nn.Linear(256, c.width))
        self.out = nn.Linear(c.width, 3)
        # Initialize these last so shared 12-bit weights match the no-register arm.
        if c.registers:
            self.encoder_registers = nn.Parameter(torch.randn(c.registers, c.width) * .02)
            self.decoder_registers = nn.Parameter(torch.randn(c.registers, c.width) * .02)
        self.register_buffer('pos_emb_cache', torch.empty(1, 0, c.width), persistent=False)

    def positions(self, n, device, dtype):
        if self.pos_emb_cache.size(1) < n or self.pos_emb_cache.dtype != dtype or self.pos_emb_cache.device != device:
            pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
            freq = torch.exp(torch.arange(0, self.config.width, 2, device=device,
                              dtype=torch.float32) * (-math.log(10000.) / self.config.width))
            pe = torch.empty(1, n, self.config.width, device=device)
            pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
            self.pos_emb_cache = pe.to(dtype)
        return self.pos_emb_cache[:, :n]

    def identities(self, tokens):
        return F.one_hot(tokens.clamp_min(0), self.config.vocab_size).float() * tokens.ge(0).unsqueeze(-1)

    def code_mask(self, tokens, atoms):
        valid = tokens.ge(0)
        mask = valid if self.config.mode == 'all_tokens' else atoms & valid
        return F.pad(mask, (0, self.config.registers), value=True)

    def encode(self, tokens, coords, atoms):
        identity = self.identities(tokens)
        h = self.tok_emb(torch.cat([identity, coords * atoms.unsqueeze(-1)], -1))
        h = h + self.positions(tokens.size(1), h.device, h.dtype)
        pad = tokens.lt(0)
        if self.config.registers:
            regs = self.encoder_registers.to(h.dtype)[None].expand(tokens.size(0), -1, -1)
            h = torch.cat([h, regs], 1)
            pad = F.pad(pad, (0, self.config.registers), value=False)
        z = self.enc(h, src_key_padding_mask=pad)
        quant, indices = self.quant(self.pre_q(z))
        mask = self.code_mask(tokens, atoms)
        return quant * mask.unsqueeze(-1), indices.masked_fill(~mask, -1)

    def decode_quant(self, tokens, quant, atoms):
        n = tokens.size(1)
        quant = quant * self.code_mask(tokens, atoms).unsqueeze(-1)
        identity = self.identities(tokens)
        h = self.post_q(torch.cat([identity.to(quant.dtype), quant[:, :n]], -1))
        h = h + self.positions(n, h.device, h.dtype)
        pad = tokens.lt(0)
        if self.config.registers:
            graph_zero = identity.new_zeros(tokens.size(0), self.config.registers, self.config.vocab_size)
            regs = self.post_q(torch.cat([graph_zero.to(quant.dtype), quant[:, n:]], -1))
            regs = regs + self.decoder_registers.to(regs.dtype)[None]
            h = torch.cat([h, regs], 1)
            pad = F.pad(pad, (0, self.config.registers), value=False)
        h = self.dec(h, src_key_padding_mask=pad)[:, :n]
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.out(h.float())

    def decode(self, tokens, indices, atoms):
        # Discarded indices are -1. Clamp for conversion, then mask their vectors
        # to zero in decode_quant: ID zero is itself a valid, nonzero FSQ vector.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backend = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if tokens.is_cuda else SDPBackend.MATH
        with sdpa_kernel(backend):
            return self.decode_quant(tokens, self.quant.indices_to_codes(indices.clamp_min(0)), atoms)

    def forward(self, tokens, coords, atoms):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backend = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if tokens.is_cuda else SDPBackend.MATH
        with sdpa_kernel(backend):
            quant, indices = self.encode(tokens, coords, atoms)
            return self.decode_quant(tokens, quant, atoms), indices

    def architecture(self):
        return {**asdict(self.config), 'parameters': sum(p.numel() for p in self.parameters())}
