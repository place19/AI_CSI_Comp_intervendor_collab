"""Positional encoding block for (B, S, F) transformer inputs.

Three modes, selected by `mode`:
    fixed_sincos:      classic sin/cos table from "Attention is All You Need",
                       stored as a non-trainable buffer.
    learnable_random:  nn.Parameter init from N(0, init_std).
    learnable_sincos:  nn.Parameter initialised with the sin/cos table, then
                       learned end-to-end. Hybrid of the above two.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from ...registry import register
from .base import Block


_MODES = ("fixed_sincos", "learnable_random", "learnable_sincos")


@register("block", "positional_encoding")
class PositionalEncodingBlock(Block):
    """Adds a (seq_len, dim) positional table to (B, S, F) inputs.

    `seq_len` and `dim` are required kwargs even though they're derivable from
    `in_shape` — this is intentional fail-fast scaffolding so YAML mistakes
    surface at build time instead of as silent shape bugs downstream.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        mode: str,
        seq_len: int,
        dim: int,
        init_std: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__(in_shape)
        if len(self.in_shape) != 2:
            raise ValueError(f"positional_encoding expects (S, F), got {self.in_shape}")
        S, F = self.in_shape
        seq_len = int(seq_len)
        dim = int(dim)
        if seq_len != S or dim != F:
            raise ValueError(
                f"positional_encoding: seq_len/dim must match in_shape — "
                f"got seq_len={seq_len}, dim={dim} vs in_shape=(S={S}, F={F})"
            )
        if mode not in _MODES:
            raise ValueError(f"positional_encoding: unknown mode {mode!r}; expected one of {_MODES}")

        if mode == "fixed_sincos":
            pe = _sinusoidal_table(seq_len, dim)
            self.register_buffer("pe", pe)
        elif mode == "learnable_random":
            self.pe = nn.Parameter(torch.randn(seq_len, dim) * float(init_std))
        else:  # learnable_sincos
            pe = _sinusoidal_table(seq_len, dim)
            self.pe = nn.Parameter(pe.clone())

        self.mode = mode
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.out_shape = (int(S), int(F))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe)

    def count_flops(self, in_shape: Tuple[int, ...]) -> int:
        # Functional add `x + self.pe`: one add per element. Default leaf walker
        # misses this because it's not an nn.Module call.
        from ...analysis.profiler import default_block_flops
        base = default_block_flops(self, in_shape)
        S, F = in_shape
        return int(base + S * F)


def _sinusoidal_table(seq_len: int, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(
            f"positional_encoding fixed/learnable_sincos modes require even `dim`, got {dim}"
        )
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
