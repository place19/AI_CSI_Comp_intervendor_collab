"""MLP / Linear projection block."""
from __future__ import annotations

from math import prod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ...registry import register
from .base import Block, make_activation


@register("block", "linear_proj")
class LinearProj(Block):
    """Flatten (batch-preserving) → Linear → (optional norm) → activation.

    Input  shape (excl. batch): any
    Output shape (excl. batch): (out_dim,)

    `norm`:
        none       → nn.Identity (default)
        batchnorm  → nn.BatchNorm1d(out_dim); fuses into the Linear at inference
        layernorm  → nn.LayerNorm(out_dim)

    `bias` default depends on `norm`:
        norm == "batchnorm" → False (BN re-centers; bias is redundant pre-fuse)
        otherwise           → True
    Explicit `bias: true|false` in YAML always wins.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_dim: int,
        activation: str = "identity",
        norm: Optional[str] = "none",
        bias: Optional[bool] = None,
    ):
        super().__init__(in_shape)
        in_features = int(prod(self.in_shape))
        if in_features <= 0:
            raise ValueError(f"linear_proj: empty in_shape {self.in_shape}")

        bias_eff = (norm != "batchnorm") if bias is None else bool(bias)
        self.linear = nn.Linear(in_features, int(out_dim), bias=bias_eff)

        if norm in (None, "none"):
            self.norm: nn.Module = nn.Identity()
        elif norm == "batchnorm":
            self.norm = nn.BatchNorm1d(int(out_dim))
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(int(out_dim))
        else:
            raise ValueError(f"linear_proj: unknown norm {norm!r}; expected none|batchnorm|layernorm")

        self.act = make_activation(activation)

        if isinstance(self.norm, nn.BatchNorm1d):
            self.fusion_pairs = [(self.linear, self.norm)]

        self.out_shape = (int(out_dim),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.linear(x)
        x = self.norm(x)
        return self.act(x)
