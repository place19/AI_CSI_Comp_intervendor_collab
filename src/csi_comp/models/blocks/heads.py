"""Encoder/decoder terminal blocks: BoundingHead, ReshapeHead."""
from __future__ import annotations

from math import prod
from typing import Tuple

import torch
import torch.nn as nn

from ...registry import register
from .base import Block


@register("block", "bounding_head")
class BoundingHead(Block):
    """Bound the encoder output to a configurable range using tanh or sigmoid.

    Input shape == Output shape.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        activation: str = "tanh",
        value_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__(in_shape)
        if activation not in ("tanh", "sigmoid"):
            raise ValueError(f"bounding_head activation must be tanh|sigmoid, got {activation!r}")
        lo, hi = value_range
        if hi <= lo:
            raise ValueError(f"bad value_range: {value_range}")
        self.activation_name = activation
        self.lo = float(lo)
        self.hi = float(hi)
        self.out_shape = tuple(self.in_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            # tanh ∈ [-1, 1] → [lo, hi]
            y = torch.tanh(x)
            y = self.lo + (y + 1.0) * (self.hi - self.lo) * 0.5
        else:  # sigmoid ∈ [0, 1] → [lo, hi]
            y = torch.sigmoid(x)
            y = self.lo + y * (self.hi - self.lo)
        return y


@register("block", "reshape_head")
class ReshapeHead(Block):
    """Final decoder block: flatten → linear → reshape to (max_S, max_P, 2).

    Output shape (excl. batch): (max_subband, max_port, 2)
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        max_subband: int,
        max_port: int,
    ):
        super().__init__(in_shape)
        in_features = int(prod(self.in_shape))
        if in_features <= 0:
            raise ValueError(f"reshape_head: empty in_shape {self.in_shape}")
        self.max_subband = int(max_subband)
        self.max_port = int(max_port)
        out_features = self.max_subband * self.max_port * 2
        self.linear = nn.Linear(in_features, out_features)
        self.out_shape = (self.max_subband, self.max_port, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.linear(x)
        return x.view(-1, self.max_subband, self.max_port, 2)
