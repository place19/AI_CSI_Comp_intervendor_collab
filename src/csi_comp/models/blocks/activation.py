"""Standalone activation block: just applies an activation function in-place in the block chain."""
from __future__ import annotations

from typing import Tuple

import torch

from ...registry import register
from .base import Block, make_activation


@register("block", "activation")
class ActivationBlock(Block):
    """Apply a single activation to the input. Shape is preserved.

    Input shape (excl. batch): any
    Output shape (excl. batch): same as input

    Use when none of the surrounding blocks accept an `activation:` kwarg and
    you need a standalone non-linearity in the chain. Zero parameters.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        activation: str = "relu",
    ):
        super().__init__(in_shape)
        self.act = make_activation(activation)
        self.activation_name = activation if activation is not None else "identity"
        self.out_shape = tuple(self.in_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
