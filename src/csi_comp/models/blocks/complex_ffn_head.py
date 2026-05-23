"""Complex-output FFN head: two independent FFN branches (real / imag) stacked into a 4-D tensor."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ...registry import register
from .base import Block, make_activation


@register("block", "complex_ffn_head")
class ComplexFFNHead(Block):
    """Two independent FFN branches (real / imag) projecting (B, S, F) → (B, S, max_port)
    each, stacked into a 4-D tensor on a configurable axis.

    Input  shape (excl. batch): (S, F)
    Output shape (excl. batch): depends on `stack_axis`:
        -1  (default) → (S, max_port, 2)
        -2            → (S, 2, max_port)
        -3 / 1        → (2, S, max_port)

    Branches do NOT share weights — `self.real_ffn` and `self.imag_ffn` are independent
    `nn.Sequential(Linear(F, ff_dim) → activation → Dropout? → Linear(ff_dim, max_port))`
    modules.
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        max_port: int,
        ff_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        stack_axis: int = -1,
    ):
        super().__init__(in_shape)
        if len(self.in_shape) != 2:
            raise ValueError(
                f"complex_ffn_head expects (S, F), got {self.in_shape}"
            )
        S, F = self.in_shape
        max_port = int(max_port)
        if max_port <= 0:
            raise ValueError(f"complex_ffn_head: max_port must be positive, got {max_port}")
        ff_dim = int(ff_dim) if ff_dim is not None else 4 * F
        if ff_dim <= 0:
            raise ValueError(f"complex_ffn_head: ff_dim must be positive, got {ff_dim}")

        # Normalise stack_axis against the 4-D output (B, S, max_port) + stacked-pair dim.
        # Allowed axis values are -1, -2, -3 (or equivalently 1, 2, 3); axis=0 would
        # stack before the batch dim and is rejected.
        out_rank = 4
        if not isinstance(stack_axis, int) or isinstance(stack_axis, bool):
            raise ValueError(f"complex_ffn_head: stack_axis must be int, got {stack_axis!r}")
        if stack_axis < -out_rank or stack_axis >= out_rank:
            raise ValueError(
                f"complex_ffn_head: stack_axis {stack_axis} out of range for 4-D output"
            )
        axis_pos = stack_axis if stack_axis >= 0 else out_rank + stack_axis
        if axis_pos == 0:
            raise ValueError(
                "complex_ffn_head: stack_axis=0 would stack before the batch dim; "
                "use -1, -2, or -3 (equivalently 1, 2, 3)."
            )

        def _branch() -> nn.Sequential:
            layers = [nn.Linear(F, ff_dim), make_activation(activation)]
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(ff_dim, max_port))
            return nn.Sequential(*layers)

        self.real_ffn = _branch()
        self.imag_ffn = _branch()
        self.max_port = max_port
        self.ff_dim = ff_dim
        self.stack_axis = axis_pos  # positive index into the 4-D output (with batch).

        # Compute out_shape (excluding batch) by simulating the stack.
        # Branch output excl. batch: (S, max_port). Inserting the pair dim at axis_pos
        # (in batch-inclusive coordinates) corresponds to axis_pos - 1 in excl-batch coords.
        excl_axis = axis_pos - 1
        per_branch = [int(S), int(max_port)]
        per_branch.insert(excl_axis, 2)
        self.out_shape = tuple(per_branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.real_ffn(x)
        i = self.imag_ffn(x)
        return torch.stack([r, i], dim=self.stack_axis)
