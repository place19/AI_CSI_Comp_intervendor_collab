"""Pure reshape block: change per-sample shape, batch dim untouched."""
from __future__ import annotations

from math import prod
from typing import Sequence, Tuple

import torch

from ...registry import register
from .base import Block


@register("block", "reshape")
class ReshapeBlock(Block):
    """Reshape (B, *in_shape) → (B, *out_shape). One -1 is allowed.

    The block does not transform values — it's a pure `view`. It has zero
    parameters and zero FLOPs. The `-1` is resolved at construction time
    against `prod(in_shape)` so `self.out_shape` is always a concrete tuple
    (the encoder/decoder builder needs this to propagate shape downstream).
    """

    def __init__(self, in_shape: Tuple[int, ...], out_shape: Sequence[int]):
        super().__init__(in_shape)
        spec = [int(d) for d in out_shape]
        n_neg = sum(1 for d in spec if d == -1)
        if n_neg > 1:
            raise ValueError(f"reshape: at most one -1 allowed, got {spec}")
        if any(d <= 0 and d != -1 for d in spec):
            raise ValueError(f"reshape: dims must be -1 or positive, got {spec}")

        total = int(prod(self.in_shape))
        if n_neg == 1:
            known = int(prod(d for d in spec if d != -1)) if len(spec) > 1 else 1
            if known <= 0 or total % known != 0:
                raise ValueError(
                    f"reshape: cannot infer -1 from in_shape={self.in_shape} "
                    f"and out_shape={spec} (total={total}, known_product={known})"
                )
            inferred = total // known
            resolved = tuple(inferred if d == -1 else d for d in spec)
        else:
            resolved = tuple(spec)
            if int(prod(resolved)) != total:
                raise ValueError(
                    f"reshape: numel mismatch — in_shape={self.in_shape} (total={total}) "
                    f"vs out_shape={resolved} (total={int(prod(resolved))})"
                )
        self.out_shape = resolved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], *self.out_shape)
