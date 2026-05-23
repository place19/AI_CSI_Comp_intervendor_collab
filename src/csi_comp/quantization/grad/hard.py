"""Hard snap with no gradient. Used for ONNX export."""
from __future__ import annotations

import torch
import torch.nn as nn

from ...registry import register
from ..base import snap_to_nearest


@register("grad", "hard")
class Hard(nn.Module):
    def forward(self, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        return snap_to_nearest(x, levels)
