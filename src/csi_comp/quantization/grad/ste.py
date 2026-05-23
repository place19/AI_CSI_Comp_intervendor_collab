"""Straight-Through Estimator: forward = hard snap, backward = identity."""
from __future__ import annotations

import torch
import torch.nn as nn

from ...registry import register
from ..base import snap_to_nearest


@register("grad", "ste")
class STE(nn.Module):
    def forward(self, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        q = snap_to_nearest(x, levels)
        return x + (q - x).detach()
