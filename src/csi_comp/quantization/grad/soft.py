"""Soft (differentiable) quantization via softmax over levels."""
from __future__ import annotations

import torch
import torch.nn as nn

from ...registry import register


@register("grad", "soft")
class SoftQuant(nn.Module):
    """Output = sum_i softmax(-(x-level_i)^2 / temperature) * level_i.

    As temperature → 0 this collapses to the nearest level.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        sq = (x.unsqueeze(-1) - levels) ** 2
        w = torch.softmax(-sq / self.temperature, dim=-1)
        return (w * levels).sum(dim=-1)
