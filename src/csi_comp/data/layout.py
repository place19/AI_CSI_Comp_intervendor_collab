"""Layout adapters: canonical (B, S, P, 2) → backbone-specific input.

CNN backbone:
    input  (real, imag) with each (B, S, P)  →  (B, 2, S, P)

Transformer backbone:
    input  (real, imag) with each (B, S, P)  →  (B, S, P*2)
    where the last dim interleaves real/imag per port: [r0, i0, r1, i1, ...]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayoutAdapter(nn.Module):
    def __init__(self, layout: str):
        super().__init__()
        if layout not in ("cnn", "transformer"):
            raise ValueError(f"unknown layout: {layout!r}")
        self.layout = layout

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        if self.layout == "cnn":
            return torch.stack([real, imag], dim=1)            # (B, 2, S, P)
        # transformer: interleave real/imag per port
        x = torch.stack([real, imag], dim=-1)                  # (B, S, P, 2)
        B, S, P, _ = x.shape
        return x.reshape(B, S, P * 2)                          # (B, S, P*2)


def cnn_mask(mask: torch.Tensor) -> torch.Tensor:
    """(B, S, P) bool  →  (B, 1, S, P) for broadcasting against (B, C, S, P)."""
    return mask.unsqueeze(1)


def transformer_seq_mask(mask: torch.Tensor) -> torch.Tensor:
    """(B, S, P) bool  →  (B, S) bool. A subband is valid if any port at that
    subband is valid."""
    return mask.any(dim=-1)
