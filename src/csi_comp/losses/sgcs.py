"""Squared Generalised Cosine Similarity, computed in real arithmetic.

For a per-subband complex precoder w (length P) and its reconstruction w_hat,
SGCS = |<w, w_hat>|^2 / (||w||^2 * ||w_hat||^2)

We keep the (real, imag) pair as the last tensor dimension instead of using
torch.complex, so this stays ONNX-friendly.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..registry import register


def sgcs_per_subband(
    w: torch.Tensor,
    w_hat: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """SGCS for each (batch, subband). Inputs are (B, S, P, 2) where the last
    dim is [real, imag]. Returns (B, S)."""
    if w.shape != w_hat.shape:
        raise ValueError(f"shape mismatch: {w.shape} vs {w_hat.shape}")
    if w.dim() < 4 or w.shape[-1] != 2:
        raise ValueError(f"expected (..., S, P, 2), got {w.shape}")

    w_r, w_i = w[..., 0], w[..., 1]
    h_r, h_i = w_hat[..., 0], w_hat[..., 1]

    # Complex inner product <w, w_hat> = sum_p conj(w_p) * w_hat_p
    inner_re = (w_r * h_r + w_i * h_i).sum(dim=-1)
    inner_im = (w_r * h_i - w_i * h_r).sum(dim=-1)
    num = inner_re * inner_re + inner_im * inner_im

    den_w = (w_r * w_r + w_i * w_i).sum(dim=-1)
    den_h = (h_r * h_r + h_i * h_i).sum(dim=-1)
    return num / (den_w * den_h + eps)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / (mask.sum() + eps)


@register("loss", "one_minus_sgcs")
class OneMinusSGCS(nn.Module):
    name = "one_minus_sgcs"

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        recon = pred_pack["recon"]                      # (B, S, P, 2)
        target = target_pack["precoder"]                # (B, S, P, 2)
        mask: Optional[torch.Tensor] = target_pack.get("mask")  # (B, S, P) bool

        if mask is not None:
            mask4d = mask.unsqueeze(-1)                         # (B, S, P, 1)
            target = target * mask4d
            recon  = recon  * mask4d
            sgcs = sgcs_per_subband(target, recon, eps=self.eps)  # (B, S)
            sb_valid = mask.any(dim=-1)                           # (B, S) bool
            mean_sgcs = _masked_mean(sgcs, sb_valid, self.eps)
        else:
            sgcs = sgcs_per_subband(target, recon, eps=self.eps)
            mean_sgcs = sgcs.mean()
        return 1.0 - mean_sgcs
