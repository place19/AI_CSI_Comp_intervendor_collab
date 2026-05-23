"""Dual-latent SGCS loss: weighted sum of full-latent and half-latent reconstruction losses."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..registry import register
from .sgcs import _masked_mean, sgcs_per_subband


@register("loss", "dual_one_minus_sgcs")
class DualOneMinusSGCS(nn.Module):
    """Loss for `dual` masking mode.

    Expects pred_pack to contain both:
      - "recon"      : reconstruction from full quantized latent  (B, S, P, 2)
      - "recon_half" : reconstruction from masked quantized latent (B, S, P, 2)

    Returns full_weight * (1 - SGCS_full) + half_weight * (1 - SGCS_half).
    """

    name = "dual_one_minus_sgcs"

    def __init__(
        self,
        full_weight: float = 0.5,
        half_weight: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.full_weight = float(full_weight)
        self.half_weight = float(half_weight)
        self.eps = float(eps)

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        recon_full = pred_pack["recon"]
        recon_half = pred_pack["recon_half"]
        target = target_pack["precoder"]
        mask = target_pack.get("mask")

        if mask is not None:
            mask4d = mask.unsqueeze(-1)                              # (B, S, P, 1)
            target     = target     * mask4d
            recon_full = recon_full * mask4d
            recon_half = recon_half * mask4d
            sgcs_full = sgcs_per_subband(target, recon_full, eps=self.eps)
            sgcs_half = sgcs_per_subband(target, recon_half, eps=self.eps)
            sb_valid = mask.any(dim=-1)
            mean_full = _masked_mean(sgcs_full, sb_valid, self.eps)
            mean_half = _masked_mean(sgcs_half, sb_valid, self.eps)
        else:
            sgcs_full = sgcs_per_subband(target, recon_full, eps=self.eps)
            sgcs_half = sgcs_per_subband(target, recon_half, eps=self.eps)
            mean_full = sgcs_full.mean()
            mean_half = sgcs_half.mean()

        return self.full_weight * (1.0 - mean_full) + self.half_weight * (1.0 - mean_half)
