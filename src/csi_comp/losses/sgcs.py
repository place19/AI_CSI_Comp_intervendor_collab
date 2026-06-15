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


def _align_unit_phase(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Unit-norm each per-subband precoder and rotate it so port 0 has zero phase.

    Input/output `(..., P, 2)` with the last dim `[real, imag]`. The L2 norm is
    taken over (P, real/imag) so the whole complex vector becomes unit norm; then
    each vector is multiplied by ``e^{-j·θ₀} = conj(z₀)/|z₀|`` (θ₀ = phase of port
    0), which lands port 0 on the positive real axis. SGCS is invariant to global
    scale and phase, so this canonicalisation is what makes a plain MSE between
    target and reconstruction meaningful (see `nmse_aligned_per_subband`).
    Masked-out ports must already be zeroed by the caller; port 0 is assumed valid.
    """
    r, i = x[..., 0], x[..., 1]                                  # (..., P)
    norm = torch.sqrt((r * r + i * i).sum(dim=-1, keepdim=True) + eps)  # (..., 1)
    r, i = r / norm, i / norm
    r0, i0 = r[..., 0:1], i[..., 0:1]                            # (..., 1) port-0 ref
    mag0 = torch.sqrt(r0 * r0 + i0 * i0 + eps)
    c, s = r0 / mag0, i0 / mag0                                  # rotation e^{-jθ₀}=c-js
    # rotate z=(r+ji) by (c - js): real = r·c + i·s, imag = i·c - r·s
    return torch.stack([r * c + i * s, i * c - r * s], dim=-1)


def nmse_aligned_per_subband(
    w: torch.Tensor,
    w_hat: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalized MSE per (batch, subband) after scale+phase alignment. (B, S).

    Both the target `w` and reconstruction `w_hat` (each `(..., S, P, 2)`) are
    unit-normed and phase-aligned per subband (`_align_unit_phase`); the NMSE is
    then the energy ratio ``Σ_p |w_align − ŵ_align|² / Σ_p |w_align|²`` over ports.
    Because both are unit-norm the denominator is ≈1, so this reduces to the
    aligned reconstruction-error energy — but the ratio form is kept so it stays
    the textbook NMSE. A test-time metric only (SGCS hides scale/phase error).
    Pre-zero masked ports (multiply by the mask) as for `sgcs_per_subband`.
    """
    if w.shape != w_hat.shape:
        raise ValueError(f"shape mismatch: {w.shape} vs {w_hat.shape}")
    if w.dim() < 4 or w.shape[-1] != 2:
        raise ValueError(f"expected (..., S, P, 2), got {w.shape}")
    w = _align_unit_phase(w, eps)
    h = _align_unit_phase(w_hat, eps)
    diff = w - h
    num = (diff[..., 0] ** 2 + diff[..., 1] ** 2).sum(dim=-1)    # (B, S)
    den = (w[..., 0] ** 2 + w[..., 1] ** 2).sum(dim=-1)          # (B, S) ≈ 1
    return num / (den + eps)


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
