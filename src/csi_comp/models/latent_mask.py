"""Latent masking spec and helpers for partial-latent experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LatentMaskSpec:
    mode: str = "full"        # "full" | "half" | "dual" | "random"
    mask_ratio: float = 0.5   # fraction of latent elements zeroed (trailing)


_VALID_MODES = {"full", "half", "dual", "random"}


def parse_latent_mask_spec(cfg: Optional[dict]) -> Optional[LatentMaskSpec]:
    if cfg is None:
        return None
    mode = str(cfg.get("mode", "full"))
    if mode not in _VALID_MODES:
        raise ValueError(f"latent_mask.mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
    mask_ratio = float(cfg.get("mask_ratio", 0.5))
    if not (0.0 < mask_ratio <= 1.0):
        raise ValueError(f"latent_mask.mask_ratio must be in (0, 1], got {mask_ratio}")
    if mode == "full":
        return None  # full mode == no masking; treat as absent
    return LatentMaskSpec(mode=mode, mask_ratio=mask_ratio)


def apply_latent_mask(q_latent: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Zero the trailing `mask_ratio` fraction of latent elements (per sample).

    Works on any shape (B, ...) by flattening to (B, D), masking, then restoring.
    The *first* (1 - mask_ratio) fraction is kept; the rest is set to 0.
    """
    B = q_latent.shape[0]
    flat = q_latent.reshape(B, -1)
    D = flat.shape[1]
    keep = max(1, int(D * (1.0 - mask_ratio)))
    out = flat.clone()
    out[:, keep:] = 0.0
    return out.view(q_latent.shape)


def apply_random_latent_mask(q_latent: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Per-sample random masking: each sample independently has 50% chance to be masked."""
    B = q_latent.shape[0]
    flat = q_latent.reshape(B, -1)
    D = flat.shape[1]
    keep = max(1, int(D * (1.0 - mask_ratio)))
    out = flat.clone()
    # Each sample independently masked with probability 0.5
    to_mask = torch.rand(B, device=q_latent.device) < 0.5  # (B,) bool
    if to_mask.any():
        out[to_mask, keep:] = 0.0
    return out.view(q_latent.shape)
