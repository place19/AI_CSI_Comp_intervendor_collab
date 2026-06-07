"""Cross-entropy over quantization levels — classify each latent element into the
teacher's codeword bin.

The encoder's pre-quant value (`rescaled_latent`, already mapped into the
quantizer's `value_range`) is scored against every level via `level_logits` — the
same ``-(x - level)^2 / T`` primitive the soft forward/backward paths use. The
teacher's Zq is snapped to its level index and used as the per-element class label,
so this is the discrete-label counterpart of `mse_quantized_latent`: instead of
regressing the code value, it maximises the probability of landing on the correct
bin (a cleaner gradient near bin boundaries, no penalty for sitting deep inside the
correct bin).

Needs `q_levels` / `q_temperature` in `pred_pack` (exposed by the trainer /
Autoencoder.forward from the quantizer) and a teacher latent in `target_pack`
selected by `target_key` (default `latent_target_zq`, the post-quant teacher Zq;
`latent_target_z` also works — it is snapped to the same grid). Independent of the
configured `quant_forward` / `quant_backward`: the logits are computed on demand
from the pre-quant latent, so no `soft` forward is required.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantization.base import snap_to_index
from ..quantization.soft_ops import level_logits
from ..registry import register
from .mse_latent import _require_target


@register("loss", "cross_entropy_levels")
class CrossEntropyLevels(nn.Module):
    """Per-element cross-entropy of the encoder's level distribution vs the teacher bin."""

    name = "cross_entropy_levels"

    def __init__(self, target_key: str = "latent_target_zq", temperature: Optional[float] = None):
        super().__init__()
        self.target_key = target_key
        # None → use the quantizer's temperature (read from pred_pack), so a single
        # annealing schedule drives CE alongside soft forward/backward. Set a float
        # to decouple the CE sharpness from the soft-value temperature.
        self.temperature = None if temperature is None else float(temperature)
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        x = pred_pack["rescaled_latent"]          # (B, ...) pre-quant, in value_range
        levels = pred_pack["q_levels"]            # (N,)
        T = self.temperature if self.temperature is not None else float(pred_pack["q_temperature"])
        target = _require_target(target_pack, self.target_key, self.name)
        if target.shape != x.shape:
            raise ValueError(
                f"{self.name}: target {self.target_key!r} shape {tuple(target.shape)} "
                f"!= rescaled_latent shape {tuple(x.shape)}"
            )
        idx = snap_to_index(target, levels)       # (B, ...) long, teacher bin index
        logits = level_logits(x, levels, T)       # (B, ..., N)
        # F.cross_entropy wants (M, C) logits + (M,) labels — flatten all but levels.
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))
