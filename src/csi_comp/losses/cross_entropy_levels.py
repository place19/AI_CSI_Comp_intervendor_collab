"""Cross-entropy over quantization levels — classify each latent element into the
teacher's codeword bin.

The encoder's pre-quant value (`rescaled_latent`, already mapped into the
quantizer's `value_range`) is scored against every level via `level_logits` — the
same ``-(x - level)^2 / T`` primitive the soft forward/backward paths use.

Two label modes:

  * **hard** (default): the teacher latent is snapped to its level index
    (`snap_to_index`) and used as the per-element class label, so this is the
    discrete-label counterpart of `mse_quantized_latent` — instead of regressing
    the code value, it maximises the probability of landing on the correct bin
    (cleaner gradient near bin boundaries, no penalty for sitting deep inside the
    correct bin). Typically supervised against the post-quant teacher Zq.
  * **soft** (`soft_labels: true`): the teacher latent is turned into a full
    distribution over levels via `soft_assign(target, levels, teacher_temperature)`
    and used as a soft cross-entropy target (knowledge-distillation style). This
    keeps the "how close to the bin boundary was the teacher" information that
    snapping discards, so it should be supervised against the **pre-quant** teacher
    Z (`target_key: latent_target_z`) — Zq is already snapped and would yield a
    near-one-hot label. `teacher_temperature` controls the label sharpness
    independently of the student-logit `temperature` (it falls back to the student
    `temperature` when unset; `teacher_temperature → 0` recovers the hard label).

Needs `q_levels` / `q_temperature` in `pred_pack` (exposed by the trainer /
Autoencoder.forward from the quantizer) and a teacher latent in `target_pack`
selected by `target_key` (default `latent_target_zq`; for soft labels set
`latent_target_z`). Independent of the configured `quant_forward` /
`quant_backward`: the logits are computed on demand from the pre-quant latent, so
no `soft` forward is required.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantization.base import snap_to_index
from ..quantization.soft_ops import level_logits, soft_assign
from ..registry import register
from .mse_latent import _require_target


@register("loss", "cross_entropy_levels")
class CrossEntropyLevels(nn.Module):
    """Per-element cross-entropy of the encoder's level distribution vs the teacher bin.

    With `soft_labels=False` (default) the teacher is a hard bin index; with
    `soft_labels=True` the teacher is a full soft distribution over levels
    (`soft_assign` at `teacher_temperature`) — see the module docstring.
    """

    name = "cross_entropy_levels"

    def __init__(
        self,
        target_key: str = "latent_target_zq",
        temperature: Optional[float] = None,
        soft_labels: bool = False,
        teacher_temperature: Optional[float] = None,
    ):
        super().__init__()
        self.target_key = target_key
        # None → use the quantizer's temperature (read from pred_pack), so a single
        # annealing schedule drives CE alongside soft forward/backward. Set a float
        # to decouple the CE sharpness from the soft-value temperature.
        self.temperature = None if temperature is None else float(temperature)
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        # Soft-label (knowledge-distillation) mode: build the teacher label as a
        # distribution over levels instead of a one-hot bin index.
        self.soft_labels = bool(soft_labels)
        # Sharpness of the teacher's soft label. None → reuse the student logit
        # temperature `T`. Only consulted when soft_labels is True.
        self.teacher_temperature = (
            None if teacher_temperature is None else float(teacher_temperature)
        )
        if self.teacher_temperature is not None and self.teacher_temperature <= 0:
            raise ValueError(
                f"teacher_temperature must be > 0, got {teacher_temperature}"
            )

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
        logits = level_logits(x, levels, T)       # (B, ..., N)
        # F.cross_entropy wants (M, C) logits + a per-row target — flatten all but levels.
        logits_flat = logits.reshape(-1, logits.shape[-1])
        if self.soft_labels:
            # Teacher distribution over levels (KD-style soft target). Detach so the
            # label is a fixed target — it derives from batch data so it carries no
            # grad anyway, but be explicit. F.cross_entropy accepts class-probability
            # targets shaped (M, C).
            T_teacher = self.teacher_temperature if self.teacher_temperature is not None else T
            soft_target = soft_assign(target, levels, T_teacher).detach()  # (B, ..., N)
            return F.cross_entropy(logits_flat, soft_target.reshape(-1, soft_target.shape[-1]))
        idx = snap_to_index(target, levels)       # (B, ...) long, teacher bin index
        return F.cross_entropy(logits_flat, idx.reshape(-1))
