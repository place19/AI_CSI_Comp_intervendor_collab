"""MSE on the encoder latent — used for encoder-only training against a known target.

Each term reads a fixed `pred` key (the encoder stage it supervises) and a
configurable `target_key` (which teacher latent in the batch to compare against).
`target_key` defaults to `latent_target` (backward compatible); set it to
`latent_target_z` (pre-quant teacher Z) or `latent_target_zq` (post-quant teacher
Zq) — exposed via `data.dataset_args.expose_z` / `expose_zq` (npz or lmdb_raw) —
to supervise different stages against different teachers in a single run.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register


def _require_target(target_pack: dict[str, Any], key: str, loss_name: str) -> torch.Tensor:
    if key not in target_pack:
        raise KeyError(
            f"{loss_name}: target_key {key!r} not in batch. Expose it via "
            f"data.dataset_args.expose_z / expose_zq (npz or lmdb_raw), "
            f"or set latent_key (npz)."
        )
    return target_pack[key]


@register("loss", "mse_latent")
class MSELatent(nn.Module):
    """MSE between the encoder's bounded latent and a known target latent."""

    name = "mse_latent"

    def __init__(self, target_key: str = "latent_target"):
        super().__init__()
        self.target_key = target_key

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        pred = pred_pack["latent"]
        target = _require_target(target_pack, self.target_key, self.name)
        return F.mse_loss(pred, target)


@register("loss", "mse_rescaled_latent")
class MSERescaledLatent(nn.Module):
    """MSE between the rescaled latent and a known target latent.

    The rescaled latent is the encoder output mapped into the quantizer's
    `value_range` (the pre-quantization affine `alpha*x + beta`), i.e. the value
    fed to the quantizer *before* snapping. Sits between `mse_latent` (raw encoder
    output) and `mse_quantized_latent` (post-quantization). Identity transform
    (== `mse_latent`) when the quantizer has no `encoder_value_range`.
    """

    name = "mse_rescaled_latent"

    def __init__(self, target_key: str = "latent_target"):
        super().__init__()
        self.target_key = target_key

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        pred = pred_pack["rescaled_latent"]
        target = _require_target(target_pack, self.target_key, self.name)
        return F.mse_loss(pred, target)


@register("loss", "mse_quantized_latent")
class MSEQuantizedLatent(nn.Module):
    """MSE against the quantized latent — relevant when matching post-quantization codes."""

    name = "mse_quantized_latent"

    def __init__(self, target_key: str = "latent_target"):
        super().__init__()
        self.target_key = target_key

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        pred = pred_pack["quantized_latent"]
        target = _require_target(target_pack, self.target_key, self.name)
        return F.mse_loss(pred, target)
