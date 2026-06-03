"""MSE on the encoder latent — used for encoder-only training against a known target."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register


@register("loss", "mse_latent")
class MSELatent(nn.Module):
    """MSE between the encoder's bounded latent and a known target latent."""

    name = "mse_latent"

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        pred = pred_pack["latent"]
        target = target_pack["latent_target"]
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

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        pred = pred_pack["rescaled_latent"]
        target = target_pack["latent_target"]
        return F.mse_loss(pred, target)


@register("loss", "mse_quantized_latent")
class MSEQuantizedLatent(nn.Module):
    """MSE against the quantized latent — relevant when matching post-quantization codes."""

    name = "mse_quantized_latent"

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        pred = pred_pack["quantized_latent"]
        target = target_pack["latent_target"]
        return F.mse_loss(pred, target)
