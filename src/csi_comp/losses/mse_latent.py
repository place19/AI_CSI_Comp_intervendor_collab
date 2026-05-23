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


@register("loss", "mse_quantized_latent")
class MSEQuantizedLatent(nn.Module):
    """MSE against the quantized latent — relevant when matching post-quantization codes."""

    name = "mse_quantized_latent"

    def forward(self, pred_pack: dict[str, Any], target_pack: dict[str, Any]) -> torch.Tensor:
        pred = pred_pack["quantized_latent"]
        target = target_pack["latent_target"]
        return F.mse_loss(pred, target)
