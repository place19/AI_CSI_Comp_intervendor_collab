"""End-to-end wrapper combining encoder, quantizer, and decoder."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Pipes (real, imag) through encoder → quantizer → decoder.

    Any of {quantizer, decoder} may be None to support encoder-only or
    decoder-skipping configurations. The forward output is a dict so loss
    terms can pull what they need.
    """

    def __init__(
        self,
        encoder: nn.Module,
        quantizer: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> dict[str, Any]:
        latent = self.encoder(real, imag)
        q_latent = self.quantizer(latent) if self.quantizer is not None else latent
        recon = self.decoder(q_latent) if self.decoder is not None else None
        return {"latent": latent, "quantized_latent": q_latent, "recon": recon}
