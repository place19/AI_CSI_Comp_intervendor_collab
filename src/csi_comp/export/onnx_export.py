"""ONNX export for encoder / encoder+quant / decoder / full autoencoder.

The same checkpoint feeds every scope. Quantization in the exported graph uses
the `hard` strategy so there's no STE detach branch and the op is clean.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from ..models import Autoencoder
from .fuse import fuse_for_inference

VALID_SCOPES = ("encoder", "encoder_quant", "decoder", "full")


# ---------- wrappers per scope ----------

class _EncoderOnly(nn.Module):
    def __init__(self, ae: Autoencoder):
        super().__init__()
        self.encoder = ae.encoder

    def forward(self, real, imag):
        return self.encoder(real, imag)


class _EncoderQuant(nn.Module):
    def __init__(self, ae: Autoencoder, hard_quant: nn.Module):
        super().__init__()
        self.encoder = ae.encoder
        self.quant = hard_quant

    def forward(self, real, imag):
        return self.quant(self.encoder(real, imag))


class _DecoderOnly(nn.Module):
    def __init__(self, ae: Autoencoder):
        super().__init__()
        self.decoder = ae.decoder

    def forward(self, latent):
        return self.decoder(latent)


class _Full(nn.Module):
    def __init__(self, ae: Autoencoder, hard_quant: nn.Module):
        super().__init__()
        self.encoder = ae.encoder
        self.quant = hard_quant
        self.decoder = ae.decoder

    def forward(self, real, imag):
        latent = self.encoder(real, imag)
        q = self.quant(latent)
        return self.decoder(q)


# ---------- helpers ----------

def _hardened_quantizer(ae: Autoencoder) -> nn.Module:
    if ae.quantizer is None:
        raise ValueError("autoencoder has no quantizer")
    q = copy.deepcopy(ae.quantizer)
    q.to_hard()
    return q


def _example_inputs(
    data_cfg: dict, latent_shape: Tuple[int, ...] | None, scope: str, batch: int = 2
):
    S, P = int(data_cfg["max_subband"]), int(data_cfg["max_port"])
    real = torch.randn(batch, S, P)
    imag = torch.randn(batch, S, P)
    if scope == "decoder":
        if latent_shape is None:
            raise ValueError("decoder export needs latent_shape")
        latent = torch.randn(batch, *latent_shape)
        return (latent,)
    return (real, imag)


def _dynamic_axes(scope: str, dynamic_shape: bool) -> dict:
    if scope == "decoder":
        out = {"latent": {0: "B"}, "output": {0: "B"}}
    else:
        out = {"real": {0: "B"}, "imag": {0: "B"}, "output": {0: "B"}}
    if dynamic_shape:
        for name in list(out.keys()):
            if name in ("real", "imag"):
                out[name].update({1: "S", 2: "P"})
            if name == "output" and scope in ("decoder", "full"):
                out[name].update({1: "S", 2: "P"})
    return out


# ---------- public API ----------

def export_to_onnx(
    ae: Autoencoder,
    cfg: dict,
    scope: str,
    out_path: Path,
    latent_shape: Tuple[int, ...] | None = None,
    dynamic_shape: bool = False,
    opset: int = 17,
    fuse: bool = True,
) -> Path:
    if scope not in VALID_SCOPES:
        raise ValueError(f"scope {scope!r} not in {VALID_SCOPES}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Deep-copy so we never mutate the live (training) model. Fusion folds
    # BN affine params into the preceding conv/linear and replaces the BN
    # with `nn.Identity()` — the exported graph then has no BatchNormalization
    # node, matching how the profiler counts these as fused.
    ae_export = copy.deepcopy(ae).eval()
    if fuse:
        fuse_for_inference(ae_export)

    if scope == "encoder":
        if ae_export.encoder is None:
            raise ValueError("encoder not present in autoencoder")
        wrapper = _EncoderOnly(ae_export)
        in_names = ["real", "imag"]
    elif scope == "encoder_quant":
        if ae_export.encoder is None or ae_export.quantizer is None:
            raise ValueError("encoder+quant export requires both encoder and quantizer")
        wrapper = _EncoderQuant(ae_export, _hardened_quantizer(ae_export))
        in_names = ["real", "imag"]
    elif scope == "decoder":
        if ae_export.decoder is None:
            raise ValueError("decoder not present in autoencoder")
        wrapper = _DecoderOnly(ae_export)
        in_names = ["latent"]
    else:  # full
        if ae_export.encoder is None or ae_export.decoder is None or ae_export.quantizer is None:
            raise ValueError("full export requires encoder, quantizer, and decoder")
        wrapper = _Full(ae_export, _hardened_quantizer(ae_export))
        in_names = ["real", "imag"]

    if scope == "decoder" and latent_shape is None:
        latent_shape = tuple(ae_export.decoder.blocks[0].in_shape)

    inputs = _example_inputs(cfg["data"], latent_shape, scope)
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        inputs,
        str(out_path),
        input_names=in_names,
        output_names=["output"],
        opset_version=opset,
        dynamic_axes=_dynamic_axes(scope, dynamic_shape),
        dynamo=False,
    )
    return out_path


def verify_onnx_parity(
    onnx_path: Path,
    ae: Autoencoder,
    cfg: dict,
    scope: str,
    latent_shape: Tuple[int, ...] | None = None,
    atol: float = 1e-4,
) -> float:
    """Run a fresh random batch through both torch and onnxruntime and return
    the max-abs difference. Used both internally and from tests."""
    if scope == "decoder" and latent_shape is None:
        latent_shape = tuple(ae.decoder.blocks[0].in_shape)
    inputs = _example_inputs(cfg["data"], latent_shape, scope)

    # Torch reference
    ae_eval = copy.deepcopy(ae).eval()
    if ae_eval.quantizer is not None:
        ae_eval.quantizer.to_hard()
    with torch.no_grad():
        if scope == "encoder":
            ref = ae_eval.encoder(*inputs)
        elif scope == "encoder_quant":
            ref = ae_eval.quantizer(ae_eval.encoder(*inputs))
        elif scope == "decoder":
            ref = ae_eval.decoder(*inputs)
        else:
            real, imag = inputs
            latent = ae_eval.encoder(real, imag)
            q = ae_eval.quantizer(latent)
            ref = ae_eval.decoder(q)

    # ONNX inference
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if scope == "decoder":
        (latent,) = inputs
        feeds = {"latent": latent.numpy()}
    else:
        real, imag = inputs
        feeds = {"real": real.numpy(), "imag": imag.numpy()}
    (ort_out,) = sess.run(["output"], feeds)
    diff = float(np.max(np.abs(ref.numpy() - ort_out)))
    if diff > atol:
        raise AssertionError(f"ONNX parity diff {diff} > atol {atol}")
    return diff
