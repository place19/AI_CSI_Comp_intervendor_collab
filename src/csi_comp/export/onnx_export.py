"""ONNX export for encoder / encoder+quant / decoder / full autoencoder.

The same checkpoint feeds every scope. Quantization in the exported graph uses
the `hard` strategy so there's no STE detach branch and the op is clean.

Input convention for encoder-facing scopes:
    The exported ONNX model accepts a single pre-combined `input` tensor rather
    than separate `real` and `imag` inputs, eliminating the LayoutAdapter stack
    from the graph. Pre-combine using the same [imag, real] channel order that
    `LayoutAdapter` uses internally:
        CNN layout:         input  (1, 2, S, P)  — ch0=imag, ch1=real
        Transformer layout: input  (1, S, 2*P)   — interleaved [i0, r0, i1, r1, …]
"""
from __future__ import annotations

import copy
import inspect
import types
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from ..models import Autoencoder
from .fuse import fuse_for_inference

VALID_SCOPES = ("encoder", "encoder_quant", "decoder", "full")


# ---------- ONNX-specific MHA patch ----------

def _patch_mha_for_onnx(model: nn.Module) -> None:
    """Replace MHA forward with a cast-free version.

    The training forward wraps softmax in an fp32 island (autocast disabled +
    scores.float() + attn.to(dtype)) to guard against AMP backprop blow-ups.
    For inference-only ONNX export the casts are unnecessary and produce
    spurious Cast nodes. This patches the deep-copied model before tracing.
    """
    from ..models.blocks.transformer import MultiHeadSelfAttention

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        S = self.seq_len if self.seq_len is not None else x.shape[1]
        q = self.W_Q(x).view(-1, S, self.nhead, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(-1, S, self.nhead, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(-1, S, self.nhead, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(-1, S, self.d_model)
        return self.W_O(out)

    for m in model.modules():
        if isinstance(m, MultiHeadSelfAttention):
            m.forward = types.MethodType(_fwd, m)


# ---------- pre-combination helper ----------

def _combine_csi(real: torch.Tensor, imag: torch.Tensor, layout: str) -> torch.Tensor:
    """Stack real/imag into a single CSI tensor matching the LayoutAdapter convention."""
    if layout == "cnn":
        return torch.stack([imag, real], dim=1)   # (B, 2, S, P) — ch0=imag, ch1=real
    # transformer: interleave per port → [i0, r0, i1, r1, …]
    x = torch.stack([imag, real], dim=-1)          # (B, S, P, 2)
    return x.flatten(2)                            # (B, S, P*2)


# ---------- wrappers per scope ----------

class _EncoderOnly(nn.Module):
    def __init__(self, ae: Autoencoder):
        super().__init__()
        self.blocks = ae.encoder.blocks

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        x = csi
        for b in self.blocks:
            x = b(x)
        return x


class _EncoderQuant(nn.Module):
    def __init__(self, ae: Autoencoder, hard_quant: nn.Module):
        super().__init__()
        self.blocks = ae.encoder.blocks
        self.quant = hard_quant

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        x = csi
        for b in self.blocks:
            x = b(x)
        return self.quant(x)


class _DecoderOnly(nn.Module):
    def __init__(self, ae: Autoencoder):
        super().__init__()
        self.decoder = ae.decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class _Full(nn.Module):
    def __init__(self, ae: Autoencoder, hard_quant: nn.Module):
        super().__init__()
        self.blocks = ae.encoder.blocks
        self.quant = hard_quant
        self.decoder = ae.decoder

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        x = csi
        for b in self.blocks:
            x = b(x)
        return self.decoder(self.quant(x))


# ---------- helpers ----------

def _hardened_quantizer(ae: Autoencoder) -> nn.Module:
    if ae.quantizer is None:
        raise ValueError("autoencoder has no quantizer")
    q = copy.deepcopy(ae.quantizer)
    q.to_hard()
    return q


def _example_inputs(
    data_cfg: dict, latent_shape: Tuple[int, ...] | None, scope: str, batch: int = 1
):
    S, P = int(data_cfg["max_subband"]), int(data_cfg["max_port"])
    layout = data_cfg.get("layout", "cnn")
    if scope == "decoder":
        if latent_shape is None:
            raise ValueError("decoder export needs latent_shape")
        return (torch.randn(batch, *latent_shape),)
    real = torch.randn(batch, S, P)
    imag = torch.randn(batch, S, P)
    return (_combine_csi(real, imag, layout),)


def _dynamic_axes(scope: str, dynamic_shape: bool, layout: str = "cnn") -> dict:
    if scope == "decoder":
        out = {"latent": {0: "B"}, "output": {0: "B"}}
    else:
        out = {"input": {0: "B"}, "output": {0: "B"}}
    if dynamic_shape:
        if scope != "decoder":
            if layout == "cnn":
                out["input"].update({2: "S", 3: "P"})
            else:  # transformer
                out["input"].update({1: "S", 2: "F"})
        if scope in ("decoder", "full"):
            out["output"].update({1: "S", 2: "P"})
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
    _patch_mha_for_onnx(ae_export)

    layout = cfg["data"].get("layout", "cnn")

    if scope == "encoder":
        if ae_export.encoder is None:
            raise ValueError("encoder not present in autoencoder")
        wrapper = _EncoderOnly(ae_export)
        in_names = ["input"]
    elif scope == "encoder_quant":
        if ae_export.encoder is None or ae_export.quantizer is None:
            raise ValueError("encoder+quant export requires both encoder and quantizer")
        wrapper = _EncoderQuant(ae_export, _hardened_quantizer(ae_export))
        in_names = ["input"]
    elif scope == "decoder":
        if ae_export.decoder is None:
            raise ValueError("decoder not present in autoencoder")
        wrapper = _DecoderOnly(ae_export)
        in_names = ["latent"]
    else:  # full
        if ae_export.encoder is None or ae_export.decoder is None or ae_export.quantizer is None:
            raise ValueError("full export requires encoder, quantizer, and decoder")
        wrapper = _Full(ae_export, _hardened_quantizer(ae_export))
        in_names = ["input"]

    if scope == "decoder" and latent_shape is None:
        latent_shape = tuple(ae_export.decoder.blocks[0].in_shape)

    inputs = _example_inputs(cfg["data"], latent_shape, scope)
    wrapper.eval()
    # dynamo= was added in PyTorch 2.2; omit on 2.1.x to avoid TypeError.
    _export_kw: dict = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        _export_kw["dynamo"] = False
    torch.onnx.export(
        wrapper,
        inputs,
        str(out_path),
        input_names=in_names,
        output_names=["output"],
        opset_version=opset,
        dynamic_axes=_dynamic_axes(scope, dynamic_shape, layout=layout),
        **_export_kw,
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
    data_cfg = cfg["data"]
    S, P = int(data_cfg["max_subband"]), int(data_cfg["max_port"])
    layout = data_cfg.get("layout", "cnn")
    batch = 1

    if scope == "decoder" and latent_shape is None:
        latent_shape = tuple(ae.decoder.blocks[0].in_shape)

    ae_eval = copy.deepcopy(ae).eval()
    if ae_eval.quantizer is not None:
        ae_eval.quantizer.to_hard()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    if scope == "decoder":
        latent = torch.randn(batch, *latent_shape)
        with torch.no_grad():
            ref = ae_eval.decoder(latent)
        feeds = {"latent": latent.numpy()}
    else:
        real = torch.randn(batch, S, P)
        imag = torch.randn(batch, S, P)
        with torch.no_grad():
            if scope == "encoder":
                ref = ae_eval.encoder(real, imag)
            elif scope == "encoder_quant":
                ref = ae_eval.quantizer(ae_eval.encoder(real, imag))
            else:  # full
                latent = ae_eval.encoder(real, imag)
                ref = ae_eval.decoder(ae_eval.quantizer(latent))
        feeds = {"input": _combine_csi(real, imag, layout).numpy()}

    (ort_out,) = sess.run(["output"], feeds)
    diff = float(np.max(np.abs(ref.numpy() - ort_out)))
    if diff > atol:
        raise AssertionError(f"ONNX parity diff {diff} > atol {atol}")
    return diff
