"""Inference-time Conv↔BN / Linear↔BN1d fusion + ONNX-export integration."""
from __future__ import annotations

import copy

import onnx
import torch
import torch.nn as nn

from csi_comp.export.fuse import fuse_for_inference, fuse_linear_bn_eval
from csi_comp.models.blocks.cnn import CnnBlock
from csi_comp.models.blocks.mlp import LinearProj


def test_fuse_conv_bn_matches_unfused_output():
    block = CnnBlock(in_shape=(2, 4, 4), channels=8, kernel=3, padding=1, norm="batchnorm")
    block.eval()
    x = torch.randn(1, 2, 4, 4)
    with torch.no_grad():
        y_ref = block(x)
    fused = fuse_for_inference(copy.deepcopy(block))
    with torch.no_grad():
        y_fused = fused(x)
    assert torch.allclose(y_ref, y_fused, atol=1e-5)
    # The BN slot is now an Identity, and fusion_pairs is consumed.
    assert isinstance(fused.norm, nn.Identity)
    assert fused.fusion_pairs == []


def test_fuse_conv_bn_drops_bn_params():
    block = CnnBlock(in_shape=(2, 4, 4), channels=8, kernel=3, padding=1, norm="batchnorm")
    block.eval()
    n_before = sum(p.numel() for p in block.parameters())
    fused = fuse_for_inference(copy.deepcopy(block))
    n_after = sum(p.numel() for p in fused.parameters())
    # BN has 2*channels affine params; bias on the fused Conv now exists (+channels)
    # so net change = -channels (since the original conv had no bias under BN).
    assert n_after < n_before


def test_fuse_linear_bn_matches_unfused_output():
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=False)
    bn = nn.BatchNorm1d(4)
    # populate running stats
    bn.train()
    for _ in range(5):
        _ = bn(lin(torch.randn(16, 8)))
    lin.eval(); bn.eval()
    x = torch.randn(3, 8)
    with torch.no_grad():
        y_ref = bn(lin(x))
    fused = fuse_linear_bn_eval(lin, bn)
    with torch.no_grad():
        y_fused = fused(x)
    assert torch.allclose(y_ref, y_fused, atol=1e-5)


def test_fuse_idempotent():
    block = CnnBlock(in_shape=(2, 4, 4), channels=8, kernel=3, padding=1, norm="batchnorm")
    block.eval()
    fused = fuse_for_inference(copy.deepcopy(block))
    once = copy.deepcopy(fused)
    fused2 = fuse_for_inference(fused)   # should be no-op
    x = torch.randn(1, 2, 4, 4)
    with torch.no_grad():
        assert torch.allclose(once(x), fused2(x), atol=1e-6)


def test_onnx_export_has_no_batchnormalization_when_fused(npz_root, tmp_path):
    from csi_comp.export import export_to_onnx
    from csi_comp.models import Autoencoder
    from csi_comp.models.encoder import build_encoder
    from csi_comp.quantization.base import build_quantizer
    from csi_comp.models.decoder import build_decoder

    data_cfg = {"layout": "cnn", "max_subband": 8, "max_port": 16}
    model_cfg = {
        "encoder": {"blocks": [
            {"name": "cnn_block", "channels": 4, "kernel": 3, "padding": "same"},
            {"name": "linear_proj", "out_dim": 8},
            {"name": "activation", "activation": "tanh"},
        ]},
        "decoder": {"blocks": [
            {"name": "reshape_head", "max_subband": 8, "max_port": 16},
        ]},
    }
    enc, enc_trace = build_encoder(model_cfg, data_cfg)
    quant = build_quantizer({"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0],
                             "unit_spaced": True, "grad": "ste"})
    dec, _ = build_decoder(model_cfg, data_cfg, enc_trace[-1].out_shape)
    ae = Autoencoder(enc, quant, dec).eval()

    cfg = {"data": data_cfg}

    fused_path = tmp_path / "fused.onnx"
    export_to_onnx(ae, cfg, scope="encoder", out_path=fused_path, fuse=True)
    g = onnx.load(str(fused_path)).graph
    op_types = {n.op_type for n in g.node}
    assert "BatchNormalization" not in op_types

    # Sanity: with fusion off, the fused-Conv path is skipped but the resulting
    # graph still won't contain BN (PyTorch's ONNX exporter folds eval-mode BN
    # into the preceding Conv on its own). The numeric value of `fuse=False` is
    # debug visibility into intermediate fp arithmetic, not a different op set.
    unfused_path = tmp_path / "unfused.onnx"
    export_to_onnx(ae, cfg, scope="encoder", out_path=unfused_path, fuse=False)
    assert unfused_path.exists()
