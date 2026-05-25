"""torch.compile wrapping must not change state_dict key shape on disk."""
from __future__ import annotations

import pytest
import torch

from csi_comp.training.compile_utils import (
    compile_autoencoder_inplace,
    maybe_compile,
    unwrap_compiled,
    uses_cuda_graphs,
)


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))


def test_unwrap_on_uncompiled_is_identity():
    m = _tiny_model()
    assert unwrap_compiled(m) is m


def test_unwrap_handles_none():
    assert unwrap_compiled(None) is None


def test_uses_cuda_graphs_false_when_disabled():
    assert uses_cuda_graphs(None) is False
    assert uses_cuda_graphs({"enabled": False}) is False
    assert uses_cuda_graphs({"enabled": True, "mode": "default"}) is False


def test_uses_cuda_graphs_true_for_cuda_graph_modes():
    assert uses_cuda_graphs({"enabled": True, "mode": "reduce-overhead"}) is True
    assert uses_cuda_graphs({"enabled": True, "mode": "max-autotune"}) is True


def test_uses_cuda_graphs_false_for_no_cudagraphs_mode():
    assert uses_cuda_graphs({"enabled": True, "mode": "max-autotune-no-cudagraphs"}) is False


def test_maybe_compile_disabled_is_passthrough():
    m = _tiny_model()
    assert maybe_compile(m, None) is m
    assert maybe_compile(m, {"enabled": False}) is m


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_maybe_compile_wraps_and_unwrap_reaches_underlying():
    m = _tiny_model()
    compiled = maybe_compile(m, {"enabled": True})
    if compiled is m:
        pytest.skip("torch.compile inactive on this build (e.g. cpu-only)")
    assert unwrap_compiled(compiled) is m


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_checkpoint_keys_have_no_orig_mod_prefix(npz_root, tmp_path):
    """A compiled autoencoder must save state_dict keys without `_orig_mod.`,
    and that file must load cleanly into a freshly-built uncompiled model."""
    from csi_comp.training import (
        build_model, get_mode_spec,
    )
    from csi_comp.training.checkpoint import load_checkpoint, save_checkpoint

    cfg = {
        "experiment": {"seed": 0},
        "data": {
            "format": "npz",
            "train_path": str(npz_root / "train.npz"),
            "val_path": str(npz_root / "val.npz"),
            "max_subband": 8, "max_port": 16, "batch_size": 4, "layout": "cnn",
        },
        "model": {
            "encoder": {"blocks": [
                {"name": "cnn_block", "channels": 4, "kernel": 3, "padding": "same"},
                {"name": "linear_proj", "out_dim": 8},
                {"name": "activation", "activation": "tanh"},
            ]},
            "decoder": {"blocks": [
                {"name": "reshape_head", "max_subband": 8, "max_port": 16},
            ]},
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0],
                      "unit_spaced": True, "grad": "ste"},
        "training": {"mode": "joint", "epochs": 1,
                     "optimizer": {"name": "adamw", "lr": 1e-3},
                     "compile": {"enabled": True}},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    compile_autoencoder_inplace(ae, cfg["training"]["compile"])

    optimizer = torch.optim.AdamW(
        [p for p in ae.parameters() if p.requires_grad] or [torch.zeros(1, requires_grad=True)],
        lr=1e-3,
    )
    ckpt_path = tmp_path / "compiled.pt"
    save_checkpoint(ckpt_path, ae, optimizer, scheduler=None,
                    epoch=0, global_step=0, best_value=0.0, config=cfg)

    sd_disk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for name in ("encoder", "decoder", "quantizer"):
        s = sd_disk.get(name)
        if s is None:
            continue
        for k in s.keys():
            assert not k.startswith("_orig_mod."), f"{name} key {k!r} carries _orig_mod. prefix"

    # Fresh uncompiled build loads the checkpoint cleanly.
    ae2, _, _ = build_model(cfg, spec)
    load_checkpoint(ckpt_path, ae2, optimizer=None, scheduler=None, strict=True)

    # Outputs match between (compiled-source state) and (uncompiled-target state)
    # because we restored identical parameters.
    real = torch.randn(2, 8, 16)
    imag = torch.randn(2, 8, 16)
    with torch.no_grad():
        out1 = unwrap_compiled(ae.encoder)(real, imag)
        out2 = ae2.encoder(real, imag)
    assert torch.allclose(out1, out2, atol=1e-6)
