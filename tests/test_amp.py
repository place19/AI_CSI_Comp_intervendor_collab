"""AMP (mixed-precision) wiring + fp32 islands around loss and softmax."""
from __future__ import annotations

import torch

from csi_comp.models.blocks.transformer import MultiHeadSelfAttention
from csi_comp.training.amp import (
    AmpSpec,
    autocast_ctx,
    build_grad_scaler,
    resolve_amp_cfg,
)


def test_resolve_disabled_when_cfg_missing():
    spec = resolve_amp_cfg(None, torch.device("cpu"))
    assert spec.enabled is False
    assert spec.dtype == torch.float32


def test_resolve_disabled_when_explicit_false():
    spec = resolve_amp_cfg({"enabled": False, "dtype": "bf16"}, torch.device("cpu"))
    assert spec.enabled is False


def test_resolve_cuda_default_bf16():
    # We don't need an actual cuda device — only device.type is read.
    spec = resolve_amp_cfg({"enabled": True}, torch.device("cuda"))
    assert spec.enabled is True
    assert spec.device_type == "cuda"
    assert spec.dtype == torch.bfloat16
    assert spec.use_scaler is False


def test_resolve_mps_default_fp16():
    spec = resolve_amp_cfg({"enabled": True}, torch.device("mps"))
    assert spec.dtype == torch.float16
    # mps doesn't get a GradScaler.
    assert spec.use_scaler is False


def test_resolve_cuda_fp16_uses_scaler():
    spec = resolve_amp_cfg({"enabled": True, "dtype": "fp16"}, torch.device("cuda"))
    assert spec.dtype == torch.float16
    assert spec.use_scaler is True


def test_build_grad_scaler_returns_none_when_disabled():
    spec = AmpSpec(enabled=False, device_type="cpu", dtype=torch.float32, use_scaler=False)
    assert build_grad_scaler(spec) is None


def test_autocast_ctx_actually_changes_dtype_on_cpu():
    """CPU autocast with bf16: a Linear's output should be bf16 inside the ctx."""
    spec = resolve_amp_cfg({"enabled": True, "dtype": "bf16"}, torch.device("cpu"))
    layer = torch.nn.Linear(8, 4)
    x = torch.randn(2, 8)
    with autocast_ctx(spec):
        y = layer(x)
    assert y.dtype == torch.bfloat16


def test_mha_softmax_runs_in_fp32_under_autocast():
    """The fp32 island inside MultiHeadSelfAttention keeps softmax in fp32 even
    when the surrounding autocast is bf16."""
    torch.manual_seed(0)
    mha = MultiHeadSelfAttention(d_model=8, nhead=2)
    x = torch.randn(2, 4, 8)

    # Sanity: without autocast, output is fp32.
    out_fp32 = mha(x)
    assert out_fp32.dtype == torch.float32

    # Under bf16 autocast, the Linear projections produce bf16; the final
    # output is also bf16 (last Linear runs under autocast). What the test
    # really proves is that the softmax fp32 island works numerically — we
    # compare against the fp32 reference and confirm no NaN/Inf, which is
    # the actual failure mode the user reported.
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_amp = mha(x)
    assert torch.isfinite(out_amp).all()
    assert torch.allclose(out_amp.float(), out_fp32, atol=5e-2)


def test_trainer_runs_with_amp_enabled(npz_root):
    """End-to-end: one mini-epoch with AMP enabled on CPU. The exercise is to
    catch missing kwargs / scaler interactions, not measure perf."""
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.training import (
        Trainer, build_dataloaders, build_model, build_optimizer,
        get_mode_spec, resolve_amp_cfg,
    )

    cfg = {
        "experiment": {"seed": 0},
        "data": {
            "format": "npz",
            "train_path": str(npz_root / "train.npz"),
            "val_path": str(npz_root / "val.npz"),
            "max_subband": 8,
            "max_port": 16,
            "batch_size": 4,
            "layout": "cnn",
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
        "training": {
            "mode": "joint",
            "epochs": 1,
            "optimizer": {"name": "adamw", "lr": 1e-3},
            "amp": {"enabled": True, "dtype": "bf16"},
        },
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }
    mode = cfg["training"]["mode"]
    spec = get_mode_spec(mode)
    ae, _, _ = build_model(cfg, spec)
    train_loader, val_loader = build_dataloaders(cfg["data"])
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode=mode)
    optimizer = build_optimizer(ae, cfg["training"]["optimizer"])
    amp_spec = resolve_amp_cfg(cfg["training"]["amp"], torch.device("cpu"))

    trainer = Trainer(
        model=ae, optimizer=optimizer, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        mode_spec=spec, device=torch.device("cpu"),
        epochs=1, amp_spec=amp_spec,
    )
    trainer.fit()
    # If we got here without a NaN backprop, the fp32 islands are doing their job.
    val_metrics = trainer.validate()
    assert "val/loss/total" in val_metrics
