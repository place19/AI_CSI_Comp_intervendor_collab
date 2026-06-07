"""Tests for latent masking: parse / apply helpers, DualOneMinusSGCS loss, and Trainer integration."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from csi_comp.data import NpzDataset, make_collate_fn
from csi_comp.losses import DualOneMinusSGCS, WeightedSumLoss
from csi_comp.models.latent_mask import (
    LatentMaskSpec,
    apply_latent_mask,
    apply_random_latent_mask,
    parse_latent_mask_spec,
)
from csi_comp.training import (
    Trainer,
    build_model,
    build_optimizer,
    get_mode_spec,
    seed_everything,
)

MAX_S, MAX_P = 8, 12


# ---------------------------------------------------------------------------
# Helpers shared across integration tests
# ---------------------------------------------------------------------------

def _joint_cfg(latent_mask_cfg: dict | None = None):
    cfg = {
        "experiment": {"device": "cpu", "seed": 42},
        "data": {"layout": "cnn", "max_subband": MAX_S, "max_port": MAX_P},
        "model": {
            "encoder": {
                "blocks": [
                    {"name": "cnn_block", "channels": 4, "kernel": 3},
                    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
                    {"name": "activation", "activation": "tanh"},
                ],
            },
            "decoder": {
                "blocks": [
                    {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                    {"name": "reshape_head", "max_subband": MAX_S, "max_port": MAX_P},
                ],
            },
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": "joint"},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }
    if latent_mask_cfg is not None:
        cfg["model"]["latent_mask"] = latent_mask_cfg
    return cfg


def _loaders(npz_root, batch_size: int = 4):
    coll = make_collate_fn(MAX_S, MAX_P)
    train = DataLoader(NpzDataset(npz_root / "train.npz"), batch_size=batch_size,
                       shuffle=False, collate_fn=coll)
    val = DataLoader(NpzDataset(npz_root / "val.npz"), batch_size=batch_size,
                     shuffle=False, collate_fn=coll)
    return train, val


def _make_trainer(cfg, npz_root, mask_spec=None, loss_terms=None):
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    train_loader, val_loader = _loaders(npz_root)
    terms = loss_terms or cfg["loss"]["terms"]
    loss_fn = WeightedSumLoss(terms, mode="joint")
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-3})
    return Trainer(
        model=ae, optimizer=opt, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        mode_spec=spec, device=torch.device("cpu"),
        epochs=1, val_every_n_epochs=1,
        best_metric={"name": "sgcs", "mode": "max"},
        mask_spec=mask_spec,
    )


# ===========================================================================
# parse_latent_mask_spec
# ===========================================================================

def test_parse_none_returns_none():
    assert parse_latent_mask_spec(None) is None


def test_parse_full_returns_none():
    # mode=full is treated as "no masking" — callers can skip the masking path
    assert parse_latent_mask_spec({"mode": "full"}) is None


def test_parse_half_returns_spec():
    spec = parse_latent_mask_spec({"mode": "half", "mask_ratio": 0.5})
    assert isinstance(spec, LatentMaskSpec)
    assert spec.mode == "half"
    assert spec.mask_ratio == pytest.approx(0.5)


def test_parse_dual_returns_spec():
    spec = parse_latent_mask_spec({"mode": "dual"})
    assert spec is not None and spec.mode == "dual"
    assert spec.mask_ratio == pytest.approx(0.5)  # default


def test_parse_random_returns_spec():
    spec = parse_latent_mask_spec({"mode": "random", "mask_ratio": 0.25})
    assert spec is not None and spec.mode == "random"
    assert spec.mask_ratio == pytest.approx(0.25)


def test_parse_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        parse_latent_mask_spec({"mode": "unknown"})


def test_parse_ratio_zero_raises():
    with pytest.raises(ValueError, match="mask_ratio"):
        parse_latent_mask_spec({"mode": "half", "mask_ratio": 0.0})


def test_parse_ratio_above_one_raises():
    with pytest.raises(ValueError, match="mask_ratio"):
        parse_latent_mask_spec({"mode": "half", "mask_ratio": 1.5})


# ===========================================================================
# apply_latent_mask
# ===========================================================================

def test_apply_mask_flat_first_half_preserved():
    q = torch.arange(8.0).view(2, 4)   # [[0,1,2,3], [4,5,6,7]]
    out = apply_latent_mask(q, mask_ratio=0.5)
    # first 2 elements kept, last 2 zeroed
    assert torch.equal(out[:, :2], q[:, :2])
    assert (out[:, 2:] == 0).all()


def test_apply_mask_shape_preserved():
    q = torch.randn(3, 4, 5)
    out = apply_latent_mask(q, mask_ratio=0.5)
    assert out.shape == q.shape


def test_apply_mask_multi_dim_latent():
    # CNN-style latent (B, C, H, W)
    q = torch.ones(2, 4, 3, 3)
    out = apply_latent_mask(q, mask_ratio=0.5)
    B = q.shape[0]
    flat = out.reshape(B, -1)
    D = flat.shape[1]
    keep = int(D * 0.5)
    assert (flat[:, :keep] == 1.0).all(), "kept region should be unchanged"
    assert (flat[:, keep:] == 0.0).all(), "masked region should be zero"


def test_apply_mask_custom_ratio():
    q = torch.ones(1, 100)
    out = apply_latent_mask(q, mask_ratio=0.25)
    flat = out.view(1, -1)
    keep = int(100 * 0.75)
    assert (flat[:, :keep] == 1.0).all()
    assert (flat[:, keep:] == 0.0).all()


def test_apply_mask_does_not_modify_input():
    q = torch.ones(2, 8)
    q_clone = q.clone()
    apply_latent_mask(q, mask_ratio=0.5)
    assert torch.equal(q, q_clone), "apply_latent_mask must not modify the input tensor"


def test_apply_mask_ratio_one_zeros_all_but_one():
    q = torch.ones(1, 10)
    out = apply_latent_mask(q, mask_ratio=1.0)
    flat = out.view(1, -1)
    # keep=max(1, int(10*0.0))=1; first element preserved, rest zeroed
    assert flat[0, 0] == pytest.approx(1.0)
    assert (flat[0, 1:] == 0.0).all()


# ===========================================================================
# apply_random_latent_mask
# ===========================================================================

def test_apply_random_mask_shape_preserved():
    q = torch.randn(4, 16)
    out = apply_random_latent_mask(q, mask_ratio=0.5)
    assert out.shape == q.shape


def test_apply_random_mask_unmasked_samples_unchanged():
    # With a large batch and a fixed seed, verify that samples NOT selected for
    # masking are byte-for-byte equal to the input.
    seed_everything(0)
    B, D = 32, 20
    q = torch.ones(B, D)
    out = apply_random_latent_mask(q, mask_ratio=0.5)
    flat = out.view(B, -1)
    keep = int(D * 0.5)
    # Every sample's first-half must equal the input's first-half (always kept).
    assert (flat[:, :keep] == 1.0).all(), "first half of every sample must be preserved"


def test_apply_random_mask_produces_both_masked_and_full():
    # Over a large batch with p=0.5, we expect both fully-ones rows and rows with trailing zeros.
    torch.manual_seed(42)
    B, D = 128, 16
    keep = int(D * 0.5)
    q = torch.ones(B, D)
    out = apply_random_latent_mask(q, mask_ratio=0.5)
    flat = out.view(B, -1)
    has_masked = (flat[:, keep:] == 0.0).all(dim=1).any()
    has_full = (flat[:, keep:] == 1.0).all(dim=1).any()
    assert has_masked, "at least some samples should be masked"
    assert has_full, "at least some samples should be unmasked"


# ===========================================================================
# DualOneMinusSGCS loss
# ===========================================================================

def test_dual_loss_both_perfect_is_zero():
    B, S, P = 2, 4, 8
    target = torch.randn(B, S, P, 2)
    loss_fn = DualOneMinusSGCS(full_weight=0.5, half_weight=0.5)
    val = loss_fn({"recon": target, "recon_half": target}, {"precoder": target})
    assert val.item() == pytest.approx(0.0, abs=1e-5)


def test_dual_loss_full_weight_only():
    B, S, P = 2, 4, 8
    target = torch.randn(B, S, P, 2)
    bad_recon = torch.zeros_like(target)
    # full=perfect, half=bad, but half_weight=0
    loss_fn = DualOneMinusSGCS(full_weight=1.0, half_weight=0.0)
    val = loss_fn({"recon": target, "recon_half": bad_recon}, {"precoder": target})
    assert val.item() == pytest.approx(0.0, abs=1e-5)


def test_dual_loss_half_weight_only():
    B, S, P = 2, 4, 8
    target = torch.randn(B, S, P, 2)
    bad_recon = torch.zeros_like(target)
    # full=bad, half=perfect, but full_weight=0
    loss_fn = DualOneMinusSGCS(full_weight=0.0, half_weight=1.0)
    val = loss_fn({"recon": bad_recon, "recon_half": target}, {"precoder": target})
    assert val.item() == pytest.approx(0.0, abs=1e-5)


def test_dual_loss_weighted_sum_correct():
    B, S, P = 1, 1, 4
    target = torch.randn(B, S, P, 2)
    from csi_comp.losses.sgcs import sgcs_per_subband
    recon_full = target.clone()                      # SGCS=1 → loss=0
    recon_half = torch.zeros(B, S, P, 2)             # SGCS=0 → loss=1

    for fw, hw in [(0.5, 0.5), (0.3, 0.7), (1.0, 0.0)]:
        loss_fn = DualOneMinusSGCS(full_weight=fw, half_weight=hw)
        val = loss_fn({"recon": recon_full, "recon_half": recon_half}, {"precoder": target})
        expected = fw * 0.0 + hw * 1.0
        assert val.item() == pytest.approx(expected, abs=1e-5), f"fw={fw}, hw={hw}"


def test_dual_loss_with_subband_mask():
    B, S, P = 2, 4, 8
    target = torch.randn(B, S, P, 2)
    # Use perfect recon for both → loss=0 regardless of mask
    loss_fn = DualOneMinusSGCS()
    mask = torch.zeros(B, S, P, dtype=torch.bool)
    mask[:, :2, :] = True
    val = loss_fn(
        {"recon": target, "recon_half": target},
        {"precoder": target, "mask": mask},
    )
    assert val.item() == pytest.approx(0.0, abs=1e-5)


def test_dual_loss_registered_via_weighted_sum():
    composite = WeightedSumLoss(
        [{"name": "dual_one_minus_sgcs", "weight": 1.0,
          "params": {"full_weight": 0.5, "half_weight": 0.5}}],
        mode="joint",
    )
    assert len(composite.term_modules) == 1
    assert composite.term_modules[0].name == "dual_one_minus_sgcs"


def test_dual_loss_produces_scalar():
    B, S, P = 3, 5, 12
    target = torch.randn(B, S, P, 2)
    loss_fn = DualOneMinusSGCS()
    val = loss_fn({"recon": target, "recon_half": target}, {"precoder": target})
    assert val.shape == ()


# ===========================================================================
# _batch_to_io output keys per masking mode
# ===========================================================================

def _make_fake_batch(B=2):
    """Minimal batch dict matching what a DataLoader would produce after collate."""
    real = torch.randn(B, MAX_S, MAX_P)
    imag = torch.randn(B, MAX_S, MAX_P)
    mask = torch.ones(B, MAX_S, MAX_P, dtype=torch.bool)
    return {"real": real, "imag": imag, "mask": mask}


def _build_ae():
    cfg = _joint_cfg()
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    return ae


def test_batch_to_io_no_mask_has_recon():
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    ae = _build_ae()
    batch = _make_fake_batch()
    pred, target = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=None)
    assert "recon" in pred
    assert "recon_half" not in pred


def test_batch_to_io_half_mode_no_recon_half_key():
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    ae = _build_ae()
    batch = _make_fake_batch()
    spec = LatentMaskSpec(mode="half", mask_ratio=0.5)
    pred, _ = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=spec)
    assert "recon" in pred
    assert "recon_half" not in pred


def test_batch_to_io_dual_mode_has_both_recons():
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    ae = _build_ae()
    batch = _make_fake_batch()
    spec = LatentMaskSpec(mode="dual", mask_ratio=0.5)
    pred, _ = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=spec)
    assert "recon" in pred, "full recon missing from dual pred_pack"
    assert "recon_half" in pred, "half recon missing from dual pred_pack"
    assert pred["recon"].shape == pred["recon_half"].shape


def test_batch_to_io_random_mode_no_recon_half_key():
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    ae = _build_ae()
    batch = _make_fake_batch()
    spec = LatentMaskSpec(mode="random", mask_ratio=0.5)
    pred, _ = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=spec)
    assert "recon" in pred
    assert "recon_half" not in pred


def test_batch_to_io_dual_recon_full_differs_from_half():
    """full and half reconstructions should differ (masking changes decoder input)."""
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    torch.manual_seed(0)
    ae = _build_ae()
    batch = _make_fake_batch(B=4)
    spec = LatentMaskSpec(mode="dual", mask_ratio=0.5)
    pred, _ = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=spec)
    assert not torch.equal(pred["recon"], pred["recon_half"]), \
        "full and half recons should differ unless latent second half is all-zero by coincidence"


def test_batch_to_io_exposes_rescaled_latent():
    """pred_pack always carries the rescaled (pre-quant, value_range) latent."""
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    ae = _build_ae()
    batch = _make_fake_batch()
    pred, _ = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=None)
    assert "rescaled_latent" in pred
    assert pred["rescaled_latent"].shape == pred["latent"].shape


def test_batch_to_io_exposes_quantizer_levels():
    """pred_pack carries q_levels / q_temperature so level-scoring loss terms
    (cross_entropy_levels) can build per-level logits without a quantizer handle.
    Checked on both the plain ae() path and a manually-built mask-mode branch."""
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    ae = _build_ae()
    batch = _make_fake_batch()
    for mask_spec in (None, LatentMaskSpec(mode="dual", mask_ratio=0.5)):
        pred, _ = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=mask_spec)
        assert "q_levels" in pred and "q_temperature" in pred
        assert torch.equal(pred["q_levels"], ae.quantizer.levels)
        assert pred["q_temperature"] == ae.quantizer.temperature


def test_batch_to_io_forwards_latent_targets():
    """latent_target / _z / _zq in the batch all reach target_pack."""
    from csi_comp.training.trainer import _batch_to_io
    from csi_comp.training.modes import get_mode_spec as _gms
    ae = _build_ae()
    batch = _make_fake_batch()
    batch["latent_target_z"] = torch.randn(2, 4)
    batch["latent_target_zq"] = torch.randn(2, 4)
    _, target = _batch_to_io(ae, batch, _gms("joint"), torch.device("cpu"), mask_spec=None)
    assert "latent_target_z" in target
    assert "latent_target_zq" in target


# ===========================================================================
# Trainer integration — each masking mode completes one epoch without error
# ===========================================================================

def test_trainer_no_mask_runs(npz_root):
    seed_everything(0)
    cfg = _joint_cfg()
    trainer = _make_trainer(cfg, npz_root, mask_spec=None)
    trainer.fit()  # must not raise


def test_trainer_half_mode_runs(npz_root):
    seed_everything(0)
    cfg = _joint_cfg({"mode": "half", "mask_ratio": 0.5})
    mask_spec = parse_latent_mask_spec(cfg["model"]["latent_mask"])
    trainer = _make_trainer(cfg, npz_root, mask_spec=mask_spec)
    trainer.fit()


def test_trainer_random_mode_runs(npz_root):
    seed_everything(0)
    cfg = _joint_cfg({"mode": "random", "mask_ratio": 0.5})
    mask_spec = parse_latent_mask_spec(cfg["model"]["latent_mask"])
    trainer = _make_trainer(cfg, npz_root, mask_spec=mask_spec)
    trainer.fit()


def test_trainer_dual_mode_runs(npz_root):
    seed_everything(0)
    cfg = _joint_cfg({"mode": "dual", "mask_ratio": 0.5})
    mask_spec = parse_latent_mask_spec(cfg["model"]["latent_mask"])
    dual_terms = [{"name": "dual_one_minus_sgcs", "weight": 1.0,
                   "params": {"full_weight": 0.5, "half_weight": 0.5}}]
    trainer = _make_trainer(cfg, npz_root, mask_spec=mask_spec, loss_terms=dual_terms)
    trainer.fit()


def test_trainer_metrics_finite_for_all_modes(npz_root):
    """All masking modes must produce finite loss and sgcs metrics."""
    configs = [
        (None, [{"name": "one_minus_sgcs", "weight": 1.0}], "no mask"),
        (LatentMaskSpec("half"), [{"name": "one_minus_sgcs", "weight": 1.0}], "half"),
        (LatentMaskSpec("random"), [{"name": "one_minus_sgcs", "weight": 1.0}], "random"),
        (LatentMaskSpec("dual"), [{"name": "dual_one_minus_sgcs", "weight": 1.0,
                                   "params": {"full_weight": 0.5, "half_weight": 0.5}}], "dual"),
    ]
    for mask_spec, terms, label in configs:
        seed_everything(0)
        cfg = _joint_cfg()
        spec = get_mode_spec("joint")
        ae, _, _ = build_model(cfg, spec)
        train_loader, val_loader = _loaders(npz_root)
        loss_fn = WeightedSumLoss(terms, mode="joint")
        opt = build_optimizer(ae, {"name": "adam", "lr": 1e-3})
        metrics_log: list[dict] = []
        from csi_comp.training import TrainerCallback

        class Recorder(TrainerCallback):
            def on_train_step_end(self, trainer, step, m):
                metrics_log.append(m)

        trainer = Trainer(
            model=ae, optimizer=opt, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            mode_spec=spec, device=torch.device("cpu"),
            epochs=1, val_every_n_epochs=1,
            best_metric={"name": "sgcs", "mode": "max"},
            mask_spec=mask_spec,
            callbacks=[Recorder()],
        )
        trainer.fit()
        for m in metrics_log:
            assert isinstance(m["loss/total"], float), f"{label}: loss/total not a float"
            assert not (m["loss/total"] != m["loss/total"]), f"{label}: loss is NaN"
            import math
            assert math.isfinite(m["loss/total"]), f"{label}: loss is not finite"


def test_trainer_full_mode_config_same_as_no_mask(npz_root):
    """model.latent_mask = {mode: full} must behave identically to no latent_mask key."""
    seed_everything(0)
    cfg = _joint_cfg({"mode": "full"})
    # parse returns None for full mode
    mask_spec = parse_latent_mask_spec(cfg["model"]["latent_mask"])
    assert mask_spec is None
    trainer = _make_trainer(cfg, npz_root, mask_spec=mask_spec)
    trainer.fit()
