import math

import pytest
import torch

import torch.nn.functional as F

from csi_comp.losses import (
    CrossEntropyLevels,
    MSELatent,
    MSEQuantizedLatent,
    MSERescaledLatent,
    OneMinusSGCS,
    WeightedSumLoss,
    nmse_aligned_per_subband,
    sgcs_per_subband,
)
from csi_comp.quantization import build_uniform, level_logits


def test_sgcs_identical_inputs_is_one():
    w = torch.randn(2, 5, 8, 2)
    sgcs = sgcs_per_subband(w, w)
    assert torch.allclose(sgcs, torch.ones_like(sgcs), atol=1e-5)


def test_sgcs_orthogonal_inputs_is_zero():
    # Construct w and w_hat that are complex-orthogonal per subband.
    # w   = (1+0j) along port 0
    # w_h = (0+0j) along port 0, (1+0j) along port 1
    w = torch.zeros(1, 1, 2, 2)
    w_hat = torch.zeros(1, 1, 2, 2)
    w[0, 0, 0, 0] = 1.0
    w_hat[0, 0, 1, 0] = 1.0
    sgcs = sgcs_per_subband(w, w_hat)
    assert torch.allclose(sgcs, torch.zeros_like(sgcs), atol=1e-6)


def test_sgcs_phase_invariance():
    # SGCS uses |<w, w_hat>|^2 so a global complex phase rotation of w_hat
    # should leave SGCS unchanged.
    w = torch.randn(1, 1, 4, 2)
    # rotate w_hat by phase pi/3
    cos, sin = math.cos(math.pi / 3), math.sin(math.pi / 3)
    w_hat_r = cos * w[..., 0] - sin * w[..., 1]
    w_hat_i = sin * w[..., 0] + cos * w[..., 1]
    w_hat = torch.stack([w_hat_r, w_hat_i], dim=-1)
    s1 = sgcs_per_subband(w, w)
    s2 = sgcs_per_subband(w, w_hat)
    assert torch.allclose(s1, s2, atol=1e-5)


def test_sgcs_bad_shape_raises():
    with pytest.raises(ValueError):
        sgcs_per_subband(torch.randn(2, 5, 8, 3), torch.randn(2, 5, 8, 3))


def test_nmse_aligned_identical_is_zero():
    w = torch.randn(2, 5, 8, 2)
    nmse = nmse_aligned_per_subband(w, w)
    assert torch.allclose(nmse, torch.zeros_like(nmse), atol=1e-6)


def test_nmse_aligned_invariant_to_scale_and_phase():
    # The metric unit-norms and zeroes port-0 phase first, so a global scale
    # and phase rotation of the reconstruction must leave NMSE at ~0.
    w = torch.randn(2, 3, 6, 2)
    scale = 3.7
    cos, sin = math.cos(1.234), math.sin(1.234)
    w_hat_r = scale * (cos * w[..., 0] - sin * w[..., 1])
    w_hat_i = scale * (sin * w[..., 0] + cos * w[..., 1])
    w_hat = torch.stack([w_hat_r, w_hat_i], dim=-1)
    nmse = nmse_aligned_per_subband(w, w_hat)
    assert torch.allclose(nmse, torch.zeros_like(nmse), atol=1e-6)


def test_nmse_aligned_unrelated_is_positive():
    # Distinct random precoders should give a clearly non-zero aligned NMSE.
    torch.manual_seed(0)
    w = torch.randn(4, 2, 8, 2)
    w_hat = torch.randn(4, 2, 8, 2)
    nmse = nmse_aligned_per_subband(w, w_hat)
    assert (nmse > 1e-3).all()


def test_nmse_aligned_bad_shape_raises():
    with pytest.raises(ValueError):
        nmse_aligned_per_subband(torch.randn(2, 5, 8, 3), torch.randn(2, 5, 8, 3))


def test_one_minus_sgcs_mask_aware():
    """The padded subbands should not contribute to the mean."""
    w = torch.randn(2, 4, 8, 2)
    w_hat = w.clone()
    # Corrupt subband indices 2,3 of batch 0 — but mask them out.
    w_hat[0, 2:, :, :] = 0.0
    mask = torch.zeros(2, 4, 8, dtype=torch.bool)
    mask[0, :2, :] = True
    mask[1, :, :] = True
    loss = OneMinusSGCS()
    val = loss({"recon": w_hat}, {"precoder": w, "mask": mask}).item()
    assert val == pytest.approx(0.0, abs=1e-5)


def test_one_minus_sgcs_port_level_mask():
    """Padded ports within a valid subband should not affect SGCS."""
    # P=4 but only first 2 ports are valid (mask[:,s,2:] = False).
    # Corrupt ports 2,3 in the reconstruction — loss should still be ~0.
    w = torch.randn(2, 4, 4, 2)
    w_hat = w.clone()
    w_hat[:, :, 2:, :] = 99.0  # garbage in padded ports
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[:, :, :2] = True  # only ports 0-1 are valid
    loss = OneMinusSGCS()
    val = loss({"recon": w_hat}, {"precoder": w, "mask": mask}).item()
    assert val == pytest.approx(0.0, abs=1e-5)


def test_one_minus_sgcs_no_mask():
    w = torch.randn(2, 4, 8, 2)
    loss = OneMinusSGCS()
    val = loss({"recon": w}, {"precoder": w}).item()
    assert val == pytest.approx(0.0, abs=1e-5)


def test_mse_latent():
    pred = torch.randn(4, 16)
    tgt = pred.clone()
    loss = MSELatent()
    assert loss({"latent": pred}, {"latent_target": tgt}).item() == pytest.approx(0.0)
    assert loss({"latent": torch.zeros(4, 16)}, {"latent_target": torch.ones(4, 16)}).item() == pytest.approx(1.0)


def test_mse_rescaled_latent():
    pred = torch.randn(4, 16)
    tgt = pred.clone()
    loss = MSERescaledLatent()
    assert loss({"rescaled_latent": pred}, {"latent_target": tgt}).item() == pytest.approx(0.0)
    assert loss(
        {"rescaled_latent": torch.zeros(4, 16)}, {"latent_target": torch.ones(4, 16)}
    ).item() == pytest.approx(1.0)


def test_mse_latent_target_key_selects_teacher():
    """target_key routes each loss to a distinct teacher in the same batch."""
    z = torch.zeros(4, 16)
    zq = torch.ones(4, 16)
    target_pack = {"latent_target_z": z, "latent_target_zq": zq}
    # mse_latent ↔ Z (latent_target_z): pred==z → 0
    loss_z = MSELatent(target_key="latent_target_z")
    assert loss_z({"latent": z.clone()}, target_pack).item() == pytest.approx(0.0)
    # mse_quantized_latent ↔ Zq (latent_target_zq): pred==zq → 0
    loss_zq = MSEQuantizedLatent(target_key="latent_target_zq")
    assert loss_zq({"quantized_latent": zq.clone()}, target_pack).item() == pytest.approx(0.0)
    # mse_rescaled_latent ↔ Z: pred==zq vs z → mean((1-0)^2) = 1
    loss_r = MSERescaledLatent(target_key="latent_target_z")
    assert loss_r({"rescaled_latent": zq.clone()}, target_pack).item() == pytest.approx(1.0)


def test_mse_latent_default_target_key_is_latent_target():
    loss = MSELatent()
    assert loss.target_key == "latent_target"
    pred = torch.randn(4, 16)
    assert loss({"latent": pred}, {"latent_target": pred.clone()}).item() == pytest.approx(0.0)


def test_mse_latent_missing_target_key_raises():
    loss = MSELatent(target_key="latent_target_zq")
    with pytest.raises(KeyError, match="latent_target_zq"):
        loss({"latent": torch.zeros(4, 16)}, {"latent_target_z": torch.zeros(4, 16)})


def test_weighted_sum_loss_combines():
    pred = torch.randn(2, 4, 8, 2)
    target = pred.clone()
    composite = WeightedSumLoss(
        [{"name": "one_minus_sgcs", "weight": 2.0}],
        mode="joint",
    )
    total, logs = composite({"recon": pred}, {"precoder": target})
    assert total.item() == pytest.approx(0.0, abs=1e-5)
    assert "one_minus_sgcs" in logs


def test_weighted_sum_loss_mode_filtering():
    composite = WeightedSumLoss(
        [
            {"name": "one_minus_sgcs", "weight": 1.0, "enabled_when": "joint"},
            {"name": "mse_latent", "weight": 0.5, "enabled_when": "encoder_only"},
        ],
        mode="encoder_only",
    )
    # Only the mse_latent term is active
    assert len(composite.term_modules) == 1
    assert composite.term_modules[0].name == "mse_latent"

    # And in 'joint' mode only the sgcs term
    j = WeightedSumLoss(
        [
            {"name": "one_minus_sgcs", "weight": 1.0, "enabled_when": "joint"},
            {"name": "mse_latent", "weight": 0.5, "enabled_when": "encoder_only"},
        ],
        mode="joint",
    )
    assert len(j.term_modules) == 1
    assert j.term_modules[0].name == "one_minus_sgcs"


def test_weighted_sum_loss_no_terms_raises():
    with pytest.raises(ValueError):
        WeightedSumLoss(
            [{"name": "one_minus_sgcs", "enabled_when": "joint"}],
            mode="encoder_only",
        )


def test_weighted_sum_loss_enabled_when_list():
    composite = WeightedSumLoss(
        [{"name": "one_minus_sgcs", "enabled_when": ["joint", "encoder_only_frozen_decoder"]}],
        mode="encoder_only_frozen_decoder",
    )
    assert len(composite.term_modules) == 1


# ----- cross_entropy_levels -----

def _ce_levels():
    return build_uniform(bits=2, value_range=(-1.0, 1.0))  # [-0.75, -0.25, 0.25, 0.75]


def test_cross_entropy_levels_perfect_is_near_zero():
    levels = _ce_levels()
    # Encoder value sits exactly on each level; teacher Zq is the same level.
    x = levels.unsqueeze(0).clone()          # (1, 4)
    pred = {"rescaled_latent": x, "q_levels": levels, "q_temperature": 1e-3}
    target = {"latent_target_zq": levels.unsqueeze(0).clone()}
    loss = CrossEntropyLevels()
    assert loss(pred, target).item() == pytest.approx(0.0, abs=1e-3)


def test_cross_entropy_levels_wrong_bin_is_large():
    levels = _ce_levels()
    x = torch.full((1, 4), 0.75)             # all snap to top level (idx 3)
    target = {"latent_target_zq": torch.full((1, 4), -0.75)}  # teacher = bottom level (idx 0)
    pred = {"rescaled_latent": x, "q_levels": levels, "q_temperature": 1e-3}
    loss = CrossEntropyLevels()
    # Confidently wrong with a tiny temperature → very large CE.
    assert loss(pred, target).item() > 100.0


def test_cross_entropy_levels_default_target_key():
    assert CrossEntropyLevels().target_key == "latent_target_zq"


def test_cross_entropy_levels_missing_target_raises():
    levels = _ce_levels()
    pred = {"rescaled_latent": torch.zeros(1, 4), "q_levels": levels, "q_temperature": 1.0}
    with pytest.raises(KeyError, match="latent_target_zq"):
        CrossEntropyLevels()(pred, {"latent_target_z": torch.zeros(1, 4)})


def test_cross_entropy_levels_shape_mismatch_raises():
    levels = _ce_levels()
    pred = {"rescaled_latent": torch.zeros(1, 4), "q_levels": levels, "q_temperature": 1.0}
    with pytest.raises(ValueError, match="shape"):
        CrossEntropyLevels()(pred, {"latent_target_zq": torch.zeros(1, 3)})


def test_cross_entropy_levels_temperature_override():
    levels = _ce_levels()
    x = torch.tensor([[0.1, -0.4, 0.6, -0.9]])       # off-grid so temperature matters
    zq = torch.tensor([[0.25, -0.25, 0.75, -0.75]])
    idx = torch.tensor([2, 1, 3, 0])
    pred = {"rescaled_latent": x, "q_levels": levels, "q_temperature": 1.0}
    target = {"latent_target_zq": zq}

    # Explicit temperature is used regardless of pred_pack["q_temperature"].
    override_T = 0.25
    got = CrossEntropyLevels(temperature=override_T)(pred, target).item()
    expected = F.cross_entropy(level_logits(x, levels, override_T).reshape(-1, 4), idx).item()
    assert got == pytest.approx(expected)
    # And it differs from using the pred_pack temperature (1.0).
    pack_T = CrossEntropyLevels()(pred, target).item()
    assert abs(got - pack_T) > 1e-4


def test_cross_entropy_levels_invalid_temperature_raises():
    with pytest.raises(ValueError, match="temperature"):
        CrossEntropyLevels(temperature=0.0)
