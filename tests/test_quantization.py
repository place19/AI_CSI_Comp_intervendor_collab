import pytest
import torch

from csi_comp.quantization import (
    Quantizer,
    build_uniform,
    level_logits,
    snap_to_index,
    snap_to_nearest,
    soft_assign,
    soft_value,
)
from csi_comp.quantization.base import build_quantizer


def test_build_uniform_2bit():
    levels = build_uniform(bits=2, value_range=(-1.0, 1.0))
    assert torch.allclose(levels, torch.tensor([-0.75, -0.25, 0.25, 0.75]))


def test_build_uniform_3bit():
    levels = build_uniform(bits=3, value_range=(0.0, 8.0))
    # step = 1.0, midpoints at 0.5, 1.5, ..., 7.5
    assert torch.allclose(
        levels, torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    )


def test_build_uniform_invalid():
    with pytest.raises(ValueError):
        build_uniform(0, (-1.0, 1.0))
    with pytest.raises(ValueError):
        build_uniform(2, (1.0, 1.0))


def test_snap_to_nearest():
    levels = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    x = torch.tensor([0.0, -0.1, 0.4, 0.9, -1.2])
    q = snap_to_nearest(x, levels)
    assert torch.equal(q, torch.tensor([-0.25, -0.25, 0.25, 0.75, -0.75]))


def test_snap_to_index_matches_nearest():
    levels = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    x = torch.tensor([0.0, -0.1, 0.4, 0.9, -1.2])
    idx = snap_to_index(x, levels)
    assert idx.dtype == torch.long
    assert torch.equal(idx, torch.tensor([1, 1, 2, 3, 0]))
    # snap_to_nearest is exactly the gather of snap_to_index.
    assert torch.equal(levels[idx], snap_to_nearest(x, levels))


def test_quantizer_ste_forward_and_grad():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste")
    x = torch.tensor([0.4, -0.6, 0.9], requires_grad=True)
    y = q(x)
    # snapped values
    assert torch.allclose(y.detach(), torch.tensor([0.25, -0.75, 0.75]))
    # STE: backward = identity
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_quantizer_soft_differentiable():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad={"name": "soft", "temperature": 0.01})
    x = torch.tensor([0.4, -0.6, 0.9], requires_grad=True)
    y = q(x)
    # Very low temperature → close to hard snap
    assert torch.allclose(y.detach(), torch.tensor([0.25, -0.75, 0.75]), atol=1e-3)
    y.sum().backward()
    assert x.grad is not None
    # All grads finite
    assert torch.isfinite(x.grad).all()


def test_quantizer_hard_no_grad_path():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="hard")
    x = torch.tensor([0.4, -0.6, 0.9])
    y = q(x)
    assert torch.allclose(y, torch.tensor([0.25, -0.75, 0.75]))


def test_soft_quantizer_hard_snaps_in_eval():
    # In eval, a soft quantizer must hard-snap (deployed behaviour), not emit the
    # continuous softmax blend it uses during training.
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad={"name": "soft", "temperature": 1.0})
    x = torch.tensor([0.4, -0.6, 0.9])
    # train mode: continuous blend, generally != snapped values
    q.train()
    assert not torch.allclose(q(x), torch.tensor([0.25, -0.75, 0.75]), atol=1e-3)
    # eval mode: exact hard snap regardless of the soft strategy
    q.eval()
    assert torch.allclose(q(x), torch.tensor([0.25, -0.75, 0.75]))


def test_eval_snap_matches_bruteforce_many_bits():
    # The rounding-based snap must agree with a brute-force argmin nearest for a
    # large level set on random inputs (well outside the range too).
    levels = build_uniform(bits=6, value_range=(-2.0, 2.0))
    x = torch.randn(5000) * 3.0
    got = snap_to_nearest(x, levels)
    bf = levels[(x.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
    assert torch.equal(got, bf)


def test_quantizer_to_hard_swaps_grad():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste")
    assert (q.forward_name, q.backward_name) == ("hard", "identity")
    q.to_hard()
    assert (q.forward_name, q.backward_name) == ("hard", "none")
    x = torch.tensor([0.4])
    y = q(x)
    assert torch.allclose(y, torch.tensor([0.25]))


def test_quantizer_output_within_range():
    q = Quantizer(bits=4, value_range=(-2.0, 2.0), grad="ste")
    x = torch.randn(1000) * 100
    y = q(x)
    assert (y >= -2.0).all() and (y <= 2.0).all()


def test_quantizer_levels_in_state_dict():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste")
    sd = q.state_dict()
    assert "levels" in sd
    assert torch.allclose(sd["levels"], torch.tensor([-0.75, -0.25, 0.25, 0.75]))


def test_build_quantizer_from_cfg():
    q = build_quantizer(
        {"type": "uniform", "bits": 2, "value_range": (-1.0, 1.0), "grad": "ste"}
    )
    assert isinstance(q, Quantizer)
    assert q.bits == 2


def test_non_uniform_not_implemented():
    with pytest.raises(NotImplementedError):
        Quantizer(bits=2, value_range=(-1.0, 1.0), unit_spaced=False)


def test_soft_invalid_temperature():
    with pytest.raises(ValueError):
        Quantizer(bits=2, value_range=(-1.0, 1.0),
                  grad={"name": "soft", "temperature": 0.0})


# --- encoder_value_range (linear transform) tests ---

def test_encoder_value_range_identity():
    # Same range → transform is identity, output must equal the no-transform case.
    q_base = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="hard")
    q_tr = Quantizer(bits=2, value_range=(-1.0, 1.0),
                     encoder_value_range=(-1.0, 1.0), grad="hard")
    x = torch.tensor([-0.9, -0.3, 0.1, 0.8])
    assert torch.allclose(q_base(x), q_tr(x))


def test_encoder_value_range_sigmoid_to_tanh():
    # Encoder uses sigmoid (output in [0,1]), decoder expects [-1,1].
    # Linear map: x' = 2*x - 1, so x=0.0→-1.0, x=0.5→0.0, x=1.0→1.0.
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  encoder_value_range=(0.0, 1.0), grad="hard")
    # levels in [-1,1] at 2 bits: [-0.75, -0.25, 0.25, 0.75]
    # x=0.5 → x'=0.0 → snaps to -0.25
    # x=0.875 → x'=0.75 → snaps to 0.75
    x = torch.tensor([0.5, 0.875])
    y = q(x)
    assert torch.allclose(y, torch.tensor([-0.25, 0.75]))


def test_encoder_value_range_ste_gradient_scaled():
    # With encoder_value_range=(0,1) and value_range=(-1,1): alpha=2.
    # STE passes gradient as identity through snap, so dL/dx = dL/dx' * alpha = 1 * 2.
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  encoder_value_range=(0.0, 1.0), grad="ste")
    x = torch.tensor([0.5, 0.75], requires_grad=True)
    q(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))


def test_encoder_value_range_stored_on_quantizer():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  encoder_value_range=(0.0, 1.0), grad="ste")
    assert q.encoder_value_range == (0.0, 1.0)
    assert q._alpha == pytest.approx(2.0)
    assert q._beta == pytest.approx(-1.0)


def test_encoder_value_range_none_by_default():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste")
    assert q.encoder_value_range is None
    assert q._alpha is None


def test_rescale_to_value_range_applies_affine():
    # encoder_value_range=(-1,1), value_range=(-2,2): alpha=2, beta=0 → x' = 2*x.
    q = Quantizer(bits=2, value_range=(-2.0, 2.0),
                  encoder_value_range=(-1.0, 1.0), grad="ste")
    x = torch.tensor([-1.0, -0.25, 0.5, 1.0])
    assert torch.allclose(q.rescale_to_value_range(x), 2.0 * x)


def test_rescale_to_value_range_identity_when_unset():
    # No encoder_value_range → transform is the identity.
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste")
    x = torch.tensor([-0.9, 0.1, 0.8])
    assert torch.allclose(q.rescale_to_value_range(x), x)


def test_rescale_matches_forward_presnap():
    # In eval, forward() = snap_to_nearest(rescale_to_value_range(x)). Verify the
    # accessor reproduces exactly the affine forward() applies before snapping.
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  encoder_value_range=(0.0, 1.0), grad="hard")
    q.eval()
    x = torch.tensor([0.0, 0.5, 0.875, 1.0])
    rescaled = q.rescale_to_value_range(x)
    assert torch.allclose(q(x), snap_to_nearest(rescaled, q.levels))


def test_encoder_value_range_invalid():
    with pytest.raises(ValueError):
        Quantizer(bits=2, value_range=(-1.0, 1.0),
                  encoder_value_range=(1.0, 0.0))


def test_build_quantizer_with_encoder_value_range():
    q = build_quantizer({
        "type": "uniform", "bits": 2,
        "value_range": (-1.0, 1.0),
        "encoder_value_range": (0.0, 1.0),
        "grad": "ste",
    })
    assert q.encoder_value_range == (0.0, 1.0)


def test_encoder_value_range_output_within_value_range():
    # Quantizer output must always be within value_range, even when encoder input
    # comes from a different range (encoder_value_range).
    q = Quantizer(bits=4, value_range=(-1.0, 1.0),
                  encoder_value_range=(0.0, 1.0), grad="ste")
    x = torch.rand(1000)  # values in [0, 1], the encoder's range
    y = q(x)
    assert (y >= -1.0).all() and (y <= 1.0).all()


def test_encoder_value_range_to_hard_preserves_transform():
    # to_hard() must leave _alpha/_beta intact so the transform still fires.
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  encoder_value_range=(0.0, 1.0), grad="ste")
    q.to_hard()
    assert (q.forward_name, q.backward_name) == ("hard", "none")
    # x=0.875 in encoder range → x'=0.75 → snaps to 0.75
    x = torch.tensor([0.875])
    assert torch.allclose(q(x), torch.tensor([0.75]))


# --- soft_ops primitive (shared by soft forward/backward and the
#     cross_entropy_levels loss, which uses level_logits as the CE logits) ---

def test_soft_ops_shapes_and_normalisation():
    levels = build_uniform(bits=2, value_range=(-1.0, 1.0))  # 4 levels
    x = torch.randn(3, 5)
    logits = level_logits(x, levels, temperature=1.0)
    assert logits.shape == (3, 5, 4)
    probs = soft_assign(x, levels, temperature=1.0)
    assert probs.shape == (3, 5, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3, 5), atol=1e-6)


def test_soft_value_matches_weighted_sum():
    levels = build_uniform(bits=3, value_range=(-1.0, 1.0))
    x = torch.randn(7)
    expected = (soft_assign(x, levels, 0.5) * levels).sum(dim=-1)
    assert torch.allclose(soft_value(x, levels, 0.5), expected)


def test_soft_value_collapses_to_nearest_at_low_temperature():
    levels = build_uniform(bits=2, value_range=(-1.0, 1.0))
    x = torch.tensor([0.4, -0.6, 0.9])
    assert torch.allclose(
        soft_value(x, levels, temperature=1e-4), snap_to_nearest(x, levels), atol=1e-3
    )


# --- two-axis forward/backward decoupling ---

def test_legacy_presets_map_to_axes():
    assert (Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste").forward_name,
            Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste").backward_name) \
        == ("hard", "identity")
    q_soft = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="soft")
    assert (q_soft.forward_name, q_soft.backward_name) == ("soft", "soft")
    q_hard = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="hard")
    assert (q_hard.forward_name, q_hard.backward_name) == ("hard", "none")


def test_grad_mapping_equivalent_to_ste_preset():
    # Explicit two-axis spec must reproduce the `ste` preset bit-for-bit (value + grad).
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  grad={"forward": "hard", "backward": "identity"})
    assert (q.forward_name, q.backward_name) == ("hard", "identity")
    x = torch.tensor([0.4, -0.6, 0.9], requires_grad=True)
    y = q(x)
    assert torch.allclose(y.detach(), torch.tensor([0.25, -0.75, 0.75]))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_grad_mapping_name_form_still_works():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  grad={"name": "soft", "temperature": 0.5})
    assert (q.forward_name, q.backward_name) == ("soft", "soft")
    assert q.temperature == 0.5


def test_hard_forward_soft_backward():
    # forward value is the exact hard snap (no train/eval gap), gradient is the
    # smooth soft surrogate (not STE's flat identity).
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  grad={"forward": "hard", "backward": "soft", "temperature": 1.0})
    q.train()
    x = torch.tensor([0.4, -0.6, 0.9], requires_grad=True)
    y = q(x)
    # value == hard snap
    assert torch.allclose(y.detach(), torch.tensor([0.25, -0.75, 0.75]))
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # soft gradient is generally not the all-ones STE gradient
    assert not torch.allclose(x.grad, torch.ones_like(x))


def test_soft_forward_identity_backward():
    # forward value is the continuous soft blend, gradient is identity (STE).
    q = Quantizer(bits=2, value_range=(-1.0, 1.0),
                  grad={"forward": "soft", "backward": "identity", "temperature": 1.0})
    q.train()
    levels = q.levels
    x = torch.tensor([0.4, -0.6, 0.9], requires_grad=True)
    y = q(x)
    assert torch.allclose(y.detach(), soft_value(x.detach(), levels, 1.0))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))  # identity backward


def test_new_combos_hard_snap_in_eval():
    # Eval always hard-snaps regardless of the forward axis.
    for spec in ({"forward": "soft", "backward": "identity"},
                 {"forward": "hard", "backward": "soft"}):
        q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad=spec)
        q.eval()
        x = torch.tensor([0.4, -0.6, 0.9])
        assert torch.allclose(q(x), torch.tensor([0.25, -0.75, 0.75]))


def test_grad_mapping_invalid_keys():
    with pytest.raises(ValueError):
        Quantizer(bits=2, value_range=(-1.0, 1.0),
                  grad={"forward": "hard", "backward": "soft", "bogus": 1})
    with pytest.raises(ValueError):
        # neither 'name' nor both axes
        Quantizer(bits=2, value_range=(-1.0, 1.0), grad={"forward": "hard"})
    with pytest.raises(ValueError):
        Quantizer(bits=2, value_range=(-1.0, 1.0), grad="bogus")


def test_temperature_validated_for_any_combo():
    with pytest.raises(ValueError):
        Quantizer(bits=2, value_range=(-1.0, 1.0),
                  grad={"forward": "hard", "backward": "soft", "temperature": 0.0})
