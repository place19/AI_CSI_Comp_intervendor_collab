import pytest
import torch

from csi_comp.quantization import Quantizer, build_uniform, snap_to_nearest
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


def test_quantizer_to_hard_swaps_grad():
    q = Quantizer(bits=2, value_range=(-1.0, 1.0), grad="ste")
    assert q.grad_name == "ste"
    q.to_hard()
    assert q.grad_name == "hard"
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
    assert q.grad_name == "hard"
    # x=0.875 in encoder range → x'=0.75 → snaps to 0.75
    x = torch.tensor([0.875])
    assert torch.allclose(q(x), torch.tensor([0.75]))
