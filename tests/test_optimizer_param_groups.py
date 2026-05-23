"""Verify AdamW (and friends) put LayerNorm + bias into a no-decay group."""
import torch
import torch.nn as nn

from csi_comp.training import build_optimizer


def _toy_model():
    return nn.Sequential(
        nn.Linear(8, 16, bias=True),
        nn.LayerNorm(16),
        nn.Linear(16, 4, bias=True),
    )


def _classify(model: nn.Module) -> tuple[list[str], list[str]]:
    """Return (decay_names, no_decay_names) according to the convention used in builders."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith(".bias"):
            no_decay.append(n)
        else:
            decay.append(n)
    return decay, no_decay


def test_adamw_excludes_ln_and_bias_from_weight_decay():
    model = _toy_model()
    opt = build_optimizer(model, {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01})
    assert len(opt.param_groups) == 2
    decay_group, no_decay_group = opt.param_groups
    # Group order matches _split_decay_no_decay: decay first
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0

    # All 2D weights (Linear.weight) should be in the decay group.
    expected_decay, expected_no_decay = _classify(model)
    decay_count = sum(p.numel() for p in decay_group["params"])
    no_decay_count = sum(p.numel() for p in no_decay_group["params"])
    expected_decay_count = sum(
        p.numel() for n, p in model.named_parameters() if n in expected_decay
    )
    expected_no_decay_count = sum(
        p.numel() for n, p in model.named_parameters() if n in expected_no_decay
    )
    assert decay_count == expected_decay_count
    assert no_decay_count == expected_no_decay_count
    # Sanity: LN's 2 affine params + the 2 Linear biases land in no-decay
    assert expected_no_decay_count == (16 + 16) + 16 + 4  # ln.weight, ln.bias, fc1.bias, fc2.bias


def test_decay_norm_bias_escape_hatch_applies_decay_to_everything():
    model = _toy_model()
    opt = build_optimizer(
        model,
        {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01, "decay_norm_bias": True},
    )
    # One uniform group containing all params with weight_decay=0.01
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.01


def test_zero_weight_decay_still_splits_but_both_groups_have_wd_zero():
    """With wd=0 the split is functionally a no-op (both groups carry wd=0),
    so we don't special-case it — keeping the code path uniform."""
    model = _toy_model()
    opt = build_optimizer(model, {"name": "adam", "lr": 1e-3, "weight_decay": 0.0})
    assert len(opt.param_groups) == 2
    assert all(g["weight_decay"] == 0.0 for g in opt.param_groups)


def test_sgd_also_supports_split():
    model = _toy_model()
    opt = build_optimizer(
        model,
        {"name": "sgd", "lr": 1e-2, "momentum": 0.9, "weight_decay": 1e-4},
    )
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 1e-4
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_decay_split_step_runs():
    """End-to-end: an optimizer step with split groups must succeed and update params."""
    model = _toy_model()
    opt = build_optimizer(model, {"name": "adamw", "lr": 1e-2, "weight_decay": 0.01})
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    opt.step()
    moved = [n for n, p in model.named_parameters() if not torch.equal(initial[n], p)]
    assert set(moved) == set(initial.keys()), f"some params never moved: {set(initial) - set(moved)}"
