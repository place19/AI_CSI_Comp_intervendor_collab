import pytest
import torch

from csi_comp.registry import REGISTRY
from csi_comp.training import build_optimizer, build_scheduler
# Importing the schedulers package triggers @register for all factories.
from csi_comp.training import schedulers  # noqa: F401


def _dummy_optimizer(lr: float = 1e-2):
    model = torch.nn.Linear(4, 4)
    return torch.optim.SGD(model.parameters(), lr=lr)


def test_registry_contains_all_schedulers():
    for name in ("cosine", "step", "none", "warmup_cosine"):
        assert name in REGISTRY["scheduler"], f"scheduler {name!r} not registered"


def test_build_cosine_via_registry_tagged_epoch():
    opt = _dummy_optimizer()
    sched = build_scheduler(opt, {"name": "cosine", "T_max": 5})
    assert sched is not None
    assert getattr(sched, "step_unit", None) == "epoch"


def test_build_none_returns_none():
    opt = _dummy_optimizer()
    assert build_scheduler(opt, {"name": "none"}) is None


def test_warmup_cosine_lr_profile():
    """LambdaLR's __init__ calls .step() once internally, so immediately after
    construction the lr is already lambda(0). After N user .step() calls, lr is
    lambda(N)."""
    base_lr = 1.0
    opt = _dummy_optimizer(base_lr)
    sched = build_scheduler(
        opt,
        {"name": "warmup_cosine", "warmup_steps": 10, "total_steps": 100, "min_lr": 0.01},
    )
    assert sched.step_unit == "iter"

    # After construction: lambda(0) applied → factor = 1/10
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * 1 / 10, rel=1e-6)

    # 9 user steps → lambda(9) → factor = 10/10 = 1 (peak of warmup)
    for _ in range(9):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-6)


def test_warmup_cosine_mid_decay():
    """Halfway through the cosine decay window, factor ≈ (1 + floor) / 2."""
    base_lr = 1.0
    opt = _dummy_optimizer(base_lr)
    sched = build_scheduler(
        opt,
        {"name": "warmup_cosine", "warmup_steps": 10, "total_steps": 110, "min_lr": 0.01},
    )
    # After construction we're at lambda(0); we need lambda(60) so call .step() 60 times.
    for _ in range(60):
        sched.step()
    # progress = (60 - 10) / (110 - 10) = 0.5 → cos(pi*0.5) = 0
    # factor = floor + (1 - floor) * 0.5
    expected = 0.01 + (base_lr - 0.01) * 0.5
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-3)


def test_warmup_cosine_floor_at_end():
    base_lr = 1.0
    opt = _dummy_optimizer(base_lr)
    sched = build_scheduler(
        opt,
        {"name": "warmup_cosine", "warmup_steps": 2, "total_steps": 10, "min_lr": 0.1},
    )
    for _ in range(10):
        sched.step()
    # at step >= total_steps the factor clamps to min_lr / base_lr
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1, rel=1e-6)


def test_total_steps_auto_filled():
    opt = _dummy_optimizer()
    sched = build_scheduler(
        opt,
        {"name": "warmup_cosine", "warmup_steps": 5},
        epochs=2,
        steps_per_epoch=10,
    )
    assert sched.total_steps == 20


def test_total_steps_explicit_wins():
    opt = _dummy_optimizer()
    sched = build_scheduler(
        opt,
        {"name": "warmup_cosine", "warmup_steps": 5, "total_steps": 999},
        epochs=2,
        steps_per_epoch=10,
    )
    assert sched.total_steps == 999


def test_cosine_does_not_receive_total_steps():
    """build_scheduler must drop unknown kwargs before forwarding to a factory
    that doesn't accept them (e.g. cosine has no `total_steps` arg)."""
    opt = _dummy_optimizer()
    sched = build_scheduler(
        opt,
        {"name": "cosine", "T_max": 5},
        epochs=2,
        steps_per_epoch=10,
    )
    assert sched is not None  # would have raised TypeError if total_steps leaked


def test_warmup_cosine_state_dict_roundtrip():
    base_lr = 1.0
    opt = _dummy_optimizer(base_lr)
    sched = build_scheduler(
        opt,
        {"name": "warmup_cosine", "warmup_steps": 4, "total_steps": 20, "min_lr": 0.0},
    )
    for _ in range(7):
        sched.step()
    snapshot_lr = opt.param_groups[0]["lr"]
    state = sched.state_dict()

    # Rebuild and restore
    opt2 = _dummy_optimizer(base_lr)
    sched2 = build_scheduler(
        opt2,
        {"name": "warmup_cosine", "warmup_steps": 4, "total_steps": 20, "min_lr": 0.0},
    )
    sched2.load_state_dict(state)
    # Trigger a step to apply the lambda from the restored counter
    sched2.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(opt2.param_groups[0]["lr"], rel=1e-6)


def test_warmup_cosine_invalid_params():
    opt = _dummy_optimizer()
    with pytest.raises(ValueError):
        build_scheduler(opt, {"name": "warmup_cosine", "warmup_steps": -1, "total_steps": 10})
    with pytest.raises(ValueError):
        build_scheduler(opt, {"name": "warmup_cosine", "warmup_steps": 5, "total_steps": 0})
    with pytest.raises(ValueError):
        build_scheduler(opt, {"name": "warmup_cosine", "warmup_steps": 100, "total_steps": 50})


def test_trainer_steps_iter_scheduler_each_iteration(npz_root, tmp_path):
    """Confirm an iter-unit scheduler is stepped batches*epochs times."""
    from torch.utils.data import DataLoader
    from csi_comp.data import NpzDataset, make_collate_fn
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.training import (
        Trainer, build_model, get_mode_spec, seed_everything,
    )

    MAX_S, MAX_P = 8, 12
    seed_everything(0)
    cfg = {
        "experiment": {"device": "cpu"},
        "data": {"layout": "cnn", "max_subband": MAX_S, "max_port": MAX_P},
        "model": {
            "encoder": {
                "blocks": [
                    {"name": "cnn_block", "channels": 4, "kernel": 3},
                    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
                    {"name": "activation", "activation": "tanh"},
                ],
            },
            "decoder": {"blocks": [
                {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                {"name": "reshape_head", "max_subband": MAX_S, "max_port": MAX_P},
            ]},
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": "joint"},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    train_loader = DataLoader(NpzDataset(npz_root / "train.npz"),
                              batch_size=4, collate_fn=make_collate_fn(MAX_S, MAX_P))
    val_loader = DataLoader(NpzDataset(npz_root / "val.npz"),
                            batch_size=4, collate_fn=make_collate_fn(MAX_S, MAX_P))
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode="joint")
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-3})
    sched = build_scheduler(
        opt, {"name": "warmup_cosine", "warmup_steps": 2, "min_lr": 0.0},
        epochs=2, steps_per_epoch=len(train_loader),
    )
    trainer = Trainer(
        model=ae, optimizer=opt, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        mode_spec=spec, device=torch.device("cpu"),
        epochs=2, val_every_n_epochs=1, scheduler=sched,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    trainer.fit()
    # LambdaLR.last_epoch reflects the number of .step() calls
    assert sched.last_epoch == 2 * len(train_loader)


def test_trainer_lr_metric_logged(npz_root):
    """on_train_step_end metrics should include a single `lr` value."""
    from torch.utils.data import DataLoader
    from csi_comp.data import NpzDataset, make_collate_fn
    from csi_comp.losses.composite import WeightedSumLoss
    from csi_comp.training import (
        Trainer, TrainerCallback, build_model, get_mode_spec, seed_everything,
    )

    MAX_S, MAX_P = 8, 12
    seed_everything(0)
    cfg = {
        "experiment": {"device": "cpu"},
        "data": {"layout": "cnn", "max_subband": MAX_S, "max_port": MAX_P},
        "model": {
            "encoder": {
                "blocks": [
                    {"name": "cnn_block", "channels": 4, "kernel": 3},
                    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
                    {"name": "activation", "activation": "tanh"},
                ],
            },
            "decoder": {"blocks": [
                {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                {"name": "reshape_head", "max_subband": MAX_S, "max_port": MAX_P},
            ]},
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": "joint"},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    train_loader = DataLoader(NpzDataset(npz_root / "train.npz"),
                              batch_size=4, collate_fn=make_collate_fn(MAX_S, MAX_P))
    val_loader = DataLoader(NpzDataset(npz_root / "val.npz"),
                            batch_size=4, collate_fn=make_collate_fn(MAX_S, MAX_P))
    loss_fn = WeightedSumLoss(cfg["loss"]["terms"], mode="joint")
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-3})

    captured: list[dict] = []
    class Recorder(TrainerCallback):
        def on_train_step_end(self, trainer, step, metrics):
            captured.append(metrics)

    trainer = Trainer(
        model=ae, optimizer=opt, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        mode_spec=spec, device=torch.device("cpu"),
        epochs=1, val_every_n_epochs=1,
        callbacks=[Recorder()],
        best_metric={"name": "sgcs", "mode": "max"},
    )
    trainer.fit()
    assert captured, "no train steps recorded"
    assert "lr" in captured[0]
    assert "lr/group0" not in captured[0]
