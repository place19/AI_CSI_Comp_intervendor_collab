"""Descriptive best filename + stable checkpoint link behavior."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import torch

from csi_comp.training import build_model, build_optimizer, get_mode_spec
from csi_comp.training.checkpoint import (
    CheckpointCallback,
    format_best_filename,
)

MAX_S, MAX_P = 8, 12


def _cfg():
    return {
        "experiment": {"device": "cpu", "seed": 0},
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
                ]
            },
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": "joint"},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }


@dataclass
class _FakeTrainer:
    model: Any
    optimizer: Any
    scheduler: Optional[Any]
    epoch: int
    global_step: int
    best_value: float
    best_metric: dict


def _trainer_fixture():
    cfg = _cfg()
    spec = get_mode_spec("joint")
    ae, _, _ = build_model(cfg, spec)
    opt = build_optimizer(ae, {"name": "adam", "lr": 1e-3})
    return cfg, ae, opt


def test_format_basic():
    assert format_best_filename(23, "sgcs", 0.8421) == "best_e023_sgcs0.8421.pt"


def test_format_sanitizes_metric_name():
    assert format_best_filename(7, "val/loss/total", 0.12345) == "best_e007_val_loss_total0.1235.pt"


def test_format_nan_falls_back():
    assert format_best_filename(5, "sgcs", float("nan")) == "best_e005.pt"
    assert format_best_filename(5, "sgcs", math.inf) == "best_e005.pt"


def test_best_pt_links_to_descriptive(tmp_path):
    cfg, ae, opt = _trainer_fixture()
    cb = CheckpointCallback(out_dir=tmp_path, config=cfg)
    trainer = _FakeTrainer(
        model=ae, optimizer=opt, scheduler=None,
        epoch=12, global_step=200, best_value=0.7531,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    cb.on_val_end(trainer, epoch=12, val_metrics={"best/sgcs": 0.7531})

    descriptive = tmp_path / "best_e012_sgcs0.7531.pt"
    best = tmp_path / "best.pt"
    assert descriptive.exists()
    assert best.exists()
    # Hardlink → same inode (fallback to copy would also pass the bytes check).
    if os.stat(descriptive).st_nlink >= 2:
        assert os.path.samefile(descriptive, best)
    assert best.read_bytes() == descriptive.read_bytes()


def test_improvement_replaces_previous_descriptive(tmp_path):
    cfg, ae, opt = _trainer_fixture()
    cb = CheckpointCallback(out_dir=tmp_path, config=cfg)

    t1 = _FakeTrainer(
        model=ae, optimizer=opt, scheduler=None,
        epoch=3, global_step=50, best_value=0.6000,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    cb.on_val_end(t1, 3, {"best/sgcs": 0.6000})
    first = tmp_path / "best_e003_sgcs0.6000.pt"
    assert first.exists()

    t2 = _FakeTrainer(
        model=ae, optimizer=opt, scheduler=None,
        epoch=8, global_step=120, best_value=0.7200,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    cb.on_val_end(t2, 8, {"best/sgcs": 0.7200})
    second = tmp_path / "best_e008_sgcs0.7200.pt"
    assert second.exists(), "new descriptive file must exist"
    assert not first.exists(), "previous descriptive file must be cleaned up"
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "best.pt").read_bytes() == second.read_bytes()


def test_no_improvement_means_no_save(tmp_path):
    cfg, ae, opt = _trainer_fixture()
    cb = CheckpointCallback(out_dir=tmp_path, config=cfg)
    trainer = _FakeTrainer(
        model=ae, optimizer=opt, scheduler=None,
        epoch=4, global_step=99, best_value=0.5,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    cb.on_val_end(trainer, 4, {"val/sgcs": 0.4})  # no `best/sgcs`
    assert not (tmp_path / "best.pt").exists()
    assert list(tmp_path.glob("best_e*.pt")) == []


def test_load_checkpoint_works_through_stable_link(tmp_path):
    cfg, ae, opt = _trainer_fixture()
    cb = CheckpointCallback(out_dir=tmp_path, config=cfg)
    trainer = _FakeTrainer(
        model=ae, optimizer=opt, scheduler=None,
        epoch=2, global_step=10, best_value=0.4321,
        best_metric={"name": "sgcs", "mode": "max"},
    )
    cb.on_val_end(trainer, 2, {"best/sgcs": 0.4321})

    from csi_comp.training.checkpoint import load_checkpoint
    ae2, _, _ = build_model(_cfg(), get_mode_spec("joint"))
    restored = load_checkpoint(tmp_path / "best.pt", ae2)
    assert restored.epoch == 2
    assert restored.best_value == pytest.approx(0.4321)
