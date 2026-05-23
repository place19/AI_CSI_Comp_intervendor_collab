"""Smoke tests for ConsoleCallback — every hook prints something sensible."""
from __future__ import annotations

import io
from dataclasses import dataclass

from csi_comp.training.console_logger import ConsoleCallback


@dataclass
class _FakeModeSpec:
    name: str = "joint"


class _FakeTrainer:
    """Minimal stand-in for the bits ConsoleCallback reads off `trainer`."""

    def __init__(self, epochs=2, steps_per_epoch=4):
        self.epochs = epochs
        self.epoch = 0
        self.train_loader = list(range(steps_per_epoch))
        self.mode_spec = _FakeModeSpec()
        self.device = "cpu"
        self.best_metric = {"name": "sgcs", "mode": "max"}
        self.best_value = float("-inf")


def _new(every=2):
    buf = io.StringIO()
    cb = ConsoleCallback(log_every_n_iters=every, stream=buf)
    return cb, buf


def test_train_begin_announces_run_shape():
    cb, buf = _new()
    cb.on_train_begin(_FakeTrainer(epochs=3, steps_per_epoch=10))
    out = buf.getvalue()
    assert "[train]" in out
    assert "epochs=3" in out
    assert "steps/epoch=10" in out


def test_step_prints_throttled_window_mean():
    cb, buf = _new(every=2)
    tr = _FakeTrainer()
    cb.on_train_begin(tr)
    buf.seek(0); buf.truncate()
    # Two steps in the window: means should average them.
    cb.on_train_step_end(tr, step=1, metrics={"loss/total": 1.0, "sgcs": 0.2})
    cb.on_train_step_end(tr, step=2, metrics={"loss/total": 3.0, "sgcs": 0.4})
    out = buf.getvalue()
    # step=2 triggers a flush since 2 % 2 == 0
    assert "loss/total=2.0000" in out
    assert "sgcs=0.3000" in out
    assert "step 2/4" in out


def test_step_does_not_print_below_threshold():
    cb, buf = _new(every=5)
    tr = _FakeTrainer()
    cb.on_train_begin(tr)
    buf.seek(0); buf.truncate()
    cb.on_train_step_end(tr, step=1, metrics={"loss/total": 1.0})
    cb.on_train_step_end(tr, step=2, metrics={"loss/total": 2.0})
    assert buf.getvalue() == ""  # nothing printed yet


def test_epoch_end_prints_summary_and_drains_window():
    cb, buf = _new(every=10)
    tr = _FakeTrainer()
    cb.on_train_begin(tr)
    cb.on_epoch_begin(tr, 0)
    cb.on_train_step_end(tr, step=1, metrics={"loss/total": 1.0})  # never flushed
    buf.seek(0); buf.truncate()
    cb.on_epoch_end(tr, 0, train_metrics={"loss/total": 1.5, "sgcs": 0.5})
    out = buf.getvalue()
    assert "train done" in out
    assert "loss/total=1.5000" in out
    # The unflushed step window should have been drained on epoch end.
    assert cb._counts == {}


def test_val_end_highlights_new_best():
    cb, buf = _new()
    tr = _FakeTrainer()
    cb.on_train_begin(tr)
    buf.seek(0); buf.truncate()
    cb.on_val_end(
        tr, epoch=0, val_metrics={"val/sgcs": 0.91, "best/sgcs": 0.91, "val/loss/total": 0.09},
    )
    out = buf.getvalue()
    assert "val/sgcs=0.9100" in out
    assert "new best" in out


def test_train_end_prints_total_time_and_best():
    cb, buf = _new()
    tr = _FakeTrainer()
    tr.best_value = 0.88
    cb.on_train_begin(tr)
    buf.seek(0); buf.truncate()
    cb.on_train_end(tr)
    out = buf.getvalue()
    assert "[train] done" in out
    assert "best sgcs=0.8800" in out


def test_lr_formatted_in_scientific():
    cb, buf = _new(every=1)
    tr = _FakeTrainer()
    cb.on_train_begin(tr)
    buf.seek(0); buf.truncate()
    cb.on_train_step_end(tr, step=1, metrics={"loss/total": 0.1, "lr": 1.0e-4})
    out = buf.getvalue()
    assert "lr=1.00e-04" in out
