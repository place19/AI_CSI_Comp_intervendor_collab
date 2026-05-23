"""Always-on console progress: prints epoch/step/loss/sgcs/lr to stdout."""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Any, Dict, Optional


def _fmt_metric(k: str, v: float) -> str:
    if k == "lr" or k.startswith("lr/"):
        return f"{k}={v:.2e}"
    return f"{k}={v:.4f}"


def _fmt_metrics(metrics: Dict[str, float], keys: list[str]) -> str:
    parts = []
    for k in keys:
        if k in metrics:
            parts.append(_fmt_metric(k, metrics[k]))
    return " ".join(parts)


class ConsoleCallback:
    """Print per-step (throttled) and per-epoch progress to stdout.

    Always enabled — independent of MLflow. Step metrics are averaged over the
    same window the user picked for MLflow logging so the two views agree.

    Output stays single-line per print so it tails nicely when redirected to
    a log file or watched via `tail -f`.
    """

    def __init__(self, log_every_n_iters: int = 50, stream=None):
        self.every = int(log_every_n_iters)
        self.stream = stream if stream is not None else sys.stdout
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._train_started: Optional[float] = None
        self._epoch_started: Optional[float] = None
        # Filled in on_train_begin so we can show "step S/T" with total per epoch.
        self._steps_per_epoch: Optional[int] = None
        self._total_epochs: Optional[int] = None

    # ----- helpers -----

    def _write(self, s: str) -> None:
        self.stream.write(s + "\n")
        self.stream.flush()

    def _add_window(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k] += float(v)
            self._counts[k] += 1

    def _flush_window(self) -> Dict[str, float]:
        out = {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}
        self._sums.clear()
        self._counts.clear()
        return out

    # ----- TrainerCallback hooks -----

    def on_train_begin(self, trainer) -> None:
        self._train_started = time.time()
        self._total_epochs = int(trainer.epochs)
        try:
            self._steps_per_epoch = len(trainer.train_loader)
        except TypeError:
            self._steps_per_epoch = None
        mode = getattr(trainer.mode_spec, "name", "?")
        spe = self._steps_per_epoch if self._steps_per_epoch is not None else "?"
        self._write(
            f"[train] mode={mode} epochs={self._total_epochs} "
            f"steps/epoch={spe} device={trainer.device}"
        )

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        self._epoch_started = time.time()
        total = self._total_epochs or "?"
        self._write(f"[epoch {epoch + 1}/{total}] start")

    def on_train_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        self._add_window(metrics)
        if self.every > 0 and step % self.every == 0:
            mean = self._flush_window()
            total = self._steps_per_epoch
            # Step within the current epoch (1-indexed display).
            step_in_epoch = (
                ((step - 1) % total) + 1 if total else step
            )
            tag = (
                f"[epoch {trainer.epoch + 1}/{self._total_epochs} "
                f"step {step_in_epoch}/{total or '?'}]"
            )
            keys_order = [
                "loss/total", "sgcs",
                # also surface individual loss terms if present
                *[k for k in mean if k.startswith("loss/") and k != "loss/total"],
                "lr",
            ]
            line = _fmt_metrics(mean, keys_order)
            self._write(f"{tag} {line}")

    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float]) -> None:
        # Drain anything still in the window (e.g. last partial window of the epoch).
        if self._counts:
            self._flush_window()
        elapsed = time.time() - (self._epoch_started or time.time())
        line = _fmt_metrics(
            train_metrics,
            ["loss/total", "sgcs", "lr"],
        )
        self._write(
            f"[epoch {epoch + 1}/{self._total_epochs}] train done in {elapsed:.1f}s | {line}"
        )

    def on_val_end(self, trainer, epoch: int, val_metrics: Dict[str, float]) -> None:
        # val_metrics keys are prefixed with "val/" already; surface the best ones.
        keys = ["val/loss/total", "val/sgcs"]
        line = _fmt_metrics(val_metrics, keys)
        best_key = f"best/{trainer.best_metric['name']}"
        if best_key in val_metrics:
            line += f" ✦ new best {best_key}={val_metrics[best_key]:.4f}"
        self._write(f"[epoch {epoch + 1}/{self._total_epochs}] val | {line}")

    def on_train_end(self, trainer) -> None:
        total = time.time() - (self._train_started or time.time())
        best_name = trainer.best_metric.get("name", "?")
        self._write(
            f"[train] done in {total:.1f}s | best {best_name}={trainer.best_value:.4f}"
        )
