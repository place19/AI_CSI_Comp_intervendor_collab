"""Latest/best checkpoint save & restore."""
from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from ..models import Autoencoder
from .compile_utils import unwrap_compiled
from .modes import ModeSpec


def _format_checkpoint_filename(prefix: str, epoch: int, metric_name: str, value: float) -> str:
    """`{prefix}_e{epoch:03d}_{metric}{value:.4f}.pt`. Non-finite values omit
    the metric so saving never fails."""
    safe_metric = re.sub(r"[^A-Za-z0-9]+", "_", metric_name).strip("_") or "metric"
    if not math.isfinite(value):
        return f"{prefix}_e{int(epoch):03d}.pt"
    return f"{prefix}_e{int(epoch):03d}_{safe_metric}{value:.4f}.pt"


def format_best_filename(epoch: int, metric_name: str, value: float) -> str:
    return _format_checkpoint_filename("best", epoch, metric_name, value)


def format_latest_filename(epoch: int, metric_name: str, value: float) -> str:
    return _format_checkpoint_filename("latest", epoch, metric_name, value)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Replace `dst` with a link to `src` atomically.
    Order: symlink (relative) → hardlink → copy."""
    src = Path(src)
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    try:
        os.symlink(src.name, tmp)   # relative symlink: robust to dir moves
    except OSError:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _state_or_none(module) -> Optional[dict[str, Any]]:
    """Return `module.state_dict()`, transparently unwrapping a `torch.compile`
    wrapper if present so saved keys never carry the `_orig_mod.` prefix."""
    if module is None:
        return None
    return unwrap_compiled(module).state_dict()


def save_checkpoint(
    path: Path,
    ae: Autoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    best_value: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "encoder": _state_or_none(ae.encoder),
        "decoder": _state_or_none(ae.decoder),
        "quantizer": _state_or_none(ae.quantizer),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_value": float(best_value) if math.isfinite(best_value) else None,
        "config": config,
    }
    torch.save(state, path)


@dataclass
class RestoredState:
    epoch: int
    global_step: int
    best_value: float
    config: dict[str, Any]


def load_checkpoint(
    path: Path,
    ae: Autoencoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> RestoredState:
    sd = torch.load(Path(path), map_location="cpu", weights_only=False)
    for name in ("encoder", "decoder", "quantizer"):
        mod = getattr(ae, name)
        if mod is not None and sd.get(name) is not None:
            # Load into the underlying module so the contract works whether the
            # live submodule is compiled or not.
            unwrap_compiled(mod).load_state_dict(sd[name], strict=strict)
    if optimizer is not None and sd.get("optimizer") is not None:
        optimizer.load_state_dict(sd["optimizer"])
    if scheduler is not None and sd.get("scheduler") is not None:
        scheduler.load_state_dict(sd["scheduler"])
    best = sd.get("best_value")
    return RestoredState(
        epoch=int(sd.get("epoch", 0)),
        global_step=int(sd.get("global_step", 0)),
        best_value=float(best) if best is not None else float("nan"),
        config=sd.get("config", {}),
    )


class CheckpointCallback:
    """Trainer callback that writes `latest.pt` every epoch and `best.pt` when
    the target validation metric improves. Checkpoints are written to
    `out_dir` only — not uploaded to MLflow (the local outputs/ folder is the
    single source of truth)."""

    def __init__(self, out_dir: Path, config: dict[str, Any] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._prev_best_descriptive: Optional[Path] = None
        self._prev_latest_descriptive: Optional[Path] = None

    def on_train_begin(self, trainer): ...
    def on_epoch_begin(self, trainer, epoch): ...
    def on_train_step_end(self, trainer, step, metrics): ...

    def on_epoch_end(self, trainer, epoch, train_metrics): ...

    def on_epoch_complete(self, trainer, epoch, train_metrics):
        # Fires after validation, best update, and epoch-unit scheduler step so
        # latest.pt captures the fully-updated state and resumes cleanly.
        metric_name = trainer.best_metric["name"]
        descriptive = self.out_dir / format_latest_filename(
            trainer.epoch, metric_name, trainer.best_value
        )
        save_checkpoint(
            descriptive, trainer.model, trainer.optimizer,
            trainer.scheduler, trainer.epoch, trainer.global_step,
            trainer.best_value, self.config,
        )
        _link_or_copy(descriptive, self.out_dir / "latest.pt")
        prev = self._prev_latest_descriptive
        if prev is not None and prev != descriptive:
            try:
                prev.unlink(missing_ok=True)
            except OSError:
                pass
        self._prev_latest_descriptive = descriptive

    def on_val_end(self, trainer, epoch, val_metrics):
        improved_key = f"best/{trainer.best_metric['name']}"
        if improved_key not in val_metrics:
            return
        metric_name = trainer.best_metric["name"]
        descriptive = self.out_dir / format_best_filename(
            trainer.epoch, metric_name, trainer.best_value
        )
        save_checkpoint(
            descriptive, trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.epoch, trainer.global_step, trainer.best_value, self.config,
        )
        # best.pt is a hardlink to the descriptive file: stable path for
        # downstream scripts (test/infer/export/resume), zero extra disk.
        best_path = self.out_dir / "best.pt"
        _link_or_copy(descriptive, best_path)
        # Retire the prior descriptive file once the new best.pt is in place.
        prev = self._prev_best_descriptive
        if prev is not None and prev != descriptive:
            try:
                prev.unlink(missing_ok=True)
            except OSError:
                pass
        self._prev_best_descriptive = descriptive

    def on_train_end(self, trainer): ...
