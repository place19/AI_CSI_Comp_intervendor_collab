"""Shared bits of the CLI entry-point scripts: argparse, device handling,
config loading order.

Importing this module also makes `csi_comp` importable on a fresh checkout
that hasn't run `pip install -e .` — it prepends `<repo>/src` to `sys.path`.
Harmless when the package is already installed (editable install points at
the same `src/csi_comp/`, so both routes resolve to the same module)."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import os
import time
from typing import Iterable

import yaml


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", type=Path, help="path to YAML config")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key (repeatable)",
    )
    ap.add_argument("--device", choices=("cpu", "cuda"), default=None)
    ap.add_argument("--gpu-index", type=int, default=None)
    # Output / MLflow naming.
    ap.add_argument(
        "--no-timestamp",
        action="store_true",
        help=(
            "Do NOT append a start-time stamp to the output folder and MLflow run "
            "name. Default is to append _YYYYMMDD_HHMMSS so each invocation gets "
            "its own folder and previous checkpoints are not overwritten."
        ),
    )


def resolve_run_name(
    base: str,
    *,
    timestamp: bool,
    suffix: str = "",
    now: float | None = None,
) -> str:
    """Compose the run name shared by `outputs/<...>/` and the MLflow run.

    Args:
        base:       experiment name from the config (or a fallback).
        timestamp:  if True, append `_YYYYMMDD_HHMMSS` of the current local time.
        suffix:     extra suffix appended *after* the timestamp (e.g. "_finetuned").
        now:        seconds-since-epoch override for tests; defaults to time.time().

    The returned string is used as both the folder under `--out-root` and the
    MLflow `run_name`, so they always match.
    """
    out = base
    if timestamp:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
        out = f"{out}_{ts}"
    if suffix:
        out = f"{out}{suffix}"
    return out


def apply_cli_device(experiment_cfg: dict, args: argparse.Namespace) -> None:
    if args.device is not None:
        experiment_cfg["device"] = args.device
    if args.gpu_index is not None:
        experiment_cfg["gpu_index"] = int(args.gpu_index)


def set_cuda_visible_early(experiment_cfg: dict) -> None:
    """Must be called BEFORE torch is imported. Sets CUDA_VISIBLE_DEVICES so
    `cuda:0` inside the process maps to the requested GPU."""
    if str(experiment_cfg.get("device", "cpu")).lower() == "cuda":
        gpu_index = experiment_cfg.get("gpu_index", 0)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(int(gpu_index)))


def set_cuda_visible_from_args(args: argparse.Namespace) -> None:
    """Set CUDA_VISIBLE_DEVICES from CLI --device/--gpu-index BEFORE torch is
    imported. Call this as early as possible in main(), before any torch import.

    This covers the case where test.py/infer.py must torch.load the checkpoint
    to read the embedded config (so set_cuda_visible_early cannot fire first).
    GPU index from the embedded config is NOT applied here — that requires
    torch.load and is an accepted limitation of the embedded-config approach."""
    if getattr(args, "device", None) == "cuda" and getattr(args, "gpu_index", None) is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(int(args.gpu_index)))


def load_resolved_config(path: Path, overrides: Iterable[str]) -> dict:
    """Lazy import to defer torch loading."""
    from csi_comp.config import load_and_resolve
    return load_and_resolve(path, overrides=overrides)


def dump_yaml(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return path
