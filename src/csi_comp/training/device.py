"""Device setup. CUDA selection uses CUDA_VISIBLE_DEVICES, set before importing torch.

The intended use is for entry-point scripts:
    1. Parse config.
    2. configure_device(cfg) — sets env var.
    3. THEN import torch (the heavy stuff).

For library usage where torch is already imported, calling configure_device only
adjusts the env var (won't affect torch's already-cached device list) and returns
the resolved torch.device.
"""
from __future__ import annotations

import os
from typing import Any

import torch


def configure_device(experiment_cfg: dict[str, Any]) -> torch.device:
    device = str(experiment_cfg.get("device", "cpu")).lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        gpu_index = experiment_cfg.get("gpu_index", 0)
        # Only set the env var if it's not already set by the launcher.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
        # We always address cuda:0 inside the process; CUDA_VISIBLE_DEVICES handles selection.
        return torch.device("cuda:0")
    if device == "mps":
        # Apple Silicon GPU via the Metal Performance Shaders backend.
        if not torch.backends.mps.is_available():
            built = torch.backends.mps.is_built()
            raise RuntimeError(
                "device=mps requested but MPS is not available "
                f"(torch built with MPS={built}). Falling back is the caller's choice — "
                "edit the config to use cpu."
            )
        return torch.device("mps")
    raise ValueError(f"unknown device: {device!r}")
