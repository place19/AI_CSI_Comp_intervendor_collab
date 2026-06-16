"""Device setup helpers.

CUDA GPU selection must happen before PyTorch initialises CUDA. The correct
call order in entry-point scripts is:

    1. Parse args / config.
    2. Call _common.set_cuda_visible_early() (or set_cuda_visible_from_args())
       — this sets CUDA_VISIBLE_DEVICES *before* torch is imported.
    3. Import torch and the rest of the heavy stack.
    4. Call configure_device(cfg) — by this point torch is already imported,
       so this function only returns the resolved torch.device object (and sets
       CUDA_VISIBLE_DEVICES as a late fallback for library callers that skipped
       step 2).

Note: this module itself imports torch at module load time, so importing it
already triggers step 3. Always call set_cuda_visible_early/from_args before
importing anything from csi_comp.
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
        # `gpu_index` is authoritative — assign directly so it overrides any
        # pre-existing CUDA_VISIBLE_DEVICES. NOTE: this only affects GPU selection
        # when it runs before torch initialises CUDA; the entry-point scripts set
        # it earlier via set_cuda_visible_early()/set_cuda_visible_from_args(), and
        # this assignment keeps the env consistent for direct/library callers.
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
