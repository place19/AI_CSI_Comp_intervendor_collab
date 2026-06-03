"""Batch collation: per-sample (S, P) → padded (B, max_S, max_P) + bool mask."""
from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import torch


def pad_and_collate(
    batch: Sequence[dict[str, Any]],
    max_subband: int,
    max_port: int,
) -> dict[str, Any]:
    """Stack a list of variable-shape samples into a padded batch.

    Output keys:
        real:           (B, max_subband, max_port) float32, zeros outside valid region — encoder input
        imag:           (B, max_subband, max_port) float32
        real_target:    (B, max_subband, max_port) float32 — reconstruction target (only if every sample carries it)
        imag_target:    (B, max_subband, max_port) float32
        mask:           (B, max_subband, max_port) bool, True on valid cells
        true_shapes:    list[(s, p)]
        latent_target:    (B, ...) — only if every sample carries one
        latent_target_z:  (B, ...) — only if every sample carries one (pre-quant teacher Z)
        latent_target_zq: (B, ...) — only if every sample carries one (post-quant teacher Zq)
    """
    B = len(batch)
    if B == 0:
        raise ValueError("empty batch")
    real = torch.zeros(B, max_subband, max_port, dtype=torch.float32)
    imag = torch.zeros(B, max_subband, max_port, dtype=torch.float32)
    real_tgt = torch.zeros(B, max_subband, max_port, dtype=torch.float32)
    imag_tgt = torch.zeros(B, max_subband, max_port, dtype=torch.float32)
    mask = torch.zeros(B, max_subband, max_port, dtype=torch.bool)
    true_shapes = []
    n_target = 0
    # Any latent target the dataset chose to emit (all-or-none per key). Kept generic
    # so new teacher latents (e.g. latent_target_zq) need no extra plumbing here.
    latent_keys = ("latent_target", "latent_target_z", "latent_target_zq")
    latents: dict[str, list[torch.Tensor]] = {k: [] for k in latent_keys}

    for i, s in enumerate(batch):
        S, P = s["real"].shape
        if S > max_subband or P > max_port:
            raise ValueError(
                f"sample exceeds bounds: ({S}, {P}) > ({max_subband}, {max_port})"
            )
        real[i, :S, :P] = s["real"]
        imag[i, :S, :P] = s["imag"]
        mask[i, :S, :P] = True
        true_shapes.append((int(S), int(P)))
        has_rt = "real_target" in s
        has_it = "imag_target" in s
        if has_rt != has_it:
            raise ValueError("real_target/imag_target must both be present or both absent")
        if has_rt:
            real_tgt[i, :S, :P] = s["real_target"]
            imag_tgt[i, :S, :P] = s["imag_target"]
            n_target += 1
        for k in latent_keys:
            if k in s:
                latents[k].append(s[k])

    out: dict[str, Any] = {
        "real": real,
        "imag": imag,
        "mask": mask,
        "true_shapes": true_shapes,
    }
    if n_target:
        if n_target != B:
            raise ValueError("real_target/imag_target must be present for all samples or none")
        out["real_target"] = real_tgt
        out["imag_target"] = imag_tgt
    for k in latent_keys:
        vals = latents[k]
        if vals:
            if len(vals) != B:
                raise ValueError(f"{k} must be present for all samples or none")
            out[k] = torch.stack(vals)
    return out


def make_collate_fn(max_subband: int, max_port: int):
    """Return a picklable collate_fn bound to these padding bounds."""
    return partial(pad_and_collate, max_subband=max_subband, max_port=max_port)
