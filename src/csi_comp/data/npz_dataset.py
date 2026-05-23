"""Single-file .npz dataset.

Layout: one .npz file holding three named arrays:
    D   int (typically int8), shape (N, S, P, 2)  — target CSI (real/imag last)
    Z   float, shape (N, latent_dim)              — pre-quant encoder output (optional)
    Zq  float, shape (N, latent_dim)              — post-quant encoder output (optional)

The whole file is loaded into RAM at construction (npz is a zip-of-npys; mmap is
not meaningful here). `D` is kept in its on-disk integer dtype to save memory
and cast to float32 ×scale per-sample in __getitem__.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..registry import register

_DEFAULT_SCALE = 1.0 / 128.0
_DEFAULT_TARGET_OFFSET = 1.0 / 256.0


@register("dataset", "npz")
class NpzDataset(Dataset):
    """Args:
        path:           path to the .npz file.
        scale:          multiplicative scale applied after int → float32 cast on `D`.
        target_offset:  additive offset applied on top of `scale` to produce the
                        reconstruction target (`real_target`/`imag_target`). Models
                        the bin-midpoint dequantization step required by 3GPP/HW
                        (int → float quantizes with `floor`, so the original
                        continuous value sits at `int*scale + scale/2`). Encoder
                        input keeps the raw `int*scale` convention so the model
                        sees what HW actually produces.
        latent_key:     which array becomes `latent_target` in each sample. One of
                        None (default, do not emit `latent_target`), "Zq"
                        (post-quant), or "Z" (pre-quant). Set to "Zq" or "Z"
                        only when `latent_target` is actually consumed (e.g.
                        `decoder_only` mode or `mse_latent` loss).
        also_expose_z:  if True and `Z` is present, every sample also carries
                        `latent_target_z` (float32) — for losses that want the
                        un-quantized teacher latent alongside whatever
                        `latent_key` selected.
    """

    def __init__(
        self,
        path: str | Path,
        scale: float = _DEFAULT_SCALE,
        target_offset: float = _DEFAULT_TARGET_OFFSET,
        latent_key: Optional[str] = None,
        also_expose_z: bool = True,
    ):
        self.path = Path(path)
        with np.load(self.path) as f:
            files = set(f.files)
            D = np.asarray(f["D"])
            Z = np.asarray(f["Z"]) if "Z" in files else None
            Zq = np.asarray(f["Zq"]) if "Zq" in files else None

        if D.ndim != 4 or D.shape[-1] != 2:
            raise ValueError(f"D must have shape (N, S, P, 2); got {D.shape}")
        self.D = torch.from_numpy(np.ascontiguousarray(D))
        self.S, self.P = int(D.shape[1]), int(D.shape[2])
        self.scale = float(scale)
        self.target_offset = float(target_offset)

        latents = {"Zq": Zq, "Z": Z, None: None}
        if latent_key not in latents:
            raise ValueError(
                f"latent_key must be 'Zq', 'Z', or None; got {latent_key!r}"
            )
        chosen = latents[latent_key]
        if latent_key is not None and chosen is None:
            raise ValueError(
                f"latent_key={latent_key!r} but array {latent_key!r} is absent "
                f"from {self.path}; available arrays: {sorted(files)}"
            )
        for arr_name, arr in (("Z", Z), ("Zq", Zq)):
            if arr is not None and arr.shape[0] != D.shape[0]:
                raise ValueError(
                    f"{arr_name}.shape[0]={arr.shape[0]} != D.shape[0]={D.shape[0]} "
                    f"in {self.path}"
                )
        self.latent_target = (
            torch.from_numpy(np.ascontiguousarray(chosen).astype(np.float32, copy=False))
            if chosen is not None else None
        )
        self.latent_z = (
            torch.from_numpy(np.ascontiguousarray(Z).astype(np.float32, copy=False))
            if (also_expose_z and Z is not None) else None
        )

    def __len__(self) -> int:
        return int(self.D.shape[0])

    def __getitem__(self, i: int) -> dict[str, Any]:
        a = self.D[i].to(torch.float32) * self.scale  # (S, P, 2)
        t = a + self.target_offset                    # bin-midpoint target
        out: dict[str, Any] = {
            "real": a[..., 0],
            "imag": a[..., 1],
            "real_target": t[..., 0],
            "imag_target": t[..., 1],
            "true_shape": (self.S, self.P),
        }
        if self.latent_target is not None:
            out["latent_target"] = self.latent_target[i]
        if self.latent_z is not None:
            out["latent_target_z"] = self.latent_z[i]
        return out
