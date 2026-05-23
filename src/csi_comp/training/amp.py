"""AMP (mixed-precision) configuration + autocast helpers.

Design notes
------------
- The model forward runs under autocast when `AmpSpec.enabled` is True.
- Loss computation runs *outside* autocast — a fp32 island. This avoids
  numerical blow-ups in backprop that a previous attempt observed.
- The MHA softmax is wrapped in `with torch.amp.autocast(..., enabled=False)`
  inside `transformer_block.py` for the same reason.
- `use_scaler` is True only for cuda+fp16 (bf16 has fp32 dynamic range so no
  GradScaler is needed; mps doesn't support GradScaler).

Device defaults when `amp.dtype` is not specified in yaml:
- cuda → bf16
- mps  → fp16
- cpu  → bf16 (autocast on cpu is unusual but supported; we honour `enabled`).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Optional

import torch


def _parse_dtype(s: Optional[str], default: torch.dtype) -> torch.dtype:
    if s is None:
        return default
    s = str(s).lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unknown amp dtype {s!r}; use bf16 | fp16 | fp32")


@dataclass(frozen=True)
class AmpSpec:
    enabled: bool
    device_type: str       # "cuda" | "mps" | "cpu"
    dtype: torch.dtype     # ignored when enabled=False
    use_scaler: bool       # True only on cuda + fp16


def resolve_amp_cfg(amp_cfg: Optional[dict[str, Any]], device: torch.device) -> AmpSpec:
    """Resolve a yaml `training.amp` block against the active device."""
    dev_type = device.type
    if not amp_cfg or not amp_cfg.get("enabled", False):
        return AmpSpec(enabled=False, device_type=dev_type, dtype=torch.float32, use_scaler=False)
    if dev_type == "cuda":
        dtype = _parse_dtype(amp_cfg.get("dtype"), default=torch.bfloat16)
    elif dev_type == "mps":
        dtype = _parse_dtype(amp_cfg.get("dtype"), default=torch.float16)
    else:  # cpu
        dtype = _parse_dtype(amp_cfg.get("dtype"), default=torch.bfloat16)
    use_scaler = (dev_type == "cuda" and dtype == torch.float16)
    return AmpSpec(enabled=True, device_type=dev_type, dtype=dtype, use_scaler=use_scaler)


def autocast_ctx(spec: Optional[AmpSpec]):
    """Return an autocast context manager, or a null context if AMP is off."""
    if spec is None or not spec.enabled:
        return contextlib.nullcontext()
    # MPS autocast lives under torch.amp.autocast (not torch.cuda.amp.autocast).
    return torch.amp.autocast(device_type=spec.device_type, dtype=spec.dtype)


def build_grad_scaler(spec: Optional[AmpSpec]) -> Optional[torch.amp.GradScaler]:
    """Construct a GradScaler iff fp16+cuda; otherwise None."""
    if spec is None or not spec.use_scaler:
        return None
    return torch.amp.GradScaler(device=spec.device_type)
