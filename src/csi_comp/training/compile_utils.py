"""Helpers for `torch.compile`-aware module wrapping.

`torch.compile(module)` returns an `OptimizedModule` whose `state_dict()` prefixes
every key with `_orig_mod.`. That breaks `state_dict` portability between
compiled and uncompiled runs.

Contract followed by this project:
- We always *build* from the yaml cfg first, then optionally compile.
- We always *save* via `unwrap_compiled(m).state_dict()` so keys never carry
  the `_orig_mod.` prefix.
- We always *load* into `unwrap_compiled(live_module)` so loading works whether
  the live model is compiled or not.

This keeps checkpoints portable: a checkpoint produced by a compiled run loads
cleanly into an uncompiled inference build, and vice-versa.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def unwrap_compiled(module: Optional[nn.Module]) -> Optional[nn.Module]:
    """Return the underlying module behind a `torch.compile` wrapper.

    Safe to call on uncompiled modules (returns the module unchanged) and on
    `None` (returns `None`).
    """
    if module is None:
        return None
    return getattr(module, "_orig_mod", module)


def maybe_compile(
    module: Optional[nn.Module],
    compile_cfg: Optional[dict[str, Any]],
) -> Optional[nn.Module]:
    """Compile `module` according to a yaml `training.compile` block.

    Returns the module unchanged when the cfg is missing, disabled, or the
    input is `None`. Forwards `mode` / `dynamic` / `fullgraph` when present.
    """
    if module is None:
        return None
    if not compile_cfg or not compile_cfg.get("enabled", False):
        return module
    if not hasattr(torch, "compile"):
        return module
    kw: dict[str, Any] = {}
    if "mode" in compile_cfg:
        kw["mode"] = compile_cfg["mode"]
    if "dynamic" in compile_cfg:
        kw["dynamic"] = bool(compile_cfg["dynamic"])
    if "fullgraph" in compile_cfg:
        kw["fullgraph"] = bool(compile_cfg["fullgraph"])
    return torch.compile(module, **kw)


def compile_autoencoder_inplace(ae: nn.Module, compile_cfg: Optional[dict[str, Any]]) -> None:
    """Apply `maybe_compile` to `ae.encoder` and `ae.decoder` in-place.

    Quantizer is intentionally left uncompiled: its STE backward uses the
    `x + (q - x).detach()` trick which interacts poorly with `torch.compile`'s
    graph capture, and the quantizer is cheap anyway.
    """
    if not compile_cfg or not compile_cfg.get("enabled", False):
        return
    if getattr(ae, "encoder", None) is not None:
        ae.encoder = maybe_compile(ae.encoder, compile_cfg)
    if getattr(ae, "decoder", None) is not None:
        ae.decoder = maybe_compile(ae.decoder, compile_cfg)
