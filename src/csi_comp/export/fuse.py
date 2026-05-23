"""Inference-time module fusion driven by each block's `fusion_pairs`.

A `Block` may declare `self.fusion_pairs: list[(absorber, absorbee)]` where
the absorber's weights/bias absorb the absorbee (typically a BatchNorm) at
inference time. The profiler already uses this metadata to count FLOPs as if
the fold had happened; this module performs the actual fold so the exported
ONNX graph matches.

Supported pairs:
- (nn.Conv2d, nn.BatchNorm2d)  — `torch.nn.utils.fusion.fuse_conv_bn_eval`
- (nn.Linear, nn.BatchNorm1d)  — custom fold (W' = diag(γ/σ̂) @ W, b' = ...)

In-place on the model the caller passes in. The caller is expected to deep-copy
before calling if they want to preserve the original (training-time) model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fuse_linear_bn_eval(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """Fold a 1-D BatchNorm into the preceding Linear (eval-mode arithmetic).

    Returns a fresh `nn.Linear` with the folded weights/bias.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear)}")
    if not isinstance(bn, nn.BatchNorm1d):
        raise TypeError(f"expected nn.BatchNorm1d, got {type(bn)}")
    fused = nn.Linear(linear.in_features, linear.out_features, bias=True)
    w = linear.weight.detach().clone()
    b = (
        linear.bias.detach().clone()
        if linear.bias is not None
        else torch.zeros(linear.out_features, dtype=w.dtype, device=w.device)
    )
    rm = bn.running_mean.detach()
    rv = bn.running_var.detach()
    eps = bn.eps
    gamma = bn.weight.detach() if bn.weight is not None else torch.ones_like(rm)
    beta = bn.bias.detach() if bn.bias is not None else torch.zeros_like(rm)
    scale = gamma / torch.sqrt(rv + eps)   # (out,)
    w_fused = w * scale.unsqueeze(1)        # diag(scale) @ W
    b_fused = (b - rm) * scale + beta
    fused.weight.data.copy_(w_fused)
    fused.bias.data.copy_(b_fused)
    return fused


def _replace_module_attr(parent: nn.Module, target: nn.Module, new: nn.Module) -> bool:
    """Find which attribute on `parent` points at `target` and rebind it.

    Walks `parent._modules` (which is what `nn.Module.__setattr__` uses) so the
    swap is visible via `parent.named_modules()` afterwards. Returns True if
    found, False otherwise (silently — the caller decides whether to raise).
    """
    for name, child in list(parent._modules.items()):
        if child is target:
            setattr(parent, name, new)
            return True
    return False


def fuse_for_inference(model: nn.Module) -> nn.Module:
    """Walk `model` and fold every declared (absorber, absorbee) pair in-place.

    Returns the same model for chaining. Idempotent — a second call is a no-op
    because each block's `fusion_pairs` is cleared after a successful fold.
    """
    model.eval()
    for block in model.modules():
        pairs = getattr(block, "fusion_pairs", None)
        if not pairs:
            continue
        kept: list[tuple[nn.Module, nn.Module]] = []
        for absorber, absorbee in pairs:
            if isinstance(absorber, nn.Conv2d) and isinstance(absorbee, (nn.BatchNorm2d, nn.BatchNorm1d)):
                fused = fuse_conv_bn_eval(absorber, absorbee)
            elif isinstance(absorber, nn.Linear) and isinstance(absorbee, nn.BatchNorm1d):
                fused = fuse_linear_bn_eval(absorber, absorbee)
            else:
                # Unknown combination — leave it.
                kept.append((absorber, absorbee))
                continue
            ok_absorb = _replace_module_attr(block, absorber, fused)
            ok_bn = _replace_module_attr(block, absorbee, nn.Identity())
            if not (ok_absorb and ok_bn):
                # Couldn't find the attribute — keep the metadata so the caller
                # has a chance to inspect/raise downstream.
                kept.append((absorber, absorbee))
        block.fusion_pairs = kept
    return model
