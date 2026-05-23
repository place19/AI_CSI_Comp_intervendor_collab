"""Weighted-sum loss composer."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..registry import get as reg_get


def _enabled_for(mode: Optional[str], term_mode: Optional[Any]) -> bool:
    if term_mode is None:
        return True
    if isinstance(term_mode, str):
        return mode == term_mode
    if isinstance(term_mode, Sequence):
        return mode in term_mode
    raise TypeError(f"enabled_when must be None | str | list, got {type(term_mode).__name__}")


class WeightedSumLoss(nn.Module):
    """Sum of named loss terms, each multiplied by a YAML-defined weight.

    Term spec (from cfg.loss.terms):
        - name: 'one_minus_sgcs'
          weight: 1.0
          params: {eps: 1e-12}          # optional kwargs forwarded to the term class
          enabled_when: 'joint'         # optional: str or list[str]; default = always on
    """

    def __init__(self, terms_cfg: Iterable[dict], mode: Optional[str] = None):
        super().__init__()
        terms: list[Tuple[float, nn.Module]] = []
        for spec in terms_cfg:
            if not _enabled_for(mode, spec.get("enabled_when")):
                continue
            name = spec["name"]
            weight = float(spec.get("weight", 1.0))
            params = spec.get("params", {}) or {}
            cls = reg_get("loss", name)
            terms.append((weight, cls(**params)))
        if not terms:
            raise ValueError(f"no loss terms enabled for mode={mode!r}")
        # Store as ModuleList so .parameters() walks them.
        self.term_modules = nn.ModuleList([t for _, t in terms])
        self.weights = [w for w, _ in terms]

    def forward(
        self,
        pred_pack: dict[str, Any],
        target_pack: dict[str, Any],
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total: torch.Tensor = torch.zeros((), device=_infer_device(pred_pack))
        logs: dict[str, torch.Tensor] = {}
        for w, term in zip(self.weights, self.term_modules):
            v = term(pred_pack, target_pack)
            total = total + w * v
            logs[term.name] = v.detach()
        return total, logs


def _infer_device(pack: dict[str, Any]) -> torch.device:
    for v in pack.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
