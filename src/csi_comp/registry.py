"""Namespaced name→class registry shared by blocks, losses, quantizers, etc."""
from __future__ import annotations

from typing import Callable, Dict, Type

REGISTRY: Dict[str, Dict[str, Type]] = {
    "block": {},
    "quantizer": {},
    "loss": {},
    "dataset": {},
    "quant_forward": {},
    "quant_backward": {},
    "scheduler": {},
}


def register(kind: str, name: str) -> Callable[[Type], Type]:
    if kind not in REGISTRY:
        raise KeyError(f"unknown registry kind: {kind!r}")

    def deco(cls: Type) -> Type:
        existing = REGISTRY[kind].get(name)
        if existing is not None and existing is not cls:
            raise KeyError(f"{kind}/{name} already registered to {existing!r}")
        REGISTRY[kind][name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> Type:
    if kind not in REGISTRY:
        raise KeyError(f"unknown registry kind: {kind!r}")
    if name not in REGISTRY[kind]:
        available = sorted(REGISTRY[kind].keys())
        raise KeyError(f"{kind}/{name} not registered; available: {available}")
    return REGISTRY[kind][name]


def available(kind: str) -> list[str]:
    if kind not in REGISTRY:
        raise KeyError(f"unknown registry kind: {kind!r}")
    return sorted(REGISTRY[kind].keys())
