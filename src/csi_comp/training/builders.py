"""Glue: build model (encoder/quantizer/decoder per mode), optimizer, scheduler."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..models import Autoencoder, BlockTraceEntry, build_decoder, build_encoder
from ..quantization.base import build_quantizer
from ..registry import get as reg_get
from ..utils import filter_kwargs
from .modes import ModeSpec


def build_model(
    cfg: dict[str, Any],
    mode_spec: ModeSpec,
) -> Tuple[Autoencoder, List[BlockTraceEntry], List[BlockTraceEntry]]:
    """Build the model pieces required by `mode_spec` and wrap them in an Autoencoder.

    Returns (autoencoder, encoder_trace, decoder_trace). Either trace may be empty
    if that submodule is not built for the mode.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    encoder = quantizer = decoder = None
    enc_trace: List[BlockTraceEntry] = []
    dec_trace: List[BlockTraceEntry] = []
    latent_shape: tuple[int, ...] = ()

    if mode_spec.needs_encoder:
        encoder, enc_trace = build_encoder(model_cfg, data_cfg)
        latent_shape = enc_trace[-1].out_shape

    if mode_spec.needs_quantizer:
        if encoder is None:
            raise ValueError(
                "quantizer requested but no encoder; "
                "decoder_only mode does not need a quantizer"
            )
        quantizer = build_quantizer(cfg["quantizer"])

    if mode_spec.needs_decoder:
        if encoder is None:
            ls_cfg = model_cfg["decoder"].get("latent_shape")
            if ls_cfg is None:
                raise ValueError(
                    "decoder_only mode requires model.decoder.latent_shape "
                    "to be specified in the config"
                )
            latent_shape = tuple(int(x) for x in ls_cfg)
        decoder, dec_trace = build_decoder(model_cfg, data_cfg, latent_shape)

        if "decoder" in mode_spec.frozen_inference:
            path = model_cfg["decoder"].get("pretrained_path")
            if not path:
                raise ValueError(
                    f"mode={mode_spec.name!r} requires model.decoder.pretrained_path"
                )
            _load_decoder_state(decoder, Path(path))
            for p in decoder.parameters():
                p.requires_grad_(False)
            decoder.eval()

    ae = Autoencoder(encoder, quantizer, decoder)
    _freeze_non_trainable(ae, mode_spec)
    return ae, enc_trace, dec_trace


def _load_decoder_state(decoder: nn.Module, path: Path) -> None:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # Accept either a raw decoder state_dict or a checkpoint dict with 'decoder' key.
    if isinstance(sd, dict) and "decoder" in sd and isinstance(sd["decoder"], dict):
        sd = sd["decoder"]
    decoder.load_state_dict(sd)


def _freeze_non_trainable(ae: Autoencoder, mode_spec: ModeSpec) -> None:
    name_to_mod = {"encoder": ae.encoder, "quantizer": ae.quantizer, "decoder": ae.decoder}
    for name, mod in name_to_mod.items():
        if mod is None:
            continue
        if name not in mode_spec.trainable:
            for p in mod.parameters():
                p.requires_grad_(False)


def _split_decay_no_decay(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Place LayerNorm parameters and 1-D / bias parameters into a no-decay group.

    Convention popularised by GPT-2, BERT, and the HuggingFace defaults:
    weight decay on bias and LayerNorm parameters is empirically harmful (LN
    affine params serve a structural role and shouldn't be pulled toward zero;
    biases don't meaningfully overfit). All 2-D+ weights still get decay.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(model: nn.Module, opt_cfg: dict[str, Any]) -> Optimizer:
    name = opt_cfg["name"].lower()
    cfg = dict(opt_cfg)
    cfg.pop("name")
    # Optional escape hatch: if a recipe genuinely wants decay applied to bias
    # and LN as well, set `decay_norm_bias: true` in YAML.
    decay_all = bool(cfg.pop("decay_norm_bias", False))
    weight_decay = float(cfg.pop("weight_decay", 0.0))

    if decay_all:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("optimizer has no trainable parameters")
        param_groups: list[Any] = params
        cfg["weight_decay"] = weight_decay
    else:
        groups = _split_decay_no_decay(model, weight_decay)
        if not groups:
            raise ValueError("optimizer has no trainable parameters")
        param_groups = groups

    if name == "adam":
        return torch.optim.Adam(param_groups, **cfg)
    if name == "adamw":
        return torch.optim.AdamW(param_groups, **cfg)
    if name == "sgd":
        return torch.optim.SGD(param_groups, **cfg)
    raise ValueError(f"unknown optimizer: {name!r}")


def build_scheduler(
    optimizer: Optimizer,
    sched_cfg: dict[str, Any] | None,
    *,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
):
    """Look up a registered scheduler factory by `sched_cfg['name']` and call it.

    If the chosen factory accepts `total_steps` and the caller passed both
    `epochs` and `steps_per_epoch`, `total_steps` is auto-filled with their
    product unless the config already supplies it.
    """
    if not sched_cfg:
        return None
    # Ensure the schedulers package is imported so @register fires.
    from . import schedulers  # noqa: F401

    name = sched_cfg["name"].lower()
    kwargs = {k: v for k, v in sched_cfg.items() if k != "name"}

    factory = reg_get("scheduler", name)
    # Auto-fill total_steps when the factory accepts it and YAML didn't set it.
    if epochs is not None and steps_per_epoch is not None:
        kwargs.setdefault("total_steps", int(epochs) * int(steps_per_epoch))
    # Drop any keys the factory doesn't accept (e.g. cosine doesn't want total_steps).
    kwargs = filter_kwargs(factory, kwargs)
    return factory(optimizer, **kwargs)
