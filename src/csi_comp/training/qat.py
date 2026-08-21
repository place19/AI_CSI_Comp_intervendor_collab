"""Quantization-Aware Training for the encoder.

Purpose
-------
The final int8 conversion of this project's encoder is done by an **external HW
toolchain** — not by PyTorch and not by ONNX. Python only ever manages float
models. So QAT here has exactly one job: *train while knowing the weights will be
quantized later*. Fake-quant makes the network experience the quantization error
during fine-tuning, so the float weights it converges to are robust to the later
int8 conversion.

Do not confuse this with `quantization/` at the package root — that quantizes the
**latent codeword** and is a completely separate mechanism. QAT targets neural-net
weights/activations and never touches the latent `Quantizer` module.

Because the HW toolchain owns the real conversion, `training.qat.weight` /
`activation` must be configured to **match that toolchain's scheme**. Training
per-channel weights when the HW does per-tensor teaches the network the wrong
error distribution.

Scope
-----
Encoder only. The decoder runs at the gNB and the latent `Quantizer` has its own
scheme, so both are left in fp32.

BatchNorm
---------
`export/fuse.py::fuse_for_inference` folds Conv↔BN at deployment, so QAT must
quantize the *folded* weight to match. We reuse each block's `fusion_pairs`
metadata to drive `fuse_modules_qat`, which swaps in `nni.qat.ConvBn*` modules
that fold BN inside forward when computing the fake-quant scale.

Crucially the fold is **simulated, not destructive**: the QAT module keeps the
unfolded conv weight in `.weight` and a live `BatchNorm2d` in `.bn`. That is what
lets `float_state_dict` rebuild the original float layout exactly (see below).

Checkpoint contract
-------------------
`float_state_dict` turns a QAT-prepared encoder back into a state_dict that is
**key-for-key identical to the float model's** — observers are dropped and
`<fused>.bn.*` is remapped to the original BN path. So a QAT run's checkpoint is
indistinguishable from a normal one ("same model, different weights"), and
`test.py` / `infer.py` / `export_onnx.py` need no changes at all.

Device
------
CPU and CUDA only. MPS has no `fake_quantize_per_{tensor,channel}_affine` kernel.

torch compatibility
-------------------
Restricted to APIs present in torch 2.1.1 (the declared minimum in
`requirements.txt`). Notably `get_default_qat_qconfig()` is **not** used: its
defaults shift between torch versions, and matching a HW scheme requires spelling
the qconfig out anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.ao.quantization as tq
import torch.nn as nn

from .compile_utils import unwrap_compiled
from .trainer import TrainerCallback

# `torch.nn.intrinsic` was moved to `torch.ao.nn.intrinsic` in torch 2.0; keep the
# old path as a fallback so an older install still imports.
try:  # pragma: no cover - exercised by whichever torch is installed
    import torch.ao.nn.intrinsic as nni
    import torch.ao.nn.intrinsic.qat as nniqat
except ImportError:  # pragma: no cover
    import torch.nn.intrinsic as nni  # type: ignore[no-redef]
    import torch.nn.intrinsic.qat as nniqat  # type: ignore[no-redef]


_QSCHEMES: Dict[str, torch.qscheme] = {
    "per_tensor_affine": torch.per_tensor_affine,
    "per_tensor_symmetric": torch.per_tensor_symmetric,
    "per_channel_affine": torch.per_channel_affine,
    "per_channel_symmetric": torch.per_channel_symmetric,
}
_DTYPES: Dict[str, torch.dtype] = {"qint8": torch.qint8, "quint8": torch.quint8}

# state_dict path components introduced by `prepare_qat`. Every observer / fake-quant
# buffer lives under one of these, so dropping keys that contain them recovers the
# plain float parameter set. QuantStub / DeQuantStub contribute only such keys, which
# is why the stubs cost nothing in the saved checkpoint.
_OBSERVER_PARTS = ("weight_fake_quant", "activation_post_process")


# ----- scheme spec -----

@dataclass(frozen=True)
class QSchemeSpec:
    """One tensor's quantization scheme (`bits` + `dtype` + `qscheme`).

    `bits` narrows `quant_min`/`quant_max` while the storage dtype stays
    `qint8`/`quint8` — the standard way to simulate sub-8-bit quantization.
    """

    bits: int
    dtype: torch.dtype
    qscheme: torch.qscheme

    @property
    def is_per_channel(self) -> bool:
        return self.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric)

    @property
    def quant_min(self) -> int:
        return 0 if self.dtype == torch.quint8 else -(2 ** (self.bits - 1))

    @property
    def quant_max(self) -> int:
        if self.dtype == torch.quint8:
            return 2 ** self.bits - 1
        return 2 ** (self.bits - 1) - 1


def _parse_scheme(
    cfg: Optional[dict], *, default_dtype: str, default_qscheme: str, what: str
) -> QSchemeSpec:
    cfg = dict(cfg or {})
    bits = int(cfg.pop("bits", 8))
    dtype_name = str(cfg.pop("dtype", default_dtype))
    qscheme_name = str(cfg.pop("qscheme", default_qscheme))
    if cfg:
        raise ValueError(f"qat.{what}: unexpected keys {sorted(cfg)}")
    if not 2 <= bits <= 8:
        raise ValueError(f"qat.{what}.bits must be in [2, 8], got {bits}")
    if dtype_name not in _DTYPES:
        raise ValueError(
            f"qat.{what}.dtype must be one of {sorted(_DTYPES)}, got {dtype_name!r}"
        )
    if qscheme_name not in _QSCHEMES:
        raise ValueError(
            f"qat.{what}.qscheme must be one of {sorted(_QSCHEMES)}, got {qscheme_name!r}"
        )
    return QSchemeSpec(
        bits=bits, dtype=_DTYPES[dtype_name], qscheme=_QSCHEMES[qscheme_name]
    )


# ----- run spec -----

@dataclass(frozen=True)
class QATSpec:
    """Resolved `training.qat` block.

    `conv_weight` / `linear_weight` are already composed from the yaml `weight`
    default plus any `conv_weight` / `linear_weight` override, so consumers never
    deal with `None`. They are separate because `ch_axis=0` means *out_channels*
    for Conv and *out_features* for Linear, and many NPU toolchains support
    per-channel for convolutions but only per-tensor for fully-connected layers.
    """

    fold_bn: bool
    quantize_input: bool
    quantize_activations: bool
    conv_weight: QSchemeSpec
    linear_weight: QSchemeSpec
    activation: QSchemeSpec
    exclude: Tuple[str, ...]
    freeze_observer_epoch: Optional[int]
    freeze_bn_epoch: Optional[int]


@dataclass
class QATPlan:
    """What `float_state_dict` needs to undo the fusion.

    Maps the path of a fused QAT module to the path the absorbed BatchNorm
    occupied in the float model, e.g. `"blocks.0.conv" -> "blocks.0.norm"`.
    """

    fused_to_bn: Dict[str, str] = field(default_factory=dict)


def resolve_qat_cfg(
    qat_cfg: Optional[dict[str, Any]], device: torch.device
) -> Optional[QATSpec]:
    """Resolve a yaml `training.qat` block against the active device.

    Returns `None` when absent or `enabled: false`, which makes every QAT code
    path a no-op (same convention as `parse_latent_mask_spec`).
    """
    if not qat_cfg or not qat_cfg.get("enabled", False):
        return None
    if device.type == "mps":
        raise ValueError(
            "training.qat is not supported on MPS: torch has no "
            "fake_quantize_per_tensor_affine / per_channel_affine kernel for the MPS "
            "backend. Run QAT with experiment.device: cpu or cuda."
        )

    cfg = dict(qat_cfg)
    cfg.pop("enabled", None)
    base_weight = cfg.pop("weight", None)
    conv_weight = cfg.pop("conv_weight", None)
    linear_weight = cfg.pop("linear_weight", None)
    activation = cfg.pop("activation", None)
    fold_bn = bool(cfg.pop("fold_bn", True))
    quantize_input = bool(cfg.pop("quantize_input", True))
    quantize_activations = bool(cfg.pop("quantize_activations", True))
    exclude = tuple(str(p) for p in (cfg.pop("exclude", ()) or ()))
    freeze_observer_epoch = cfg.pop("freeze_observer_epoch", None)
    freeze_bn_epoch = cfg.pop("freeze_bn_epoch", None)
    if cfg:
        raise ValueError(f"training.qat: unexpected keys {sorted(cfg)}")

    def _weight(override, what):
        # An override inherits every key it doesn't set from the shared `weight`.
        merged = {**(base_weight or {}), **(override or {})}
        return _parse_scheme(
            merged, default_dtype="qint8",
            default_qscheme="per_channel_symmetric", what=what,
        )

    act = _parse_scheme(
        activation, default_dtype="quint8",
        default_qscheme="per_tensor_affine", what="activation",
    )
    if act.is_per_channel:
        raise ValueError(
            "qat.activation.qscheme must be per-tensor: eager-mode QAT has no "
            "per-channel activation observer (the channel axis of an activation is "
            "not fixed across the graph)."
        )
    return QATSpec(
        fold_bn=fold_bn,
        quantize_input=quantize_input,
        quantize_activations=quantize_activations,
        conv_weight=_weight(conv_weight, "conv_weight"),
        linear_weight=_weight(linear_weight, "linear_weight"),
        activation=act,
        exclude=exclude,
        freeze_observer_epoch=(
            None if freeze_observer_epoch is None else int(freeze_observer_epoch)
        ),
        freeze_bn_epoch=None if freeze_bn_epoch is None else int(freeze_bn_epoch),
    )


# ----- qconfig construction -----

def _fake_quant(spec: QSchemeSpec):
    """A `FakeQuantize` factory for `spec`, per-channel or per-tensor."""
    common = dict(
        quant_min=spec.quant_min,
        quant_max=spec.quant_max,
        dtype=spec.dtype,
        qscheme=spec.qscheme,
        reduce_range=False,
    )
    if spec.is_per_channel:
        return tq.FakeQuantize.with_args(
            observer=tq.MovingAveragePerChannelMinMaxObserver, ch_axis=0, **common
        )
    return tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver, **common
    )


def build_qconfigs(spec: QATSpec) -> Tuple[tq.QConfig, tq.QConfig]:
    """`(conv_qconfig, linear_qconfig)` — they differ only in the weight scheme.

    With `quantize_activations: false` the activation slot becomes `nn.Identity`,
    which `prepare_qat` instantiates as a pass-through observer, leaving
    weight-only fake-quant.
    """
    activation = nn.Identity if not spec.quantize_activations else _fake_quant(spec.activation)
    return (
        tq.QConfig(activation=activation, weight=_fake_quant(spec.conv_weight)),
        tq.QConfig(activation=activation, weight=_fake_quant(spec.linear_weight)),
    )


# ----- preparation -----

def _classes(module, names: Tuple[str, ...]) -> Tuple[type, ...]:
    return tuple(getattr(module, n) for n in names if hasattr(module, n))


_CONV_FUSED = _classes(nni, (
    "ConvBn1d", "ConvBn2d", "ConvBn3d",
    "ConvBnReLU1d", "ConvBnReLU2d", "ConvBnReLU3d",
    "ConvReLU1d", "ConvReLU2d", "ConvReLU3d",
))
_LINEAR_FUSED = _classes(nni, ("LinearBn1d", "LinearReLU"))
_BARE_CONV = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _is_excluded(path: str, exclude: Tuple[str, ...]) -> bool:
    return any(pat in path for pat in exclude)


def _trailing_relu(block: nn.Module, names: list[str], bn_name: str) -> Optional[str]:
    """Name of the `nn.ReLU` immediately following `bn_name`, if any.

    Assumes a block's `_modules` declaration order matches its forward order —
    true for every block in this repo (`cnn_block`: conv/norm/act, `dw_sep_conv`:
    dw/bn1/act1/pw/bn2/act2, `linear_proj`: linear/norm/act). If a future block
    breaks that convention the ReLU simply isn't fused, which costs a little
    activation-range fidelity but never corrupts the state_dict layout.
    Non-ReLU activations (GELU/tanh/sigmoid) have no fused QAT module, so they
    are left alone and only Conv+BN is fused.
    """
    i = names.index(bn_name)
    if i + 1 >= len(names):
        return None
    nxt = names[i + 1]
    return nxt if isinstance(block._modules[nxt], nn.ReLU) else None


def _fuse_declared_bn(encoder: nn.Module, exclude: Tuple[str, ...]) -> Dict[str, str]:
    """Fuse every `fusion_pairs`-declared Conv↔BN (plus a trailing ReLU) in place.

    Returns the `fused_path -> bn_path` map that `float_state_dict` needs to undo
    the fusion at save time. Each block's `fusion_pairs` is cleared for the pairs
    that were fused (same convention as `fuse_for_inference`) so stale module
    references can't mislead the profiler or a later fold.
    """
    from ..models.blocks.base import Block

    plan: Dict[str, str] = {}
    for path, block in encoder.named_modules():
        if not isinstance(block, Block):
            continue
        pairs = list(getattr(block, "fusion_pairs", ()) or ())
        if not pairs:
            continue
        # Captured before fusing: `fuse_modules_qat` rebinds these attribute names
        # but never adds or removes them, so the ordering stays valid throughout.
        names = list(block._modules.keys())
        by_id = {id(m): n for n, m in block._modules.items()}
        kept: list[tuple[nn.Module, nn.Module]] = []
        for absorber, absorbee in pairs:
            conv_name = by_id.get(id(absorber))
            bn_name = by_id.get(id(absorbee))
            if conv_name is None or bn_name is None:
                kept.append((absorber, absorbee))
                continue
            conv_path = f"{path}.{conv_name}" if path else conv_name
            if _is_excluded(conv_path, exclude):
                # Must skip fusion too, not just the qconfig: a fused container with
                # no qconfig survives `prepare_qat` unconverted and silently rewrites
                # the state_dict keys to `<conv>.0.weight` / `<conv>.1.*`.
                kept.append((absorber, absorbee))
                continue
            group = [conv_name, bn_name]
            relu_name = _trailing_relu(block, names, bn_name)
            if relu_name is not None:
                group.append(relu_name)
            tq.fuse_modules_qat(block, [group], inplace=True)
            plan[conv_path] = f"{path}.{bn_name}" if path else bn_name
        block.fusion_pairs = kept
    return plan


def _assign_qconfigs(
    encoder: nn.Module, spec: QATSpec, conv_qc: tq.QConfig, linear_qc: tq.QConfig
) -> int:
    """Tag quantizable modules with their qconfig; returns how many were tagged.

    Bare convs/linears are matched by exact `type(...)` rather than `isinstance`
    so QAT subclasses and `nni` fused containers (which subclass `nn.Conv2d` /
    `nn.Sequential`) can't be double-matched.
    """
    n = 0
    for path, module in encoder.named_modules():
        if _is_excluded(path, spec.exclude):
            continue
        if isinstance(module, (tq.QuantStub, tq.DeQuantStub)):
            # Stubs only ever use the activation half of the qconfig.
            module.qconfig = conv_qc
        elif isinstance(module, _CONV_FUSED) or type(module) in _BARE_CONV:
            module.qconfig = conv_qc
        elif isinstance(module, _LINEAR_FUSED) or type(module) is nn.Linear:
            module.qconfig = linear_qc
        else:
            continue
        n += 1
    return n


def prepare_encoder_qat(ae: nn.Module, spec: QATSpec) -> QATPlan:
    """Convert `ae.encoder` in place into a QAT (fake-quantized) encoder.

    Must run **before** the optimizer is built so it sees the swapped modules, and
    before `torch.compile`. `ae.decoder` and `ae.quantizer` are never touched.
    """
    encoder = unwrap_compiled(ae.encoder)
    if encoder is None:
        raise ValueError(
            "training.qat is enabled but this training mode has no encoder "
            "(decoder_only); QAT in this project targets the encoder only."
        )
    conv_qc, linear_qc = build_qconfigs(spec)

    if spec.quantize_input:
        encoder.quant = tq.QuantStub()
        encoder.dequant = tq.DeQuantStub()

    plan = QATPlan(
        fused_to_bn=_fuse_declared_bn(encoder, spec.exclude) if spec.fold_bn else {}
    )

    if _assign_qconfigs(encoder, spec, conv_qc, linear_qc) == 0:
        raise ValueError(
            "training.qat: no quantizable module found in the encoder (after "
            f"exclude={list(spec.exclude)}). Nothing would be fake-quantized."
        )

    encoder.train()  # prepare_qat asserts training mode
    tq.prepare_qat(encoder, inplace=True)

    # A fused container that didn't get converted keeps its Sequential layout and
    # would rewrite the state_dict keys, silently breaking the float round-trip.
    unconverted = [
        p for p, m in encoder.named_modules()
        if isinstance(m, nni._FusedModule) and not hasattr(m, "weight_fake_quant")
    ]
    if unconverted:
        raise RuntimeError(
            f"QAT preparation left fused modules unconverted: {unconverted}. "
            "This would corrupt the float state_dict layout on save."
        )
    return plan


# ----- checkpoint helpers -----

def float_state_dict(
    encoder: nn.Module, plan: Optional[QATPlan] = None
) -> Dict[str, torch.Tensor]:
    """A QAT encoder's state_dict rewritten to the plain float model's layout.

    Two transforms: drop every observer / fake-quant buffer, and move each fused
    module's absorbed BatchNorm (`<fused>.bn.*`) back to the path it had in the
    float model. The result loads into a freshly built (non-QAT) encoder with
    `strict=True`.
    """
    sd = unwrap_compiled(encoder).state_dict()
    remaps = tuple((f"{fused}.bn.", bn) for fused, bn in (plan.fused_to_bn if plan else {}).items())
    out: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if any(part in key.split(".") for part in _OBSERVER_PARTS):
            continue
        for prefix, bn_path in remaps:
            if key.startswith(prefix):
                key = f"{bn_path}.{key[len(prefix):]}"
                break
        out[key] = value
    return out


def qat_observer_state_dict(encoder: nn.Module) -> Dict[str, torch.Tensor]:
    """The complement of `float_state_dict`: only observer / fake-quant buffers.

    Saved under a separate top-level checkpoint key so the model entries stay
    float-shaped. Useful for resuming a QAT run without re-converging the
    observers, and for reading off the scales/zero-points QAT settled on when
    configuring the external HW toolchain.
    """
    return {
        k: v for k, v in unwrap_compiled(encoder).state_dict().items()
        if any(part in k.split(".") for part in _OBSERVER_PARTS)
    }


def load_qat_observer_state(encoder: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Restore observer state produced by `qat_observer_state_dict`."""
    result = unwrap_compiled(encoder).load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise KeyError(
            f"qat observer state has keys absent from the model: {result.unexpected_keys}"
        )


# ----- callback -----

class QATCallback(TrainerCallback):
    """Applies the two standard late-QAT schedule steps.

    Both are the usual recipe for the tail of a QAT run: once the network has
    adapted, pin the quantization grid (`freeze_observer_epoch`) and the BN
    statistics (`freeze_bn_epoch`) so the last epochs train against exactly the
    parameters deployment will use. Epochs are 0-based, matching `Trainer.epoch`.
    """

    def __init__(self, spec: QATSpec):
        self.spec = spec
        self._observers_frozen = False
        self._bn_frozen = False

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        encoder = unwrap_compiled(trainer.model.encoder)
        if encoder is None:
            return
        obs_at = self.spec.freeze_observer_epoch
        if not self._observers_frozen and obs_at is not None and epoch >= obs_at:
            encoder.apply(tq.disable_observer)
            self._observers_frozen = True
            print(f"[epoch {epoch}] qat: observers frozen (quantization grid pinned)")
        bn_at = self.spec.freeze_bn_epoch
        if not self._bn_frozen and bn_at is not None and epoch >= bn_at:
            encoder.apply(nniqat.freeze_bn_stats)
            self._bn_frozen = True
            print(f"[epoch {epoch}] qat: BatchNorm running stats frozen")
