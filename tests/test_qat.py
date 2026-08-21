"""Tests for encoder QAT: config resolution, fusion, and the float-layout save contract.

The single most important property under test is the checkpoint contract: a QAT run
must produce a `state_dict` that is key-for-key identical to a plain float run's, so
`test.py` / `infer.py` / `export_onnx.py` keep working with no QAT awareness.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.ao.quantization as tq

try:
    import torch.ao.nn.intrinsic as nni
    import torch.ao.nn.intrinsic.qat as nniqat
except ImportError:  # pragma: no cover - older torch layout
    import torch.nn.intrinsic as nni
    import torch.nn.intrinsic.qat as nniqat

from csi_comp.models import Autoencoder, build_encoder
from csi_comp.training import build_model, get_mode_spec
from csi_comp.training.checkpoint import load_checkpoint, save_checkpoint
from csi_comp.training.qat import (
    QATCallback,
    build_qconfigs,
    float_state_dict,
    prepare_encoder_qat,
    qat_observer_state_dict,
    resolve_qat_cfg,
)

CPU = torch.device("cpu")
MAX_S, MAX_P = 8, 12

_CNN_BLOCKS = [
    {"name": "cnn_block", "channels": 4, "kernel": 3},
    {"name": "cnn_block", "channels": 8, "kernel": 3},
    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
    {"name": "activation", "activation": "tanh"},
]
# dw_sep_conv declares two (conv, BN) fusion pairs per block; residual nests blocks.
_DWSEP_BLOCKS = [
    {"name": "dw_sep_conv", "out_channels": 4},
    {
        "name": "residual",
        "main_blocks": [{"name": "dw_sep_conv", "out_channels": 4}],
        "skip_blocks": [],
    },
    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
    {"name": "activation", "activation": "tanh"},
]
_TRANSFORMER_BLOCKS = [
    {"name": "transformer_block", "d_model": MAX_P * 2, "nhead": 2},
    {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
    {"name": "activation", "activation": "tanh"},
]


def _qat_cfg(**overrides) -> dict:
    cfg = {"enabled": True}
    cfg.update(overrides)
    return cfg


def _data_cfg(layout: str = "cnn") -> dict:
    return {"layout": layout, "max_subband": MAX_S, "max_port": MAX_P}


def _encoders(blocks, layout="cnn"):
    """Two identically-built encoders: one to prepare, one as the float reference."""
    model_cfg = {"encoder": {"blocks": copy.deepcopy(blocks)}}
    enc, _ = build_encoder(copy.deepcopy(model_cfg), _data_cfg(layout))
    ref, _ = build_encoder(copy.deepcopy(model_cfg), _data_cfg(layout))
    return enc, ref


def _prepare(blocks, layout="cnn", **qat_overrides):
    enc, ref = _encoders(blocks, layout)
    spec = resolve_qat_cfg(_qat_cfg(**qat_overrides), CPU)
    plan = prepare_encoder_qat(Autoencoder(encoder=enc), spec)
    return enc, ref, spec, plan


def _inputs(n: int = 4):
    return torch.randn(n, MAX_S, MAX_P), torch.randn(n, MAX_S, MAX_P)


# ---------------------------------------------------------------------------
# resolve_qat_cfg
# ---------------------------------------------------------------------------

def test_resolve_returns_none_when_absent_or_disabled():
    assert resolve_qat_cfg(None, CPU) is None
    assert resolve_qat_cfg({}, CPU) is None
    assert resolve_qat_cfg({"enabled": False}, CPU) is None


def test_resolve_rejects_mps():
    with pytest.raises(ValueError, match="not supported on MPS"):
        resolve_qat_cfg(_qat_cfg(), torch.device("mps"))


def test_resolve_defaults():
    spec = resolve_qat_cfg(_qat_cfg(), CPU)
    assert spec.fold_bn and spec.quantize_input and spec.quantize_activations
    assert spec.conv_weight.qscheme is torch.per_channel_symmetric
    assert spec.conv_weight.dtype is torch.qint8
    assert spec.activation.qscheme is torch.per_tensor_affine
    assert spec.activation.dtype is torch.quint8
    assert spec.exclude == ()


@pytest.mark.parametrize(
    "bits,dtype,lo,hi",
    [(8, "qint8", -128, 127), (4, "qint8", -8, 7), (8, "quint8", 0, 255), (4, "quint8", 0, 15)],
)
def test_bits_map_to_quant_range(bits, dtype, lo, hi):
    spec = resolve_qat_cfg(
        _qat_cfg(weight={"bits": bits, "dtype": dtype, "qscheme": "per_tensor_symmetric"}), CPU
    )
    assert (spec.conv_weight.quant_min, spec.conv_weight.quant_max) == (lo, hi)


def test_resolve_rejects_unknown_keys_and_per_channel_activation():
    with pytest.raises(ValueError, match="unexpected keys"):
        resolve_qat_cfg(_qat_cfg(bogus=1), CPU)
    with pytest.raises(ValueError, match="unexpected keys"):
        resolve_qat_cfg(_qat_cfg(weight={"nope": 1}), CPU)
    with pytest.raises(ValueError, match="per-tensor"):
        resolve_qat_cfg(_qat_cfg(activation={"qscheme": "per_channel_affine"}), CPU)


def test_quantize_activations_false_gives_identity_activation():
    spec = resolve_qat_cfg(_qat_cfg(quantize_activations=False), CPU)
    conv_qc, _ = build_qconfigs(spec)
    assert conv_qc.activation is nn.Identity


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "blocks,layout,n_fused",
    [(_CNN_BLOCKS, "cnn", 2), (_DWSEP_BLOCKS, "cnn", 4), (_TRANSFORMER_BLOCKS, "transformer", 0)],
)
def test_fusion_counts_and_forward(blocks, layout, n_fused):
    enc, _, _, plan = _prepare(blocks, layout)
    assert len(plan.fused_to_bn) == n_fused
    out = enc(*_inputs())
    out.sum().backward()
    assert out.shape == (4, 16)


def test_cnn_block_fuses_conv_bn_relu_and_clears_fusion_pairs():
    enc, _, _, plan = _prepare(_CNN_BLOCKS)
    assert plan.fused_to_bn == {"blocks.0.conv": "blocks.0.norm",
                                "blocks.1.conv": "blocks.1.norm"}
    block = enc.blocks[0]
    # ReLU follows the BN in _modules order, so all three fuse.
    assert isinstance(block.conv, nniqat.ConvBnReLU2d)
    assert isinstance(block.norm, nn.Identity) and isinstance(block.act, nn.Identity)
    # BN survives inside the fused module — that is what makes the save round-trip work.
    assert isinstance(block.conv.bn, nn.BatchNorm2d)
    assert block.fusion_pairs == []


def test_non_relu_activation_fuses_conv_bn_only():
    blocks = [
        {"name": "cnn_block", "channels": 4, "kernel": 3, "activation": "gelu"},
        {"name": "linear_proj", "out_dim": 16, "activation": "relu"},
        {"name": "activation", "activation": "tanh"},
    ]
    enc, _, _, plan = _prepare(blocks)
    assert len(plan.fused_to_bn) == 1
    assert isinstance(enc.blocks[0].conv, nniqat.ConvBn2d)
    assert isinstance(enc.blocks[0].act, nn.GELU)  # untouched, no fused QAT module exists


def test_stubs_installed_and_skippable():
    enc, _, _, _ = _prepare(_CNN_BLOCKS)
    assert isinstance(enc.quant, tq.FakeQuantizeBase) or hasattr(enc.quant, "activation_post_process")
    enc2, _, _, _ = _prepare(_CNN_BLOCKS, quantize_input=False)
    assert isinstance(enc2.quant, nn.Identity) and isinstance(enc2.dequant, nn.Identity)


def test_decoder_and_latent_quantizer_untouched():
    cfg = {
        "data": _data_cfg(),
        "model": {
            "encoder": {"blocks": copy.deepcopy(_CNN_BLOCKS)},
            "decoder": {"blocks": [
                {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                {"name": "reshape_head", "max_subband": MAX_S, "max_port": MAX_P},
            ]},
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
    }
    ae, _, _ = build_model(cfg, get_mode_spec("joint"))
    dec_before = {k: v.clone() for k, v in ae.decoder.state_dict().items()}
    prepare_encoder_qat(ae, resolve_qat_cfg(_qat_cfg(), CPU))
    assert set(ae.decoder.state_dict()) == set(dec_before)
    assert not any(hasattr(m, "weight_fake_quant") for m in ae.decoder.modules())
    assert not any(hasattr(m, "weight_fake_quant") for m in ae.quantizer.modules())


def test_decoder_only_mode_raises():
    spec = resolve_qat_cfg(_qat_cfg(), CPU)
    with pytest.raises(ValueError, match="no encoder"):
        prepare_encoder_qat(Autoencoder(encoder=None), spec)


# ---------------------------------------------------------------------------
# the float-layout save contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "blocks,layout",
    [(_CNN_BLOCKS, "cnn"), (_DWSEP_BLOCKS, "cnn"), (_TRANSFORMER_BLOCKS, "transformer")],
)
def test_float_state_dict_matches_float_model_exactly(blocks, layout):
    enc, ref, _, plan = _prepare(blocks, layout)
    enc(*_inputs())  # let observers see data so their buffers are populated
    fsd = float_state_dict(enc, plan)
    assert list(fsd) == list(ref.state_dict())
    assert {k: v.shape for k, v in fsd.items()} == {
        k: v.shape for k, v in ref.state_dict().items()
    }
    ref.load_state_dict(fsd, strict=True)  # raises on any mismatch


def test_float_state_dict_carries_trained_values_and_no_observer_keys():
    enc, ref, _, plan = _prepare(_CNN_BLOCKS)
    before = enc.blocks[0].conv.weight.detach().clone()
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        enc(*_inputs()).square().mean().backward()
        opt.step()

    fsd = float_state_dict(enc, plan)
    assert not any(
        p in k.split(".") for k in fsd for p in ("weight_fake_quant", "activation_post_process")
    )
    assert not any(k.startswith(("quant.", "dequant.")) for k in fsd)
    assert not torch.equal(fsd["blocks.0.conv.weight"], before)
    # BN moved back out of the fused module and reflects QAT-updated running stats.
    assert torch.equal(fsd["blocks.0.norm.running_var"], enc.blocks[0].conv.bn.running_var)
    ref.load_state_dict(fsd, strict=True)


def test_observer_state_is_the_exact_complement():
    enc, _, _, plan = _prepare(_CNN_BLOCKS)
    enc(*_inputs())
    parts = ("weight_fake_quant", "activation_post_process")
    full = set(enc.state_dict())
    observer = set(qat_observer_state_dict(enc))
    plain = {k for k in full if not any(p in k.split(".") for p in parts)}

    assert observer, "expected observer buffers after preparation"
    assert observer | plain == full
    assert observer & plain == set()
    # float_state_dict covers exactly the non-observer half; it only renames the
    # fused BN keys, so compare on size rather than on the key sets themselves.
    assert len(float_state_dict(enc, plan)) == len(plain)


def test_fake_quant_actually_changes_the_output():
    enc, _, _, _ = _prepare(_CNN_BLOCKS)
    x = _inputs()
    enc(*x)  # calibrate
    enc.eval()
    with torch.no_grad():
        quantized = enc(*x)
        enc.apply(tq.disable_fake_quant)
        floated = enc(*x)
    assert not torch.allclose(quantized, floated, atol=1e-7)


# ---------------------------------------------------------------------------
# per-kind weight schemes / exclude
# ---------------------------------------------------------------------------

def test_linear_weight_override_changes_only_linear_granularity():
    enc, _, _, _ = _prepare(
        _CNN_BLOCKS,
        weight={"qscheme": "per_channel_symmetric"},
        linear_weight={"qscheme": "per_tensor_symmetric"},
    )
    enc(*_inputs())  # a per-channel observer sizes `scale` only once it sees data
    assert enc.blocks[0].conv.weight_fake_quant.scale.numel() == 4   # per out_channel
    assert enc.blocks[1].conv.weight_fake_quant.scale.numel() == 8
    assert enc.blocks[2].linear.weight_fake_quant.scale.numel() == 1  # per tensor


def test_without_override_linear_inherits_the_shared_weight_scheme():
    enc, _, _, _ = _prepare(_CNN_BLOCKS, weight={"qscheme": "per_channel_symmetric"})
    enc(*_inputs())
    # linear_proj out_dim=16 -> one scale per output neuron (ch_axis=0 is the output dim)
    assert enc.blocks[2].linear.weight_fake_quant.scale.numel() == 16


def test_exclude_leaves_the_module_float_and_unfused():
    enc, ref, _, plan = _prepare(_CNN_BLOCKS, exclude=["blocks.0"])
    assert "blocks.0.conv" not in plan.fused_to_bn
    assert type(enc.blocks[0].conv) is nn.Conv2d
    assert isinstance(enc.blocks[0].norm, nn.BatchNorm2d)
    assert not hasattr(enc.blocks[0].conv, "weight_fake_quant")
    assert hasattr(enc.blocks[1].conv, "weight_fake_quant")
    ref.load_state_dict(float_state_dict(enc, plan), strict=True)


# ---------------------------------------------------------------------------
# QATCallback
# ---------------------------------------------------------------------------

class _FakeTrainer:
    def __init__(self, model):
        self.model = model


def test_callback_freezes_observers_and_bn_at_the_configured_epochs():
    enc, _, spec, _ = _prepare(_CNN_BLOCKS, freeze_observer_epoch=2, freeze_bn_epoch=3)
    cb = QATCallback(spec)
    trainer = _FakeTrainer(Autoencoder(encoder=enc))

    def observers_on():
        return [int(m.observer_enabled) for m in enc.modules() if isinstance(m, tq.FakeQuantizeBase)]

    cb.on_epoch_begin(trainer, 0)
    assert all(observers_on())
    assert all(m.bn.training for m in enc.modules() if isinstance(m, nniqat.ConvBnReLU2d))

    cb.on_epoch_begin(trainer, 2)
    assert not any(observers_on())

    cb.on_epoch_begin(trainer, 3)
    assert not any(m.bn.training for m in enc.modules() if isinstance(m, nniqat.ConvBnReLU2d))


def test_callback_is_a_noop_when_epochs_are_unset():
    enc, _, spec, _ = _prepare(_CNN_BLOCKS)
    cb = QATCallback(spec)
    cb.on_epoch_begin(_FakeTrainer(Autoencoder(encoder=enc)), 99)
    assert all(
        int(m.observer_enabled) for m in enc.modules() if isinstance(m, tq.FakeQuantizeBase)
    )


# ---------------------------------------------------------------------------
# checkpoint integration
# ---------------------------------------------------------------------------

def _joint_cfg():
    return {
        "data": _data_cfg(),
        "model": {
            "encoder": {"blocks": copy.deepcopy(_CNN_BLOCKS)},
            "decoder": {"blocks": [
                {"name": "linear_proj", "out_dim": 32, "activation": "relu"},
                {"name": "reshape_head", "max_subband": MAX_S, "max_port": MAX_P},
            ]},
        },
        "quantizer": {"type": "uniform", "bits": 2, "value_range": [-1.0, 1.0], "grad": "ste"},
        "training": {"mode": "joint"},
        "loss": {"terms": [{"name": "one_minus_sgcs", "weight": 1.0}]},
    }


def test_qat_checkpoint_loads_into_a_plain_float_model(tmp_path):
    cfg = _joint_cfg()
    spec_mode = get_mode_spec("joint")
    ae, _, _ = build_model(copy.deepcopy(cfg), spec_mode)
    plan = prepare_encoder_qat(ae, resolve_qat_cfg(_qat_cfg(), CPU))
    ae.encoder(*_inputs())

    path = tmp_path / "qat.pt"
    save_checkpoint(
        path, ae, torch.optim.SGD(ae.parameters(), lr=0.0), None,
        epoch=1, global_step=10, best_value=0.5, config=cfg, qat_plan=plan,
    )
    sd = torch.load(path, map_location="cpu", weights_only=False)
    assert "qat_observers" in sd and sd["qat_observers"]

    # A fresh float build — no QAT awareness at all — must accept it strictly.
    fresh, _, _ = build_model(copy.deepcopy(cfg), spec_mode)
    fresh.encoder.load_state_dict(sd["encoder"], strict=True)
    load_checkpoint(path, fresh, strict=True)


def test_non_qat_checkpoint_is_unchanged_by_the_stub_slots(tmp_path):
    """Regression: adding Encoder.quant/dequant must not alter the float contract."""
    cfg = _joint_cfg()
    ae, _, _ = build_model(copy.deepcopy(cfg), get_mode_spec("joint"))
    assert not any(k.startswith(("quant.", "dequant.")) for k in ae.encoder.state_dict())
    path = tmp_path / "float.pt"
    save_checkpoint(
        path, ae, torch.optim.SGD(ae.parameters(), lr=0.0), None,
        epoch=0, global_step=0, best_value=0.0, config=cfg,
    )
    sd = torch.load(path, map_location="cpu", weights_only=False)
    assert "qat_observers" not in sd
    fresh, _, _ = build_model(copy.deepcopy(cfg), get_mode_spec("joint"))
    load_checkpoint(path, fresh, strict=True)
