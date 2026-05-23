"""Unit tests for per-op FLOP formulas in csi_comp.analysis.op_flops."""
from __future__ import annotations

import pytest

from csi_comp.analysis import op_flops as F


# ---------- Linear ----------

def test_linear_no_bias():
    # N=4, in_F=8, out_F=16: 4*16*8 muls + 4*16*7 adds = 512 + 448 = 960
    assert F.linear_flops(N=4, in_F=8, out_F=16, bias=False) == 960


def test_linear_with_bias():
    # adds extra 4*16 = 64 → 1024
    assert F.linear_flops(N=4, in_F=8, out_F=16, bias=True) == 1024


def test_linear_bias_equals_2_N_in_out():
    # Closed-form sanity: with bias = 2 * N * in_F * out_F
    assert F.linear_flops(N=10, in_F=20, out_F=30, bias=True) == 2 * 10 * 20 * 30


# ---------- Conv2d ----------

def test_conv2d_no_bias_hand_computed():
    # M = 1*2*3*3 = 18; kK = 3*3*4 = 36
    # muls = 18*36 = 648; adds = 18*35 = 630; total = 1278
    f = F.conv2d_flops(N=1, C_out=2, H_out=3, W_out=3, k_h=3, k_w=3, k_C=4, bias=False)
    assert f == 1278


def test_conv2d_with_bias():
    # Above + M = 18 → 1296
    f = F.conv2d_flops(N=1, C_out=2, H_out=3, W_out=3, k_h=3, k_w=3, k_C=4, bias=True)
    assert f == 1296


# ---------- Norms ----------

def test_batchnorm_4_ops_per_element():
    assert F.batchnorm_flops(N_elements=100) == 400


def test_layernorm_5_per_token_per_d():
    # N_tokens=2, d=64 → 5*2*64 = 640
    assert F.layernorm_flops(N_tokens=2, d=64) == 640


# ---------- Pool ----------

def test_avgpool_total_ops_per_output_is_kk():
    f = F.avgpool_flops(N=1, C=4, H_out=8, W_out=8, k_h=3, k_w=3)
    assert f == 1 * 4 * 8 * 8 * 9


# ---------- Activations ----------

@pytest.mark.parametrize("name,cost", [
    ("identity", 0),
    ("relu", 1),
    ("sigmoid", 4),
    ("tanh", 6),
    ("gelu", 8),
])
def test_activation_per_element_cost(name, cost):
    assert F.activation_flops(name, N_elements=10) == 10 * cost


def test_activation_unknown_is_zero():
    assert F.activation_flops("softplus", N_elements=100) == 0


# ---------- Softmax ----------

def test_softmax_5ND():
    assert F.softmax_flops(N=3, D=7) == 5 * 3 * 7


# ---------- Attention ----------

def test_qk_t_flops():
    # B=1, H=4, S=13, d_head=16 → 4 * 169 * (2*16 - 1) = 4 * 169 * 31
    expected = 1 * 4 * 13 * 13 * (2 * 16 - 1)
    assert F.qk_t_flops(B=1, H=4, S=13, d_head=16) == expected


def test_attn_v_flops():
    expected = 1 * 4 * 13 * 16 * (2 * 13 - 1)
    assert F.attn_v_flops(B=1, H=4, S=13, d_head=16) == expected
