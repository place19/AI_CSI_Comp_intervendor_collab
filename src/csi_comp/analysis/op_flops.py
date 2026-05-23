"""Per-op FLOP formulas. Strict count: every multiply and every add is one op.

These are pure functions of integer shape parameters. They report the FLOPs of a
**fused-model inference** view of each op:
- Linear/Conv `bias=True` adds an explicit `+ M` for the bias add.
- Absorbed BN contributes 0; the absorbing Conv/Linear is invoked with `bias=True`
  even if its actual PyTorch flag is `False`.
- Activations, softmax, and norms use HW-style approximate per-element costs
  (better than 0; documented as approximations).
"""
from __future__ import annotations


# ----- Linear / Conv -----


def linear_flops(N: int, in_F: int, out_F: int, bias: bool) -> int:
    """Strict ops for `(N, in_F) @ (in_F, out_F) [+ bias]`.

    - muls: `N · out_F · in_F`
    - adds (summing in_F products): `N · out_F · (in_F − 1)`
    - bias add: `+ N · out_F` if biased.
    """
    muls = N * out_F * in_F
    adds = N * out_F * (in_F - 1)
    bias_adds = N * out_F if bias else 0
    return int(muls + adds + bias_adds)


def conv2d_flops(
    N: int, C_out: int, H_out: int, W_out: int,
    k_h: int, k_w: int, k_C: int, bias: bool,
) -> int:
    """Strict ops for a Conv2d (or ConvTranspose) producing `(N, C_out, H_out, W_out)`.

    `k_C = C_in / groups`. Each output element is a sum of `k_h·k_w·k_C` products:
    - muls per output: `k_h · k_w · k_C`
    - adds per output: `k_h · k_w · k_C − 1` (sum reduction)
    - bias add: `+ M` if biased, where `M = N · C_out · H_out · W_out`.
    """
    M = N * C_out * H_out * W_out
    kK = k_h * k_w * k_C
    muls = M * kK
    adds = M * (kK - 1)
    bias_adds = M if bias else 0
    return int(muls + adds + bias_adds)


# ----- Norms -----


def batchnorm_flops(N_elements: int) -> int:
    """Unfused BatchNorm. Per element:
        normalize ((x−μ)·inv_std) → 1 sub + 1 mul = 2 ops
        affine  (γ·x + β)          → 1 mul + 1 add = 2 ops
    Total 4 ops/element. (Running stats μ, σ², inv_std are precomputed buffers.)
    """
    return int(4 * N_elements)


def layernorm_flops(N_tokens: int, d: int) -> int:
    """Approximate LayerNorm cost per token:
        mean  → d adds + 1 mul (×1/d)
        var   → d subs + d muls + d adds + 1 mul     (var is mean of squared deviations)
        norm  → d subs + d muls (×inv_std)            (inv_std computed scalar-ish per token)
        affine→ d muls + d adds                       (γ·x̂ + β)
    Summed and rounded to `≈ 5·d` ops per token. Documented as approximate.
    """
    return int(5 * N_tokens * d)


# ----- Pool -----


def avgpool_flops(N: int, C: int, H_out: int, W_out: int, k_h: int, k_w: int) -> int:
    """Sum of k_h·k_w values per output (k_h·k_w−1 adds) + 1 mul by 1/k².
    Total `k_h · k_w` ops per output element.
    """
    M = N * C * H_out * W_out
    return int(M * k_h * k_w)


# ----- Element-wise activations -----


# HW-implementation-style per-element op cost. ReLU is a single comparison
# (counted as 1 op). Sigmoid/Tanh/GELU collapse exp/erf to constant-op approximations.
_ACT_OPS_PER_ELEM = {
    "identity": 0,
    "relu": 1,
    "sigmoid": 4,   # 1 exp + 1 add + 1 div + 1 sign / mul
    "tanh": 6,      # 2 exp + sub + add + div + sign
    "gelu": 8,      # Hendrycks polynomial form: ~8 ops/elem
}


def activation_flops(activation: str, N_elements: int) -> int:
    """Approximate per-element ops for common activations. Unknown name → 0."""
    return int(_ACT_OPS_PER_ELEM.get(activation, 0) * N_elements)


# ----- Softmax -----


def softmax_flops(N: int, D: int) -> int:
    """Softmax over `D`-length vectors, applied `N` times.
    Approx 5 ops/element: max scan + sub-max + exp + sum + div.
    """
    return int(5 * N * D)


# ----- Attention's functional matmuls (B·H independent S×S blocks) -----


def qk_t_flops(B: int, H: int, S: int, d_head: int) -> int:
    """QKᵀ: for each of (B·H) batches and each of S² output entries, a dot product
    over `d_head` elements: `d_head` muls + `d_head − 1` adds.
    """
    return int(B * H * S * S * (2 * d_head - 1))


def attn_v_flops(B: int, H: int, S: int, d_head: int) -> int:
    """attn @ V: for each of (B·H·S·d_head) outputs, dot product over `S` weighted
    values: `S` muls + `S − 1` adds.
    """
    return int(B * H * S * d_head * (2 * S - 1))
