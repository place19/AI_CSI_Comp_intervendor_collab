"""Transformer encoder block: explicit Q/K/V/O linear projections + FFN, selectable pre/post-LN."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ...registry import register
from .base import Block, make_activation


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with explicit Q/K/V/O `nn.Linear` projections.

    Behaviour is equivalent to `nn.MultiheadAttention(..., batch_first=True)`
    called with `query=key=value=x`, but every weight matrix is laid out as a
    separate `nn.Linear(d_model, d_model)` so it's straightforward to inspect,
    swap, or freeze each one independently (and they appear distinctly in
    `state_dict` / FLOPs traces).
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.d_head = self.d_model // self.nhead
        self.scale = self.d_head ** -0.5

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        # Project then split into heads: (B, S, d_model) -> (B, H, S, d_head)
        q = self.W_Q(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, S, self.nhead, self.d_head).transpose(1, 2)

        # Scores: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # fp32 island around softmax: a previous AMP attempt blew up in backprop
        # when softmax ran in fp16. Cost is negligible (one (B,H,S,S) cast).
        with torch.amp.autocast(device_type=scores.device.type, enabled=False):
            attn = torch.softmax(scores.float(), dim=-1)
        attn = attn.to(v.dtype)
        attn = self.attn_dropout(attn)
        # (B, H, S, d_head) → (B, S, d_model)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_O(out)


@register("block", "transformer_block")
class TransformerBlock(Block):
    """Single transformer encoder layer (no positional encoding — caller adds it
    if needed). Operates on tokens of shape (B, S, F).

    Input  shape (excl. batch): (S, F)
    Output shape (excl. batch): (S, F)   — d_model must equal F.

    `d_model` defaults to F. Passing a `d_model` that disagrees with F raises:
    align shapes upstream with a `linear_proj` (or similar) block instead.

    `norm_position`:
        pre  — x = x + attn(ln1(x));  x = x + ff(ln2(x))   (GPT-2 / LLaMA style; default)
        post — x = ln1(x + attn(x));  x = ln2(x + ff(x))   (original Transformer paper)
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        d_model: Optional[int] = None,
        nhead: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_position: str = "pre",
    ):
        super().__init__(in_shape)
        if len(self.in_shape) != 2:
            raise ValueError(f"transformer_block expects (S, F), got {self.in_shape}")
        S, F = self.in_shape
        d_model = int(d_model) if d_model is not None else int(F)
        if d_model != F:
            raise ValueError(
                f"transformer_block: d_model ({d_model}) must equal the feature dim of "
                f"in_shape (F={F}). Insert a linear_proj/reshape block upstream to "
                f"align shapes instead of relying on an in-block projection."
            )
        if norm_position not in ("pre", "post"):
            raise ValueError(
                f"transformer_block: norm_position must be 'pre' or 'post', got {norm_position!r}"
            )
        ff_dim = int(ff_dim) if ff_dim is not None else 4 * d_model

        self.attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            make_activation(activation),
            nn.Linear(ff_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.norm_position = norm_position
        self.out_shape = (int(S), int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_position == "pre":
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ff(x))
        return x

    def count_flops(self, in_shape: Tuple[int, ...]) -> int:
        # Default leaf walk covers Q/K/V/O Linears, FFN Linears, LayerNorms,
        # GELU/ReLU. The attention's functional ops (QKᵀ matmul, softmax,
        # attn·V matmul) are not nn.Modules, so we add them here.
        from ...analysis.profiler import default_block_flops
        from ...analysis import op_flops as F
        base = default_block_flops(self, in_shape)
        S, _ = in_shape
        H = self.attn.nhead
        d = self.attn.d_head
        extra = (
            F.qk_t_flops(B=1, H=H, S=S, d_head=d)
            + F.softmax_flops(N=H * S, D=S)
            + F.attn_v_flops(B=1, H=H, S=S, d_head=d)
        )
        return int(base + extra)
