"""Importing this package registers every built-in block."""
from . import (  # noqa: F401
    activation,
    avg_pool,
    cnn,
    complex_ffn_head,
    dw_sep_conv,
    heads,
    mlp,
    positional_encoding,
    reshape,
    residual,
    transformer,
)
from .base import Block

__all__ = ["Block"]
