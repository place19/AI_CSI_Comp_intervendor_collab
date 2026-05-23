"""Per-block metadata recorded by build_encoder / build_decoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BlockTraceEntry:
    idx: int                       # position within the encoder or decoder
    name: str                      # registry key, e.g. 'cnn_block'
    in_shape: Tuple[int, ...]      # shape excluding batch dim
    out_shape: Tuple[int, ...]     # shape excluding batch dim
    num_params: int
