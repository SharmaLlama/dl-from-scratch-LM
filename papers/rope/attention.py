import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention.base import BaseMultiHeadAttention
from papers.rope.positional_encoding import RotaryEmbedding


class RoPEMultiHeadAttention(BaseMultiHeadAttention):
    """
    Multi-head attention with rotary positional embedding (RoPE) applied to Q
    and K. Rotation acts over the per-head channel dim (dk).
    Args:
        max_seq_len: size of the cached sin/cos table. If a longer sequence is
            seen at runtime the cache is rebuilt.
        rope_base: base for the inverse-frequency schedule, default 10000 from
            the original paper.
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        dk: int,
        dv: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__(n_heads, d_model, dk, dv, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(head_dim=dk, max_seq_len=max_seq_len, base=rope_base)

    def _apply_position_bias(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Q, K: (B, H, N, dk). Rotate over the last dim.
        Q = self.rotary(Q)
        K = self.rotary(K)
        return Q, K, V

    def attention_pattern(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)

        if mask is not None:
            scores.masked_fill_(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        if return_attention:
            return output, attn_weights
        return output, None
