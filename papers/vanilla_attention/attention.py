import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention.base import BaseMultiHeadAttention


class VanillaMultiHeadAttention(BaseMultiHeadAttention):
    """
    Scaled dot-product multi-head attention from 'Attention Is All You Need'.
    No positional bias — position is handled by the embedding layer (sinusoidal or learned).
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        dk: int,
        dv: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(n_heads, d_model, dk, dv, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def attention_pattern(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Q, K, V: (B, H, N, d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)

        if mask is not None:
            # In-place masked_fill_ avoids the (B, H, N, N) temp that the OOP
            # variant allocates — meaningful at long context lengths.
            scores.masked_fill_(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # Note: under a strict causal mask every row has at least one True
        # (position i always attends to itself), so softmax cannot produce NaN
        # and the previous nan_to_num call has been removed — it created a
        # full-size fp32 copy of attn_weights every layer.

        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        if return_attention:
            return output, attn_weights
        return output, None
