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
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # Replace NaN from fully-masked rows (can occur at the first token position)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        if return_attention:
            return output, attn_weights
        return output, None
