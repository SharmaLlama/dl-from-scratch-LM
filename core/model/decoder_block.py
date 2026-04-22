from typing import Optional

import torch
import torch.nn as nn

from core.attention.base import BaseMultiHeadAttention
from core.layers.utils import LayerNorm, PositionWiseFFN, ResidualConnection, SwiGLU


class DecoderBlock(nn.Module):
    """
    Single decoder layer: causal self-attention + FFN + residual connections.

    The attention module is injected rather than constructed internally,
    so swapping attention mechanisms requires no changes here.
    """

    def __init__(
        self,
        attention: BaseMultiHeadAttention,
        d_model: int,
        d_ff: int,
        dropout: float,
        pre_norm: bool = True,
        ffn_type: str = "relu",
    ) -> None:
        super().__init__()
        self.attention = attention
        self.residual_attn = ResidualConnection(d_model, dropout, pre_norm)
        self.residual_ffn = ResidualConnection(d_model, dropout, pre_norm)

        if ffn_type == "swiglu":
            self.ffn: nn.Module = SwiGLU(d_model, d_ff, dropout)
        else:
            self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # Final norm for post-norm architecture (pre_norm=False only)
        self.final_norm = LayerNorm(d_model) if not pre_norm else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.residual_attn(
            x, lambda x_: self.attention(x_, x_, x_, mask, return_attention)
        )

        attn_weights = None
        if return_attention:
            attn_out, attn_weights = attn_out

        x = self.residual_ffn(attn_out, self.ffn)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if return_attention:
            return x, attn_weights
        return x
