from typing import Optional

import torch
import torch.nn as nn

from core.layers.utils import LayerNorm
from core.model.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, layers: list[DecoderBlock], d_model: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        all_attn_weights: list[torch.Tensor] = []

        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, mask, return_attention=True)
                all_attn_weights.append(attn_weights)
            else:
                x = layer(x, mask)

        x = self.norm(x)

        if return_attention:
            return x, all_attn_weights
        return x
