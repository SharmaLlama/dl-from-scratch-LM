from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from core.layers.utils import LayerNorm
from core.model.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        layers: list[DecoderBlock],
        d_model: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        all_attn_weights: list[torch.Tensor] = []

        # Checkpointing only kicks in during training and when we're not
        # capturing attention weights (the capture path needs the live forward).
        use_ckpt = self.gradient_checkpointing and self.training and not return_attention

        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, mask, return_attention=True)
                all_attn_weights.append(attn_weights)
            elif use_ckpt:
                # use_reentrant=False is the modern path: cleaner DDP/compile
                # interop and lets autograd handle the saved-tensor bookkeeping.
                x = ckpt.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)

        x = self.norm(x)

        if return_attention:
            return x, all_attn_weights
        return x
