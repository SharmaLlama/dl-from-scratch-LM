from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers.utils import Projection
from core.model.decoder import Decoder
from core.positional_encoding.base import BasePositionalEncoding


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding: BasePositionalEncoding,
        decoder: Decoder,
        projection: Projection,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.projection = projection

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embedding(x)

        if return_attention:
            x, attn_weights = self.decoder(x, mask, return_attention=True)
            return self.projection(x), attn_weights

        x = self.decoder(x, mask)
        return self.projection(x)

    @torch.inference_mode()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            seq_len = prompt.shape[1]
            mask = self._causal_mask(seq_len, prompt.device)
            logits = self.forward(prompt, mask)[:, -1, :]  # (B, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat([prompt, next_token], dim=1)

        return prompt

    def initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True = position is allowed to attend
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool().unsqueeze(0).unsqueeze(0)
