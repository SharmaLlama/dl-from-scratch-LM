import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BasePositionalEncoding(nn.Module, ABC):
    """
    Abstract base for all positional encoding schemes.

    Handles token embedding and dropout. Subclasses implement
    _get_position_encoding() to return an additive positional tensor,
    or None when position is handled inside attention (RoPE, ALiBi, NoPE).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def _get_position_encoding(
        self, seq_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Return (seq_len, d_model) additive tensor, or None."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) token ids → (B, N, d_model)
        embeddings = self.token_embedding(x) * math.sqrt(self.d_model)
        pe = self._get_position_encoding(x.shape[1], x.device)
        if pe is not None:
            embeddings = embeddings + pe.unsqueeze(0)
        return self.dropout(embeddings)

    def update_max_seq_len(self, new_max: int) -> None:
        self.max_seq_len = new_max


class SinusoidalPE(BasePositionalEncoding):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__(vocab_size, d_model, max_seq_len, dropout)
        self._build_buffer(max_seq_len)

    def _build_buffer(self, max_seq_len: int) -> None:
        pe = torch.zeros(max_seq_len, self.d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def _get_position_encoding(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            self.update_max_seq_len(seq_len)
        return self.pe[:seq_len]  # type: ignore[attr-defined]

    def update_max_seq_len(self, new_max: int) -> None:
        super().update_max_seq_len(new_max)
        self._build_buffer(new_max)


class LearnedPE(BasePositionalEncoding):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__(vocab_size, d_model, max_seq_len, dropout)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def _get_position_encoding(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.pos_embedding(positions)
