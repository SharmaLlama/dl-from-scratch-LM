from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseMultiHeadAttention(nn.Module, ABC):
    """
    Abstract base for all attention variants.

    Subclasses implement attention_pattern() to define how scores are computed.
    Subclasses can override _apply_position_bias() to inject position info into
    Q/K before the attention pattern runs (e.g. RoPE, ALiBi).
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        dk: int,
        dv: int,
        group_sizes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dk = dk
        self.dv = dv
        self.group_sizes = group_sizes  # >1 enables grouped-query attention

        self.w_q = nn.Linear(d_model, n_heads * dk, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * dk, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * dv, bias=False)
        self.w_o = nn.Linear(n_heads * dv, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Populated during forward; read by AttentionHookManager
        self.attention_scores: Optional[torch.Tensor] = None
        self.queries: Optional[torch.Tensor] = None
        self.keys: Optional[torch.Tensor] = None

    @abstractmethod
    def attention_pattern(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention output given projected, reshaped Q/K/V.

        Args:
            Q, K, V: (B, H, N, d)
            mask: (B, 1, N, N) boolean mask — True where attention is allowed
            return_attention: whether to return the attention weight matrix

        Returns:
            output: (B, H, N, dv)
            attn_weights: (B, H, N, N) or None
        """
        ...

    def _apply_position_bias(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hook called after projection/reshape, before attention_pattern().
        Default is identity. Override to inject position information
        """
        return Q, K, V

    def _split_heads(self, x: torch.Tensor, d: int) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, d).transpose(1, 2)  # (B, H, N, d)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, dv = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dv)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        Q = self._split_heads(self.w_q(Q), self.dk)
        K = self._split_heads(self.w_k(K), self.dk)
        V = self._split_heads(self.w_v(V), self.dv)

        if self.group_sizes > 1:
            K = K.view(
                K.shape[0], self.n_heads // self.group_sizes, self.group_sizes,
                K.shape[2], self.dk
            ).mean(dim=2)
            V = V.view(
                V.shape[0], self.n_heads // self.group_sizes, self.group_sizes,
                V.shape[2], self.dv
            ).mean(dim=2)

        Q, K, V = self._apply_position_bias(Q, K, V)

        output, attn_weights = self.attention_pattern(Q, K, V, mask, return_attention)

        self.attention_scores = attn_weights
        self.queries = Q
        self.keys = K

        output = self.w_o(self._merge_heads(output))

        if return_attention:
            return output, attn_weights
        return output
