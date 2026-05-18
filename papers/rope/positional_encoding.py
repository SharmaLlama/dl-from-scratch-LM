import torch
import torch.nn as nn

from core.positional_encoding.base import BasePositionalEncoding


class RoPEPositionalEncoding(BasePositionalEncoding):
    """
    Token embedding for RoPE-attention models.

    RoPE injects position information by rotating Q and K inside the attention
    block, not via an additive PE on the embeddings. So this class only handles
    the token embedding (inherited from BasePositionalEncoding) and returns None
    from _get_position_encoding.

    The actual rotation lives in RotaryEmbedding, constructed by
    RoPEMultiHeadAttention with the per-head rotation dim.
    """

    def _get_position_encoding(self, seq_len: int, device: torch.device):
        return None


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding (Su et al. 2021).

    Applies a position-dependent rotation to pairs of dimensions in Q/K so the
    dot product Q_m · K_n depends only on the relative offset (n - m). Rotation
    is over the *last* dim (head_dim).

    Uses the half-rotation form (LLaMA / HF). For position m and the i-th pair
    (dim i and dim i + head_dim/2):
        x'_i              =  x_i cos(m·θ_i) - x_{i+d/2} sin(m·θ_i)
        x'_{i+head_dim/2} =  x_{i+d/2} cos(m·θ_i) + x_i sin(m·θ_i)

    Equivalent to:
        x' = x * cos + rotate_half(x) * sin
    where sin/cos are tiled along the last dim as [s0..s_{d/2-1}, s0..s_{d/2-1}]
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.base = float(base)

        # θ_i = base^(-2i/d) for i = 0, 1, ..., head_dim/2 - 1
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        # angles[t, i] = t · inv_freq[i]   → shape (seq_len, head_dim / 2)
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq)

        sin = torch.cat([angles.sin(), angles.sin()], dim=-1)  # (seq_len, head_dim)
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)
        self._cached_seq_len = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to x with shape (..., seq_len, head_dim).
        Typically called separately on Q and on K, each shape (B, H, N, dk).
        """
        seq_len = x.size(-2)
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)

        sin = self.sin_cached[:seq_len].to(dtype=x.dtype)
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype)
        return x * cos + self._rotate_half(x) * sin
