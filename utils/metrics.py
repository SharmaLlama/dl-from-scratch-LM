import math
from typing import Optional

import torch


def perplexity(loss: float) -> float:
    """Convert mean cross-entropy loss (nats) to perplexity."""
    return math.exp(loss)


def bits_per_byte(loss: float, avg_bytes_per_token: float) -> float:
    """
    Normalises loss to bits-per-byte, making it comparable across tokenisers.
    avg_bytes_per_token is dataset-specific; ~4.0 for BPE on English text.
    """
    return loss / (math.log(2) * avg_bytes_per_token)


def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int = -1,
) -> float:
    """Fraction of non-pad positions predicted correctly."""
    preds = logits.argmax(dim=-1)
    mask = targets != pad_token_id
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()


def compute_attention_entropy(
    attn_weights: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute per-head entropy averaged over query positions.

    Args:
        attn_weights: (..., N, N) — last two dims are (query, key)
    Returns:
        (...) — entropy per head, averaged over N query positions
    """
    # Clamp to avoid log(0)
    p = attn_weights.clamp(min=eps)
    entropy = -(p * p.log()).sum(dim=-1)   # (..., N)
    return entropy.mean(dim=-1)            # (...) mean over query positions


def compute_mean_attended_distance(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Expected |query_pos - key_pos| weighted by attention.

    Args:
        attn_weights: (B, H, N, N)
    Returns:
        (B, H) — mean attended distance per head
    """
    N = attn_weights.shape[-1]
    q_pos = torch.arange(N, device=attn_weights.device).view(1, 1, N, 1).float()
    k_pos = torch.arange(N, device=attn_weights.device).view(1, 1, 1, N).float()
    distances = (q_pos - k_pos).abs()                      # (1, 1, N, N)
    return (attn_weights * distances).sum(dim=(-2, -1)) / N  # (B, H)