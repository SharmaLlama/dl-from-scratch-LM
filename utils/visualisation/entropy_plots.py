"""
Static matplotlib plots for attention entropy and head importance.
Used in analysis notebooks and logged to WandB as images.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import compute_attention_entropy


def plot_attention_entropy(
    attention_weights: np.ndarray,
    layer_names: Optional[list[str]] = None,
    title: str = "Attention entropy per head",
) -> plt.Figure:
    """
    Heatmap of mean attention entropy per (layer, head).
    High entropy = diffuse attention (attends everywhere).
    Low entropy = peaked attention (attends to few tokens).

    Args:
        attention_weights: (n_layers, n_heads, N, N)
        layer_names: optional list of n_layers labels for the y-axis
        title: plot title

    Returns:
        matplotlib Figure
    """
    import torch

    n_layers, n_heads, N, _ = attention_weights.shape
    weights_t = torch.from_numpy(attention_weights).float()  # (L, H, N, N)

    # Compute entropy per (layer, head) averaged over the sequence
    entropy_grid = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        # compute_attention_entropy expects (..., N, N); returns (...) averaged over N
        layer_entropy = compute_attention_entropy(weights_t[l])  # (H,)
        entropy_grid[l] = layer_entropy.numpy()

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.8), max(4, n_layers * 0.6)))
    im = ax.imshow(entropy_grid, aspect="auto", cmap="RdYlGn", vmin=0)

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_names if layer_names else [f"L{l}" for l in range(n_layers)], fontsize=9)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Mean entropy (nats)", fontsize=8)

    # Annotate each cell with its entropy value
    for l in range(n_layers):
        for h in range(n_heads):
            ax.text(h, l, f"{entropy_grid[l, h]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    return fig


def plot_head_importance_bar(
    ablation_df: pd.DataFrame,
    top_n: Optional[int] = None,
    title: str = "Head importance (perplexity increase on removal)",
) -> plt.Figure:
    """
    Horizontal bar chart of perplexity increase when each head is removed.
    Sorted descending — most important head at the top.

    Args:
        ablation_df: DataFrame with columns [layer_idx, head_idx, perplexity_increase]
        top_n: show only the top N most important heads (None = all)
        title: plot title

    Returns:
        matplotlib Figure
    """
    df = ablation_df.copy()
    df["label"] = df.apply(lambda r: f"L{int(r['layer_idx'])}H{int(r['head_idx'])}", axis=1)
    df = df.sort_values("perplexity_increase", ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    colors = ["#e94560" if v > 0 else "#4ecca3" for v in df["perplexity_increase"]]
    ax.barh(df["label"], df["perplexity_increase"], color=colors)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Perplexity increase (Δ PPL)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mean_attended_distance(
    attention_weights: np.ndarray,
    layer_names: Optional[list[str]] = None,
    title: str = "Mean attended distance per layer",
) -> plt.Figure:
    """
    Line plot of mean |query_pos - key_pos| per layer, averaged over heads.
    Reveals whether each layer attends locally or globally.

    Args:
        attention_weights: (n_layers, n_heads, N, N)
        layer_names: optional layer labels for x-axis

    Returns:
        matplotlib Figure
    """
    import torch
    from utils.metrics import compute_mean_attended_distance

    n_layers, n_heads, N, _ = attention_weights.shape
    weights_t = torch.from_numpy(attention_weights).float()

    mean_distances = []
    for l in range(n_layers):
        dist = compute_mean_attended_distance(weights_t[l].unsqueeze(0))  # (1, H)
        mean_distances.append(dist.squeeze(0).numpy())  # (H,)

    mean_distances = np.array(mean_distances)  # (L, H)

    fig, ax = plt.subplots(figsize=(max(6, n_layers * 0.8), 4))
    for h in range(n_heads):
        ax.plot(range(n_layers), mean_distances[:, h], marker="o", label=f"H{h}", alpha=0.7)

    ax.plot(range(n_layers), mean_distances.mean(axis=1), color="black",
            linewidth=2, linestyle="--", label="Mean", zorder=10)

    x_labels = layer_names if layer_names else [f"L{l}" for l in range(n_layers)]
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |q_pos − k_pos|")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=max(1, n_heads // 4))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
