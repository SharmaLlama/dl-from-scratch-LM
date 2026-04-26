"""
Head ablation study: iteratively zero out attention heads and measure
perplexity impact to rank head importance.

Ablation mechanic: temporarily zero the w_o weight columns that correspond
to the target head, run inference, then restore. This avoids modifying the
forward pass and works regardless of attention variant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.attention.base import BaseMultiHeadAttention
from utils.metrics import perplexity


class HeadAblationStudy:
    """
    Greedy head importance analysis via perplexity-based ablation.

    The head whose removal causes the smallest perplexity increase is
    considered least important and is removed first in greedy mode.
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        output_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.output_path = Path(output_path) if output_path else None

        self._attn_layers: list[BaseMultiHeadAttention] = [
            m for m in model.modules() if isinstance(m, BaseMultiHeadAttention)
        ]
        if not self._attn_layers:
            raise ValueError("No BaseMultiHeadAttention modules found in model.")

        self._n_heads = self._attn_layers[0].n_heads
        self._dv = self._attn_layers[0].dv


    def head_importance_scores(self) -> pd.DataFrame:
        """
        Ablate each (layer, head) independently and measure perplexity increase.
        Returns a DataFrame sorted ascending by perplexity_increase
        (least important head first).
        """
        baseline = self._eval_perplexity([])
        records = []

        for layer_idx in range(len(self._attn_layers)):
            for head_idx in range(self._n_heads):
                ppl = self._eval_perplexity([(layer_idx, head_idx)])
                records.append(
                    {
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "perplexity_after_removal": ppl,
                        "perplexity_increase": ppl - baseline,
                    }
                )

        df = pd.DataFrame(records).sort_values("perplexity_increase").reset_index(drop=True)
        self._save_results(df.to_dict("records"))
        return df

    def greedy_removal(
        self,
        stopping_perplexity: float = float("inf"),
    ) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Greedily remove the least important head at each step until
        perplexity exceeds stopping_perplexity or all heads are removed.

        Returns (removal_order, perplexity_after_each_removal).
        Resumable: loads prior results from output_path if it exists.
        """
        removal_order: list[tuple[int, int]] = []
        perplexities: list[float] = []

        # Resume from prior results
        if self.output_path and self.output_path.exists():
            prior = json.loads(self.output_path.read_text())
            removal_order = [tuple(r["removed"]) for r in prior]  # type: ignore[misc]
            perplexities = [r["perplexity"] for r in prior]

        all_heads = [
            (l, h)
            for l in range(len(self._attn_layers))
            for h in range(self._n_heads)
        ]
        remaining = [h for h in all_heads if h not in removal_order]

        while remaining:
            best_head = None
            best_ppl = float("inf")

            for candidate in remaining:
                ppl = self._eval_perplexity(removal_order + [candidate])
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_head = candidate

            removal_order.append(best_head)
            perplexities.append(best_ppl)
            remaining.remove(best_head)

            self._save_results(
                [{"removed": list(h), "perplexity": p} for h, p in zip(removal_order, perplexities)]
            )

            if best_ppl > stopping_perplexity:
                break

        return removal_order, perplexities


    @torch.inference_mode()
    def _eval_perplexity(self, ablated: list[tuple[int, int]]) -> float:
        """Run validation loop with the specified (layer, head) pairs zeroed."""
        saved = self._zero_heads(ablated)
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        loss_fn = nn.CrossEntropyLoss()

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            seq_len = x.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool().unsqueeze(0).unsqueeze(0)
            logits = self.model(x, mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_tokens += y.numel()
            total_loss += loss.item() * y.numel()

        self._restore_heads(saved)
        return perplexity(total_loss / total_tokens)

    def _zero_heads(
        self, ablated: list[tuple[int, int]]
    ) -> dict[tuple[int, int], torch.Tensor]:
        """
        Zero the w_o columns corresponding to each ablated head.
        Returns saved weights for restoration.
        """
        saved: dict[tuple[int, int], torch.Tensor] = {}
        for layer_idx, head_idx in ablated:
            layer = self._attn_layers[layer_idx]
            col_start = head_idx * self._dv
            col_end = (head_idx + 1) * self._dv
            saved[(layer_idx, head_idx)] = layer.w_o.weight.data[:, col_start:col_end].clone()
            layer.w_o.weight.data[:, col_start:col_end] = 0.0
        return saved

    def _restore_heads(self, saved: dict[tuple[int, int], torch.Tensor]) -> None:
        for (layer_idx, head_idx), weights in saved.items():
            layer = self._attn_layers[layer_idx]
            col_start = head_idx * self._dv
            col_end = (head_idx + 1) * self._dv
            layer.w_o.weight.data[:, col_start:col_end] = weights

    def _save_results(self, results: list[dict]) -> None:
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps(results, indent=2))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ablation_curve(
    removal_order: list[tuple[int, int]],
    perplexities: list[float],
) -> plt.Figure:
    """Perplexity vs number of heads removed."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(perplexities) + 1), perplexities, marker="o", linewidth=2)
    ax.set_xlabel("Heads removed")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Greedy head removal — perplexity curve")

    # Annotate every 5th removed head with its (layer, head) label
    for i, (layer_idx, head_idx) in enumerate(removal_order):
        if i % 5 == 0:
            ax.annotate(
                f"L{layer_idx}H{head_idx}",
                (i + 1, perplexities[i]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=7,
                ha="center",
            )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
