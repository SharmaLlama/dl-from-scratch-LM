from typing import Any

import numpy as np
import torch.nn as nn

import wandb

from training.configs.schemas import WandBConfig
from utils.metrics import compute_attention_entropy, compute_mean_attended_distance


class WandBLogger:
    """
    Thin wrapper around wandb that gates all calls on rank==0.
    Call log_metrics() every step, log_attention_stats() every N steps.
    """

    def __init__(self, cfg: WandBConfig, config_dict: dict, enabled: bool = True) -> None:
        self.enabled = enabled
        self.log_every_steps = cfg.log_every_steps
        self.log_attention_every = cfg.log_attention_every

        if enabled:
            wandb.init(
                project=cfg.project,
                name=cfg.run_name,
                entity=cfg.entity,
                tags=cfg.tags,
                config=config_dict,
            )
            # Use tokens_seen as the x-axis for all metric groups.
            # WandB still needs an integer step internally; tokens_seen becomes the display axis.
            wandb.define_metric("tokens_seen")
            for prefix in ("train/*", "val/*", "attn/*"):
                wandb.define_metric(prefix, step_metric="tokens_seen")

    def watch_model(self, model: nn.Module) -> None:
        if self.enabled:
            wandb.watch(model, log="gradients", log_freq=100)

    def log_metrics(self, metrics: dict[str, Any], step: int, tokens_seen: int = 0) -> None:
        if self.enabled:
            wandb.log({"tokens_seen": tokens_seen, **metrics}, step=step)

    def log_attention_stats(
        self,
        attn_weights: np.ndarray,  # (n_layers, B, n_heads, N, N)
        step: int,
        tokens_seen: int = 0,
    ) -> None:
        """
        Log per-head entropy and mean attended distance for all layers.
        attn_weights should come from AttentionHookManager.get_attention_weights().numpy()
        """
        if not self.enabled:
            return

        import torch
        weights_t = torch.from_numpy(attn_weights).float()  # (L, B, H, N, N)
        n_layers, _, n_heads, _, _ = weights_t.shape

        metrics: dict[str, Any] = {}

        for layer_idx in range(n_layers):
            layer_weights = weights_t[layer_idx]  # (B, H, N, N)

            # Entropy per head: (B, H) → mean over batch → (H,)
            entropy = compute_attention_entropy(layer_weights).mean(dim=0)  # (H,)
            dist = compute_mean_attended_distance(layer_weights).mean(dim=0)  # (H,)

            for head_idx in range(n_heads):
                metrics[f"attn/entropy_L{layer_idx}_H{head_idx}"] = entropy[head_idx].item()

                # Sparsity: fraction of weights below threshold
                sparsity = (layer_weights[:, head_idx] < 0.01).float().mean().item()
                metrics[f"attn/sparsity_L{layer_idx}_H{head_idx}"] = sparsity

            metrics[f"attn/mean_distance_L{layer_idx}"] = dist.mean().item()

        # Aggregate stats
        all_entropy = compute_attention_entropy(weights_t.view(-1, *weights_t.shape[2:]))
        metrics["attn/entropy_mean"] = all_entropy.mean().item()
        metrics["attn/entropy_std"] = all_entropy.std().item()

        wandb.log({"tokens_seen": tokens_seen, **metrics}, step=step)

    def log_image(self, key: str, figure, step: int) -> None:
        if self.enabled:
            wandb.log({key: wandb.Image(figure)}, step=step)

    def log_table(self, key: str, dataframe, step: int) -> None:
        if self.enabled:
            wandb.log({key: wandb.Table(dataframe=dataframe)}, step=step)

    def update_summary(self, key: str, value: Any) -> None:
        if self.enabled:
            wandb.run.summary[key] = value  # type: ignore[union-attr]

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()
