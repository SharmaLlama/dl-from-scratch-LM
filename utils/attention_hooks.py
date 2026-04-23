from typing import Optional

import torch
import torch.nn as nn

from core.attention.base import BaseMultiHeadAttention


class AttentionHookManager:
    """
    Context manager that captures attention weights, queries, and keys
    from every BaseMultiHeadAttention layer in the model.

    Usage:
        with AttentionHookManager(model) as hooks:
            _ = model(x, mask)
        attn = hooks.get_attention_weights()  # (n_layers, B, H, N, N)
    """

    def __init__(self, model: nn.Module) -> None:
        self._layers: list[BaseMultiHeadAttention] = [
            m for m in model.modules() if isinstance(m, BaseMultiHeadAttention)
        ]
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> "AttentionHookManager":
        # Hooks aren't strictly needed — forward() already writes to self.attention_scores.
        # We register post-hooks anyway so captured tensors are snapshotted (not aliased).
        def make_hook(layer: BaseMultiHeadAttention):
            def hook(module, input, output):
                if module.attention_scores is not None:
                    module.attention_scores = module.attention_scores.detach().clone()
                if module.queries is not None:
                    module.queries = module.queries.detach().clone()
                if module.keys is not None:
                    module.keys = module.keys.detach().clone()
            return hook

        for layer in self._layers:
            handle = layer.register_forward_hook(make_hook(layer))
            self._handles.append(handle)
        return self

    def __exit__(self, *args) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Returns (n_layers, B, H, N, N) stacked tensor, or None if not captured."""
        weights = [l.attention_scores for l in self._layers if l.attention_scores is not None]
        if not weights:
            return None
        return torch.stack(weights, dim=0)

    def get_queries_keys(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns Q, K each (n_layers, B, H, N, dk)."""
        qs = [l.queries for l in self._layers if l.queries is not None]
        ks = [l.keys for l in self._layers if l.keys is not None]
        Q = torch.stack(qs, dim=0) if qs else None
        K = torch.stack(ks, dim=0) if ks else None
        return Q, K

    def clear(self) -> None:
        for layer in self._layers:
            layer.attention_scores = None
            layer.queries = None
            layer.keys = None

    @property
    def n_layers(self) -> int:
        return len(self._layers)
