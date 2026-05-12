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
        attn = hooks.get_attention_weights()  # (n_layers, batch_slice, H, N, N)

    By default we only keep the first `batch_slice` entries along the batch dim
    and immediately offload to CPU. The full (B, H, N, N) tensor across L layers
    is huge (multi-GB at production batch sizes) and downstream consumers only
    ever look at the first batch element, so capturing all of it would OOM.
    """

    def __init__(self, model: nn.Module, batch_slice: int = 1) -> None:
        self._layers: list[BaseMultiHeadAttention] = [
            m for m in model.modules() if isinstance(m, BaseMultiHeadAttention)
        ]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._attn: list[Optional[torch.Tensor]] = [None] * len(self._layers)
        self._q: list[Optional[torch.Tensor]] = [None] * len(self._layers)
        self._k: list[Optional[torch.Tensor]] = [None] * len(self._layers)
        self._batch_slice = batch_slice

    def __enter__(self) -> "AttentionHookManager":
        # Reset captures — fresh state per context entry.
        n = len(self._layers)
        self._attn = [None] * n
        self._q = [None] * n
        self._k = [None] * n

        bs = self._batch_slice

        def make_hook(idx: int):
            def hook(module: BaseMultiHeadAttention, _inp, _out):
                # Slice along batch BEFORE the device transfer so we don't ship
                # the full (B, H, N, N) tensor across PCIe. .to("cpu", non_blocking=True)
                # creates an independent CPU copy, removing all VRAM pressure.
                if module.attention_scores is not None:
                    self._attn[idx] = module.attention_scores[:bs].detach().to(
                        "cpu", non_blocking=True
                    )
                if module.queries is not None:
                    self._q[idx] = module.queries[:bs].detach().to(
                        "cpu", non_blocking=True
                    )
                if module.keys is not None:
                    self._k[idx] = module.keys[:bs].detach().to(
                        "cpu", non_blocking=True
                    )
                # Drop refs on the layer immediately; manager already holds the snapshot.
                module.attention_scores = None
                module.queries = None
                module.keys = None
            return hook

        for idx, layer in enumerate(self._layers):
            layer._capture = True
            handle = layer.register_forward_hook(make_hook(idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *args) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer in self._layers:
            layer._capture = False
            layer.attention_scores = None
            layer.queries = None
            layer.keys = None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Returns (n_layers, B, H, N, N) stacked tensor, or None if not captured."""
        weights = [a for a in self._attn if a is not None]
        if not weights:
            return None
        return torch.stack(weights, dim=0)

    def get_queries_keys(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns Q, K each (n_layers, B, H, N, dk)."""
        qs = [q for q in self._q if q is not None]
        ks = [k for k in self._k if k is not None]
        Q = torch.stack(qs, dim=0) if qs else None
        K = torch.stack(ks, dim=0) if ks else None
        return Q, K

    def clear(self) -> None:
        n = len(self._layers)
        self._attn = [None] * n
        self._q = [None] * n
        self._k = [None] * n

    @property
    def n_layers(self) -> int:
        return len(self._layers)
