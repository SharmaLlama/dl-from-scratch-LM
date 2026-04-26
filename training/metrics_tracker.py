from __future__ import annotations

import torch


class MetricsTracker:
    """
    Rolling-average accumulator for training metrics.

    Typical usage:
        tracker.update({"train/loss": 2.4, "train/grad_norm": 0.8})  # every optimizer step
        averaged = tracker.compute()                                   # at log time
        tracker.reset()                                                # after logging

    For DDP — call all_reduce() between compute() and reset() for metrics
    that should be globally averaged across ranks. Per-GPU metrics (hw/*)
    skip all_reduce and are gathered separately by the caller.
    """

    def __init__(self, keys: list[str]) -> None:
        self._keys = list(keys)
        self._sums: dict[str, float] = {k: 0.0 for k in keys}
        self._counts: dict[str, int] = {k: 0 for k in keys}

    def update(self, metrics: dict[str, float]) -> None:
        """Add values to running sums. Unknown keys are silently ignored."""
        for k, v in metrics.items():
            if k in self._sums:
                self._sums[k] += v
                self._counts[k] += 1

    def compute(self) -> dict[str, float]:
        """Return the mean of accumulated values since last reset."""
        return {
            k: self._sums[k] / self._counts[k] if self._counts[k] > 0 else 0.0
            for k in self._keys
        }

    def all_reduce(
        self,
        keys: list[str],
        world_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Average the given keys across all DDP ranks in-place.
        Must be called on every rank simultaneously (collective op).
        No-op when world_size <= 1 or keys is empty.
        `device` must be the CUDA device for this rank — NCCL requires GPU tensors.
        """
        if world_size <= 1 or not keys:
            return
        t = torch.tensor([self._sums[k] for k in keys], dtype=torch.float64, device=device)
        c = torch.tensor([float(self._counts[k]) for k in keys], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t)
        torch.distributed.all_reduce(c)
        for i, k in enumerate(keys):
            self._sums[k] = t[i].item()
            self._counts[k] = int(c[i].item())

    def reset(self) -> None:
        """Clear all accumulators. Call immediately after logging."""
        for k in self._keys:
            self._sums[k] = 0.0
            self._counts[k] = 0
