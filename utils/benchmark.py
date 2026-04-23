"""
GPU benchmarking: forward-pass latency, memory usage, and throughput
across a grid of batch sizes and sequence lengths.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    batch_size: int
    seq_len: int
    mean_latency_ms: float
    std_latency_ms: float
    peak_memory_mb: float
    tokens_per_second: float
    device: str


class ModelBenchmarker:
    """
    Runs a grid of (batch_size, seq_len) benchmarks and exports results to CSV.

    Usage:
        benchmarker = ModelBenchmarker(model, device, vocab_size=50257)
        results = benchmarker.run(batch_sizes=[4, 8, 16], seq_lens=[512, 1024, 2048])
        ModelBenchmarker.to_csv(results, "experiments/vanilla/benchmark.csv")
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        vocab_size: int,
        n_warmup: int = 5,
        n_iters: int = 20,
    ) -> None:
        self.model = model
        self.device = device
        self.vocab_size = vocab_size
        self.n_warmup = n_warmup
        self.n_iters = n_iters

    def run(
        self,
        batch_sizes: list[int],
        seq_lens: list[int],
    ) -> list[BenchmarkResult]:
        results = []
        self.model.eval()

        for bs in batch_sizes:
            for sl in seq_lens:
                result = self._benchmark_single(bs, sl)
                if result is not None:
                    results.append(result)
                    logger.info(
                        f"bs={bs} sl={sl}: {result.mean_latency_ms:.1f}ms ± {result.std_latency_ms:.1f}ms, "
                        f"{result.tokens_per_second:.0f} tok/s, "
                        f"{result.peak_memory_mb:.0f} MB"
                    )

        return results

    def _benchmark_single(
        self, batch_size: int, seq_len: int
    ) -> Optional[BenchmarkResult]:
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=self.device))
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        is_cuda = self.device.type == "cuda"

        try:
            # Warmup
            with torch.inference_mode():
                for _ in range(self.n_warmup):
                    _ = self.model(x, mask)
                    if is_cuda:
                        torch.cuda.synchronize()

            if is_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)

            latencies_ms: list[float] = []

            with torch.inference_mode():
                for _ in range(self.n_iters):
                    if is_cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = self.model(x, mask)
                    if is_cuda:
                        torch.cuda.synchronize()
                    latencies_ms.append((time.perf_counter() - t0) * 1000)

            latency_tensor = torch.tensor(latencies_ms)
            mean_ms = latency_tensor.mean().item()
            std_ms = latency_tensor.std().item()
            tokens_per_sec = (batch_size * seq_len) / (mean_ms / 1000)

            peak_mb = 0.0
            if is_cuda:
                peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2

            return BenchmarkResult(
                batch_size=batch_size,
                seq_len=seq_len,
                mean_latency_ms=round(mean_ms, 3),
                std_latency_ms=round(std_ms, 3),
                peak_memory_mb=round(peak_mb, 1),
                tokens_per_second=round(tokens_per_sec, 1),
                device=str(self.device),
            )

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at batch_size={batch_size}, seq_len={seq_len} — skipping.")
            if is_cuda:
                torch.cuda.empty_cache()
            return None

    @staticmethod
    def to_csv(results: list[BenchmarkResult], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
