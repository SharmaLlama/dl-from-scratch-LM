"""
GPU benchmarking: forward-pass latency, memory usage, and throughput
across a grid of batch sizes and sequence lengths.

Also provides lightweight hardware monitoring functions (NVML-based memory,
throttle detection, MFU, inter-GPU bandwidth) intended to be called inside
the training loop.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

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



PEAK_FLOPS_DEFAULT = 25e12


def init_nvml_handle(rank: int) -> Optional[Any]:
    """
    Initialize NVML and return a device handle for `rank`.
    Returns None if pynvml is not installed or NVML init fails.
    """
    if not _NVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(rank)
    except Exception:
        return None


def get_memory_stats(rank: int) -> dict[str, float]:
    """VRAM allocation, reservation, fragmentation, and inactive splits for `rank`."""
    stats = torch.cuda.memory_stats(rank)
    allocated = torch.cuda.memory_allocated(rank)
    reserved = torch.cuda.memory_reserved(rank)
    return {
        f"hw/vram_allocated_gb": allocated / 1e9,
        f"hw/vram_reserved_gb": reserved / 1e9,
        f"hw/fragmentation_b": (reserved - allocated) / reserved if reserved > 0 else 0.0,
        f"hw/inactive_split_gb": stats.get("inactive_split_bytes.all.current", 0) / 1e9,
    }


def get_throttle_reasons(handle: Optional[Any]) -> list[str]:
    """Returns active GPU throttle reason strings, or [] if none / NVML unavailable."""
    if handle is None or not _NVML_AVAILABLE:
        return []
    try:
        reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        reason_map = {0x01: "Power", 0x02: "Thermal", 0x04: "Voltage", 0x08: "OS", 0x10: "Sync"}
        active = [v for k, v in reason_map.items() if reasons & k]
        return active if active else []
    except Exception:
        return []


def calculate_mfu(
    num_params: int,
    tokens_per_step: int,
    step_time: float,
    peak_flops: float = PEAK_FLOPS_DEFAULT,
) -> float:
    """
    Model FLOPs Utilization (MFU).
    Uses the 6·P·T approximation for a forward+backward pass.
    Returns 0.0 if step_time is zero.
    """
    if step_time <= 0:
        return 0.0
    flops_achieved = (6 * num_params * tokens_per_step) / step_time
    return flops_achieved / peak_flops


def measure_inter_gpu_bandwidth(rank: int, world_size: int, tensor_size_mb: int = 100) -> float:
    """
    Measures effective All-Reduce bandwidth in GB/s.
    Returns 0.0 on single-GPU setups.
    """
    if world_size < 2:
        return 0.0
    num_el = (tensor_size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_el, device=f"cuda:{rank}")

    # warmup
    torch.distributed.all_reduce(tensor)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        torch.distributed.all_reduce(tensor)
    torch.cuda.synchronize()

    avg_time = (time.perf_counter() - start) / 10
    return (tensor_size_mb / 1024) / avg_time  # GB/s
