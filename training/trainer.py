import itertools
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from training.checkpointing import CheckpointManager
from training.configs.schemas import TrainingConfig
from training.metrics_tracker import MetricsTracker
from training.wandb_logger import WandBLogger
from utils.attention_hooks import AttentionHookManager
from utils.benchmark import (
    calculate_mfu,
    get_memory_stats,
    get_throttle_reasons,
    init_nvml_handle,
    measure_inter_gpu_bandwidth,
)
from utils.metrics import perplexity
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Metrics all-reduced across DDP ranks and logged as a single global value.
_SHARED_KEYS = ["train/loss", "train/perplexity", "train/grad_norm", "train/lr"]

# Metrics tracked per GPU — gathered to rank 0 and logged with a _rankN suffix.
_HW_KEYS = ["hw/tokens_per_second", "hw/mfu"]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig,
        logger: WandBLogger,
        checkpointer: CheckpointManager,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.checkpointer = checkpointer
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing,
            ignore_index=-1,
        )
        self.global_step = 0
        self.tokens_seen = 0
        self.train_time = 0.0         # cumulative training wall time (excludes eval)
        self._next_eval = cfg.eval_interval
        self._next_ckpt = cfg.checkpoint_interval
        self._accum_step = 0          # micro-batch counter within accumulation window
        self._accum_time = 0.0        # wall time for the current optimizer-step window (per GPU)
        self._accum_tokens = 0        # tokens for the current optimizer-step window (per GPU)

        self.scaler = GradScaler('cuda', enabled=cfg.mixed_precision)

        self._num_params = sum(p.numel() for p in model.parameters())
        self._nvml_handle = init_nvml_handle(rank) if device.type == "cuda" else None

        # Rolling-average trackers — window = log_every_steps optimizer steps.
        self._shared_tracker = MetricsTracker(_SHARED_KEYS)
        self._hw_tracker = MetricsTracker(_HW_KEYS)

    def fit(self) -> None:
        """
        Train until the token budget (cfg.max_tokens) is exhausted.
        Iterates through the dataloader repeatedly if necessary.
        Evals and checkpoints are triggered by token count, not epochs.
        """
        self.model.train()

        for x, y in itertools.cycle(self.train_loader):
            t_step = time.perf_counter()
            if self.tokens_seen >= self.cfg.max_tokens:
                break

            x, y = x.to(self.device), y.to(self.device)
            tokens_this_batch = x.numel()
            mask = self._causal_mask(x.shape[1], x.shape[0])

            # ── forward + backward (micro-batch) ──────────────────────────────
            with torch.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
                logits = self.model(x, mask)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / self.cfg.grad_accum_steps

            self.scaler.scale(loss).backward()
            self._accum_step += 1

            # Accumulate timing for every micro-batch (optimizer step or not).
            torch.cuda.synchronize()
            step_time = time.perf_counter() - t_step
            self._accum_time += step_time
            self._accum_tokens += tokens_this_batch
            self.tokens_seen += tokens_this_batch
            self.train_time += step_time

            # ── optimizer step ────────────────────────────────────────────────
            if self._accum_step % self.cfg.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                loss_for_log = loss.item() * self.cfg.grad_accum_steps
                tok_s = self._accum_tokens / self._accum_time if self._accum_time > 0 else 0.0
                mfu   = calculate_mfu(self._num_params, self._accum_tokens, self._accum_time)

                self._shared_tracker.update({
                    "train/loss":       loss_for_log,
                    "train/perplexity": perplexity(loss_for_log),
                    "train/grad_norm":  grad_norm.item(),
                    "train/lr":         self.scheduler.get_last_lr()[0],
                })
                self._hw_tracker.update({
                    "hw/tokens_per_second": tok_s,
                    "hw/mfu":               mfu,
                })

                # Reset per-optimizer-step window (tok/s and mfu are per-step, not cumulative).
                self._accum_time   = 0.0
                self._accum_tokens = 0

                logger.info(
                    f"step={self.global_step} tokens={self.tokens_seen:,} "
                    f"loss={loss_for_log:.4f} ppl={perplexity(loss_for_log):.2f} "
                    f"tok/s={tok_s:.0f}"
                )

                # ── WandB logging (every log_every_steps optimizer steps) ─────
                if self.global_step % self.logger.log_every_steps == 0:
                    # All-reduce loss/ppl/grad_norm across ranks; lr is identical on all ranks.
                    self._shared_tracker.all_reduce(
                        ["train/loss", "train/perplexity", "train/grad_norm"],
                        self.world_size,
                        device=self.device,
                    )
                    shared = self._shared_tracker.compute()

                    # Gather per-GPU hw metrics and VRAM snapshots to rank 0.
                    hw = self._gather_per_gpu_metrics()

                    throttle = get_throttle_reasons(self._nvml_handle)
                    if throttle:
                        logger.warning(f"GPU throttling active: {throttle}")

                    self.logger.log_metrics(
                        {**shared, **hw},
                        step=self.global_step,
                        tokens_seen=self.tokens_seen,
                    )
                    self._shared_tracker.reset()
                    self._hw_tracker.reset()

                # ── Inter-GPU bandwidth (10x less frequent) ───────────────────
                if self.global_step % (self.logger.log_every_steps * 10) == 0:
                    bw = measure_inter_gpu_bandwidth(self.rank, self.world_size)
                    if bw > 0:
                        self.logger.log_metrics(
                            {"hw/inter_gpu_bandwidth_gbs": bw},
                            step=self.global_step,
                            tokens_seen=self.tokens_seen,
                        )

            # ── Eval + checkpoint triggers ────────────────────────────────────
            if self.tokens_seen >= self._next_eval:
                val_loss = self._eval()
                self._next_eval += self.cfg.eval_interval
                self.model.train()

                if self.tokens_seen >= self._next_ckpt:
                    self.checkpointer.save(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, val_loss,
                        {"tokens_seen": self.tokens_seen},
                    )
                    self._next_ckpt += self.cfg.checkpoint_interval

        # Final eval and checkpoint at end of training.
        self._eval()
        self.checkpointer.save(
            self.model, self.optimizer, self.scheduler,
            self.global_step, float("inf"),
            {"tokens_seen": self.tokens_seen},
        )
        self.logger.finish()

    def _gather_per_gpu_metrics(self) -> dict[str, float]:
        """
        Collect hw tracker averages + VRAM snapshots from all ranks onto rank 0.
        Returns a flat dict with keys suffixed by _rankN (e.g. hw/mfu_rank0).
        """
        local_hw  = self._hw_tracker.compute()
        local_mem = get_memory_stats(self.rank) if self.device.type == "cuda" else {}
        local: dict[str, float] = {**local_hw, **local_mem}

        if self.world_size <= 1:
            return {f"{k}_rank{self.rank}": v for k, v in local.items()}

        keys    = sorted(local.keys())
        local_t = torch.tensor(
            [local[k] for k in keys], dtype=torch.float32, device=self.device
        )
        gathered = [torch.zeros_like(local_t) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, local_t)

        result: dict[str, float] = {}
        for rank_idx, t in enumerate(gathered):
            for i, k in enumerate(keys):
                result[f"{k}_rank{rank_idx}"] = t[i].item()
        return result

    @torch.inference_mode()
    def _eval(self) -> float:
        self.model.eval()
        total_loss   = 0.0
        total_tokens = 0

        log_attn       = (self.global_step % self.logger.log_attention_every == 0)
        attn_weights_np = None
        hook_manager   = AttentionHookManager(self.model)

        eval_tokens_seen = 0
        for batch_idx, (x, y) in enumerate(self.val_loader):
            if eval_tokens_seen >= self.cfg.max_eval_tokens:
                break

            x, y = x.to(self.device), y.to(self.device)
            eval_tokens_seen += x.numel()
            mask = self._causal_mask(x.shape[1], x.shape[0])

            if log_attn and batch_idx == 0:
                with hook_manager:
                    logits = self.model(x, mask)
                captured = hook_manager.get_attention_weights()
                if captured is not None:
                    attn_weights_np = captured[:, :1].cpu().numpy()
            else:
                logits = self.model(x, mask)

            loss          = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_tokens += y.numel()
            total_loss   += loss.item() * y.numel()

        mean_loss = total_loss / total_tokens
        metrics = {
            "val/loss":       mean_loss,
            "val/perplexity": perplexity(mean_loss),
        }
        self.logger.log_metrics(metrics, step=self.global_step, tokens_seen=self.tokens_seen)
        self.logger.update_summary("best_val_perplexity", perplexity(mean_loss))

        if attn_weights_np is not None:
            self.logger.log_attention_stats(
                attn_weights_np, step=self.global_step, tokens_seen=self.tokens_seen
            )

        return mean_loss

    def _causal_mask(self, seq_len: int, batch_size: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
