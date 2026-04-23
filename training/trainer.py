import itertools
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from training.checkpointing import CheckpointManager
from training.configs.schemas import TrainingConfig
from training.wandb_logger import WandBLogger
from utils.attention_hooks import AttentionHookManager
from utils.metrics import perplexity
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing,
            ignore_index=-1,
        )
        self.global_step = 0
        self.tokens_seen = 0          # total tokens consumed so far
        self._next_eval = cfg.eval_interval
        self._next_ckpt = cfg.checkpoint_interval

    def fit(self) -> None:
        """
        Train until the token budget (cfg.max_tokens) is exhausted.
        Iterates through the dataloader repeatedly if necessary.
        Evals and checkpoints are triggered by token count, not epochs.
        """
        t0 = time.perf_counter()
        self.model.train()

        # itertools.cycle means we never crash if the dataset is smaller than max_tokens
        for x, y in itertools.cycle(self.train_loader):
            if self.tokens_seen >= self.cfg.max_tokens:
                break

            x, y = x.to(self.device), y.to(self.device)
            tokens_this_batch = x.numel()
            mask = self._causal_mask(x.shape[1], x.shape[0])

            logits = self.model(x, mask)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            self.tokens_seen += tokens_this_batch
            self.global_step += 1

            elapsed = time.perf_counter() - t0
            tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0.0

            logger.info(
                f"step={self.global_step} tokens={self.tokens_seen:,} "
                f"loss={loss.item():.4f} ppl={perplexity(loss.item()):.2f} "
                f"tok/s={tokens_per_sec:.0f}"
            )

            if self.global_step % self.logger.log_every_steps == 0:
                self.logger.log_metrics(
                    {
                        "train/loss": loss.item(),
                        "train/perplexity": perplexity(loss.item()),
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/tokens_per_second": tokens_per_sec,
                    },
                    step=self.global_step,
                    tokens_seen=self.tokens_seen,
                )

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

        # Final eval and checkpoint at end of training
        self._eval()
        self.checkpointer.save(
            self.model, self.optimizer, self.scheduler,
            self.global_step, float("inf"),
            {"tokens_seen": self.tokens_seen},
        )
        self.logger.finish()

    @torch.inference_mode()
    def _eval(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        log_attn = (self.global_step % self.logger.log_attention_every == 0)
        attn_weights_np = None
        hook_manager = AttentionHookManager(self.model)

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

            loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_tokens += y.numel()
            total_loss += loss.item() * y.numel()

        mean_loss = total_loss / total_tokens
        metrics = {
            "val/loss": mean_loss,
            "val/perplexity": perplexity(mean_loss),
            "val/tokens_seen": self.tokens_seen,
        }
        self.logger.log_metrics(metrics, step=self.global_step, tokens_seen=self.tokens_seen)
        self.logger.update_summary("best_val_perplexity", perplexity(mean_loss))

        if attn_weights_np is not None:
            self.logger.log_attention_stats(attn_weights_np, step=self.global_step, tokens_seen=self.tokens_seen)

        return mean_loss

    def _causal_mask(self, seq_len: int, batch_size: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
