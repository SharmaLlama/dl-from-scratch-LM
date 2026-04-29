import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from training.configs.schemas import TrainingConfig


def get_optimizer(model: nn.Module, cfg: TrainingConfig) -> AdamW:
    # Exclude biases and LayerNorm params from weight decay
    decay, no_decay = [], []
    seen: set[int] = set()  # dedupe tied params (e.g. embedding ↔ output projection)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    # fused=True dispatches AdamW to a single fused CUDA kernel — much less
    # launch overhead per step than the foreach (default) path.
    fused = torch.cuda.is_available()
    return AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
    )


def get_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)
