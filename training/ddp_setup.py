import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def get_device(rank: int) -> torch.device:
    return torch.device(f"cuda:{rank}")


def is_main_process(rank: int) -> bool:
    return rank == 0


def wrap_model_ddp(model: nn.Module, rank: int) -> DDP:
    return DDP(
        model.to(get_device(rank)),
        device_ids=[rank],
        find_unused_parameters=False,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP or DataParallel wrapper for checkpointing."""
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model
