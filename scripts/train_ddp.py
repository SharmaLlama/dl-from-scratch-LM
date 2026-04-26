"""
Distributed (DDP) training entrypoint.

Usage:
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/vanilla_small.yaml
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/vanilla_small.yaml --resume experiments/.../best.pt
"""

import argparse
import dataclasses
import logging
import os

import torch

from training.checkpointing import CheckpointManager
from training.configs.schemas import ExperimentConfig
from training.data import build_fineweb_cache, get_dataloaders
from training.ddp_setup import cleanup_ddp, get_device, is_main_process, setup_ddp, wrap_model_ddp
from training.factory import build_model
from training.optimizer import get_optimizer, get_warmup_cosine_scheduler
from training.trainer import Trainer
from training.wandb_logger import WandBLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [rank%(process)d] %(message)s")
logger = logging.getLogger(__name__)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--cache-dir", default="data/fineweb_cache")
    parser.add_argument("--n-shards", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_ddp(rank, world_size)
    device = get_device(rank)

    cfg = ExperimentConfig.from_yaml(args.config)
    torch.manual_seed(cfg.training.seed + rank)

    # ── Data — only rank 0 downloads/caches shards ────────────────────────────
    if is_main_process(rank):
        build_fineweb_cache(args.cache_dir, seq_len=cfg.model.max_seq_len, n_shards=args.n_shards)
    torch.distributed.barrier()  # wait for rank 0 to finish caching

    train_loader, val_loader = get_dataloaders(
        cache_dir=args.cache_dir,
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        train_ratio=cfg.data.train_ratio,
        num_workers=cfg.data.num_workers,
        rank=rank,
        world_size=world_size,
    )
    tokens_per_step = cfg.training.batch_size * cfg.model.max_seq_len
    total_steps = cfg.training.max_tokens // tokens_per_step

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    model = wrap_model_ddp(model, rank)

    if is_main_process(rank):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = get_optimizer(model, cfg.training)
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg.training.warmup_steps, total_steps)

    # ── Logging + checkpointing (main process only) ───────────────────────────
    config_dict = dataclasses.asdict(cfg)
    wandb_logger = WandBLogger(cfg.wandb, config_dict=config_dict, enabled=is_main_process(rank))
    if is_main_process(rank):
        wandb_logger.watch_model(model)

    run_dir = f"experiments/{cfg.model.attention_type}/{cfg.wandb.run_name}"
    checkpointer = CheckpointManager(run_dir, rank=rank)

    if args.resume:
        checkpointer.load(args.resume, model, optimizer, scheduler, device)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        cfg.training, wandb_logger, checkpointer, device,
        rank=rank, world_size=world_size,
    )
    trainer.fit()

    cleanup_ddp()


if __name__ == "__main__":
    main()
