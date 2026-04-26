"""
Single-GPU / DataParallel training entrypoint.

Usage:
    python scripts/train.py --config configs/vanilla_small.yaml
    python scripts/train.py --config configs/vanilla_small.yaml --resume experiments/rope/run_001/best.pt
"""

import argparse
import dataclasses
import logging

import torch
import torch.nn as nn

from training.checkpointing import CheckpointManager
from training.configs.schemas import ExperimentConfig
from training.data import build_fineweb_cache, get_dataloaders
from training.factory import build_model
from training.optimizer import get_optimizer, get_warmup_cosine_scheduler
from training.trainer import Trainer
from training.wandb_logger import WandBLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cache-dir", default="data/fineweb_cache", help="Tokenised shard cache directory")
    parser.add_argument("--n-shards", type=int, default=100, help="Number of FineWebEdu shards to cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    build_fineweb_cache(args.cache_dir, seq_len=cfg.model.max_seq_len, n_shards=args.n_shards)
    train_loader, val_loader = get_dataloaders(
        cache_dir=args.cache_dir,
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        train_ratio=cfg.data.train_ratio,
        num_workers=cfg.data.num_workers,
        n_shards=args.n_shards,
    )
    tokens_per_step = cfg.training.batch_size * cfg.model.max_seq_len
    total_steps = cfg.training.max_tokens // tokens_per_step

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Token budget: {cfg.training.max_tokens:,} tokens (~{total_steps:,} steps)")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = get_optimizer(model, cfg.training)
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg.training.warmup_steps, total_steps)

    # ── Logging + checkpointing ───────────────────────────────────────────────
    config_dict = dataclasses.asdict(cfg)
    logger_obj = WandBLogger(cfg.wandb, config_dict=config_dict)
    logger_obj.watch_model(model)

    run_dir = f"experiments/{cfg.model.attention_type}/{cfg.wandb.run_name}"
    checkpointer = CheckpointManager(run_dir)

    if args.resume:
        checkpointer.load(args.resume, model, optimizer, scheduler, device)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        cfg.training, logger_obj, checkpointer, device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
