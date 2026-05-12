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

# TF32 for fp32 matmuls (Ampere+). 'high' = TF32; safe under mixed-precision training
# since the compute-heavy paths run in bf16/fp16 anyway.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Cap dynamo's recompilation cache. Without this, every new input shape (e.g. a
# partial eval batch) adds another compiled graph in VRAM, and over a 1B-token run
# this can grow unboundedly. 16 entries is plenty for one train + one eval shape.
torch._dynamo.config.cache_size_limit = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--cache-dir", default="data/fineweb_cache")
    parser.add_argument("--n-shards", type=int, default=100, help="Total shards to build/cache")
    parser.add_argument("--shards-per-chunk", type=int, default=100, help="Shards loaded into RAM at once")
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
        shards_per_chunk=args.shards_per_chunk,
        n_shards=args.n_shards,
    )
    tokens_per_step = cfg.training.batch_size * cfg.model.max_seq_len * cfg.training.grad_accum_steps
    total_steps = cfg.training.max_tokens // tokens_per_step

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)

    if is_main_process(rank):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

    # Compile BEFORE DDP wrap (matches nanoGPT). DDP's reducer hooks then attach to
    # the compiled module's parameters, and `model.no_sync()` stays directly on DDP.
    if cfg.training.compile:
        if is_main_process(rank):
            logger.info("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    model = wrap_model_ddp(model, rank)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = get_optimizer(model, cfg.training)
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg.training.warmup_steps, total_steps)

    # ── Logging + checkpointing (main process only) ───────────────────────────
    config_dict = dataclasses.asdict(cfg)
    wandb_logger = WandBLogger(cfg.wandb, config_dict=config_dict, enabled=is_main_process(rank))
    if is_main_process(rank):
        wandb_logger.watch_model(model)

    checkpointer = CheckpointManager("experiments", cfg, rank=rank)
    if is_main_process(rank):
        logger.info(f"Checkpointing to {checkpointer.dir}")

    resume_state = None
    if args.resume:
        resume_state = checkpointer.load(args.resume, model, optimizer, scheduler, device)
        if is_main_process(rank):
            resumed_tokens = resume_state.get("metrics", {}).get("tokens_seen", 0)
            resumed_step = resume_state.get("epoch", 0)
            logger.info(f"Resumed from {args.resume}: step={resumed_step}, tokens_seen={resumed_tokens:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        cfg.training, wandb_logger, checkpointer, device,
        rank=rank, world_size=world_size,
    )
    if resume_state is not None:
        trainer.restore_progress(resume_state)
    trainer.fit()

    cleanup_ddp()


if __name__ == "__main__":
    main()
