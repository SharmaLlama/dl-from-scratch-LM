"""
Head ablation study entrypoint.

Usage:
    python scripts/run_ablation.py \\
        --checkpoint experiments/vanilla/run_001/best.pt \\
        --config configs/vanilla_small.yaml \\
        --cache-dir data/fineweb_cache \\
        --output experiments/vanilla/run_001/ablation.json \\
        --greedy
"""

import argparse
import logging

import torch

from training.checkpointing import CheckpointManager
from training.configs.schemas import ExperimentConfig
from training.data import get_dataloaders
from training.factory import build_model
from utils.ablation import HeadAblationStudy, plot_ablation_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default="data/fineweb_cache")
    parser.add_argument("--output", default=None, help="Path to save ablation results JSON")
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    checkpointer = CheckpointManager(run_dir="", rank=0)
    checkpointer.load(args.checkpoint, model, device=device)
    model.eval()

    _, val_loader = get_dataloaders(
        cache_dir=args.cache_dir,
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        train_ratio=cfg.data.train_ratio,
        num_workers=cfg.data.num_workers,
        n_shards=args.n_shards,

    )

    output = args.output or f"experiments/{cfg.model.attention_type}/ablation.json"
    study = HeadAblationStudy(model, val_loader, device, output_path=output)

    logger.info("Running per-head importance scores...")
    scores_df = study.head_importance_scores()
    print("\nHead importance (least → most important):")
    print(scores_df.to_string(index=False))

    if args.greedy:
        logger.info("Running greedy head removal...")
        removal_order, perplexities = study.greedy_removal()
        fig = plot_ablation_curve(removal_order, perplexities)
        curve_path = output.replace(".json", "_curve.png")
        fig.savefig(curve_path, dpi=150)
        logger.info(f"Saved ablation curve → {curve_path}")


if __name__ == "__main__":
    main()
