import dataclasses
import json
from pathlib import Path
from typing import Optional

import yaml
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from training.configs.schemas import ExperimentConfig
from training.ddp_setup import unwrap_model


class CheckpointManager:
    """
    Saves and restores model + optimizer + scheduler state.

    Run directory resolution:
      - Scans base_dir for subdirectories named {attention_type}_N that contain a
        config.yaml whose 'model' section matches the current ModelConfig.
      - If a match is found, checkpoints are written there (overriding old ones).
      - If no match, a new directory {attention_type}_{N+1} is created and the
        full config is saved as config.yaml inside it.
    """

    def __init__(self, base_dir: str, cfg: ExperimentConfig, rank: int = 0) -> None:
        self.base = Path(base_dir)
        self.rank = rank
        self.best_val_loss = float("inf")

        # All ranks resolve the same path via the same deterministic read-only scan.
        self.dir = self._resolve_run_dir(cfg)

        if rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)
            config_file = self.dir / "config.yaml"
            if not config_file.exists():
                with open(config_file, "w") as f:
                    yaml.dump(
                        dataclasses.asdict(cfg), f,
                        default_flow_style=False, sort_keys=False,
                    )

    def _resolve_run_dir(self, cfg: ExperimentConfig) -> Path:
        cfg_dict = dataclasses.asdict(cfg)
        prefix = cfg.model.attention_type

        if self.base.exists():
            for run_dir in sorted(self.base.glob(f"{prefix}_*")):
                config_file = run_dir / "config.yaml"
                if not config_file.exists():
                    continue
                with open(config_file) as f:
                    saved = yaml.safe_load(f)
                if saved == cfg_dict:
                    return run_dir

        existing = list(self.base.glob(f"{prefix}_*")) if self.base.exists() else []
        return self.base / f"{prefix}_{len(existing) + 1}"

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        epoch: int,
        val_loss: float,
        metrics: Optional[dict] = None,
    ) -> None:
        if self.rank != 0:
            return

        state = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics or {},
        }

        path = self.dir / f"step_{epoch:04d}.pt"
        torch.save(state, path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(state, self.dir / "best.pt")

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        device: Optional[torch.device] = None,
    ) -> dict:
        state = torch.load(path, map_location=device, weights_only=True)
        unwrap_model(model).load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        return state

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[dict]:
        checkpoints = sorted(self.dir.glob("step_*.pt"))
        if not checkpoints:
            return None
        return self.load(str(checkpoints[-1]), model, optimizer, scheduler, device)

    def load_best(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Optional[dict]:
        best = self.dir / "best.pt"
        if not best.exists():
            return None
        return self.load(str(best), model, device=device)
