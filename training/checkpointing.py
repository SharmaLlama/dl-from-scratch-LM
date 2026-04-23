import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from training.ddp_setup import unwrap_model


class CheckpointManager:
    """Saves and restores model + optimizer + scheduler state."""

    def __init__(self, run_dir: str, rank: int = 0) -> None:
        self.dir = Path(run_dir)
        self.rank = rank
        if rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        epoch: int,
        val_loss: float,
        metrics: Optional[dict] = None,
    ) -> None:
        # Only the main process writes checkpoints
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

        path = self.dir / f"epoch_{epoch:04d}.pt"
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
        checkpoints = sorted(self.dir.glob("epoch_*.pt"))
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
