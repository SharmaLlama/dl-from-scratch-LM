from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    attention_type: str           # vanilla | rope | alibi | big_bird | performer | flash | nope
    pe_type: str                  # sinusoidal | learned | rope | alibi | none
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    dropout: float
    pre_norm: bool = True
    ffn_type: str = "relu"        # relu | swiglu
    dk: Optional[int] = None      # defaults to d_model // n_heads
    dv: Optional[int] = None      # defaults to d_model // n_heads
    attention_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dk is None:
            self.dk = self.d_model // self.n_heads
        if self.dv is None:
            self.dv = self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    max_tokens: int               # total token budget — training stops when this is consumed
    batch_size: int
    lr: float
    warmup_steps: int
    eval_interval: int            # run validation every N tokens
    checkpoint_interval: int      # save checkpoint every N tokens
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    seed: int = 42
    max_eval_tokens: int = 5_000_000  # cap val loop — no need to eval the full val split
    grad_accum_steps: int = 1         # accumulate gradients over N micro-batches before stepping
    mixed_precision: bool = False     # enable AMP (torch.autocast + GradScaler)


@dataclass
class DataConfig:
    dataset_path: str
    tokenizer_path: str
    train_ratio: float = 0.9
    num_workers: int = 4


@dataclass
class WandBConfig:
    project: str
    run_name: str
    entity: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    log_every_steps: int = 50       # how often to send metrics to wandb
    log_attention_every: int = 200  # how often to capture and log attention stats
    watch_model: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    wandb: WandBConfig

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            data=DataConfig(**raw["data"]),
            wandb=WandBConfig(**raw["wandb"]),
        )
