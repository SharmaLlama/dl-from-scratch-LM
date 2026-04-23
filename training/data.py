"""
Data pipeline for FineWebEdu (HuggingFaceFW/fineweb-edu, sample-10BT).

Flow:
  1. Stream documents from HF Hub
  2. Tokenise with tiktoken (gpt2 encoding, ~50k vocab)
  3. Pack tokens into fixed-length chunks — no padding, no waste
  4. Cache shards to disk as .pt files so subsequent runs skip re-tokenisation
  5. Return standard DataLoaders (with DistributedSampler for DDP)

Vocab note: tiktoken gpt2 has 50,257 tokens. Make sure model.vocab_size matches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

EOT_TOKEN = 50256  # tiktoken gpt2 end-of-text token, used as document separator


class PackedTokenDataset(Dataset):
    """
    Flat tensor of pre-tokenised ids split into non-overlapping chunks.
    Each item is (input_ids, target_ids) where target = input shifted by 1.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def _tokenise_shard(
    shard_docs: list[str],
    enc: tiktoken.Encoding,
) -> torch.Tensor:
    """Tokenise a list of documents and concatenate with EOT separators."""
    all_tokens: list[int] = []
    for doc in shard_docs:
        tokens = enc.encode_ordinary(doc)
        tokens.append(EOT_TOKEN)
        all_tokens.extend(tokens)
    return torch.tensor(all_tokens, dtype=torch.int32)


def build_fineweb_cache(
    cache_dir: str,
    seq_len: int,
    n_shards: int = 100,
    docs_per_shard: int = 10_000,
    split: str = "train",
) -> Path:
    """
    Stream FineWebEdu from HF, tokenise, and save shards to cache_dir.
    Skips shards that already exist on disk.

    Returns the cache directory path.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=split,
        streaming=True,
    )

    shard_idx = 0
    current_docs: list[str] = []

    for example in dataset:
        shard_file = cache_path / f"shard_{shard_idx:04d}.pt"
        if shard_file.exists():
            current_docs = []
            shard_idx += 1
            if shard_idx >= n_shards:
                break
            continue

        current_docs.append(example["text"])

        if len(current_docs) >= docs_per_shard:
            tokens = _tokenise_shard(current_docs, enc)
            torch.save(tokens, shard_file)
            logger.info(f"Saved shard {shard_idx} ({len(tokens):,} tokens) → {shard_file}")
            current_docs = []
            shard_idx += 1
            if shard_idx >= n_shards:
                break

    # Save any remaining docs
    if current_docs and shard_idx < n_shards:
        shard_file = cache_path / f"shard_{shard_idx:04d}.pt"
        if not shard_file.exists():
            tokens = _tokenise_shard(current_docs, enc)
            torch.save(tokens, shard_file)

    return cache_path


def load_cached_shards(cache_dir: str) -> torch.Tensor:
    """Load all cached shards and concatenate into a single flat token tensor."""
    shard_files = sorted(Path(cache_dir).glob("shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(
            f"No shards found in {cache_dir}. Run build_fineweb_cache() first."
        )
    logger.info(f"Loading {len(shard_files)} shards from {cache_dir}")
    return torch.cat([torch.load(f, weights_only=True) for f in shard_files])


def get_dataloaders(
    cache_dir: str,
    seq_len: int,
    batch_size: int,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """
    Load tokenised FineWebEdu shards and return train/val DataLoaders.

    Args:
        cache_dir: directory containing shard_*.pt files from build_fineweb_cache()
        seq_len: context window length (should match model max_seq_len)
        batch_size: per-GPU batch size
        train_ratio: fraction of tokens used for training
        num_workers: DataLoader workers
        rank: DDP rank (0 for single-GPU)
        world_size: total DDP processes (1 for single-GPU)
    """
    token_ids = load_cached_shards(cache_dir)
    token_ids = token_ids.long()  # CrossEntropyLoss expects long

    split = int(len(token_ids) * train_ratio)
    train_ds = PackedTokenDataset(token_ids[:split], seq_len)
    val_ds = PackedTokenDataset(token_ids[split:], seq_len)

    train_sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # num_workers=0 for val: eval is compute-bound not IO-bound, and multiprocessing
    # workers accumulate stale state when the loop is broken early on each eval call.
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
