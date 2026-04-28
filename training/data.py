"""
Data pipeline for FineWebEdu (HuggingFaceFW/fineweb-edu, sample-10BT).

Flow:
  1. Stream documents from HF Hub
  2. Tokenise with tiktoken (gpt2 encoding, ~50k vocab)
  3. Pack tokens into fixed-length chunks
  4. Cache shards to disk as .pt files so subsequent runs skip re-tokenisation
  5. Return DataLoaders backed by ShardedTokenDataset, which loads
     `shards_per_chunk` shards at a time — arbitrarily large corpora
     can be trained on without holding everything in RAM.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator

import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

EOT_TOKEN = 50256  # tiktoken gpt2 end-of-text token, used as document separator


class ShardedTokenDataset(IterableDataset):
    """
    Lazily loads pre-tokenised shards from disk in chunks of `shards_per_chunk`.

    Memory at any time: `shards_per_chunk` shards (~1M tokens each by default).

    For DDP, shard files are interleaved across ranks so each rank owns a
    disjoint subset. DataLoader workers further subdivide each rank's shards,
    so there is no duplicated data regardless of num_workers or world_size.
    """

    def __init__(
        self,
        shard_files: list[Path],
        seq_len: int,
        shards_per_chunk: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
    ) -> None:
        self.local_shards = shard_files[rank::world_size]
        self.seq_len = seq_len
        self.shards_per_chunk = shards_per_chunk
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        files = list(self.local_shards)

        # DataLoader workers subdivide the shard list so no two workers
        # yield the same sequences.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        if self.shuffle:
            random.shuffle(files)

        for chunk_start in range(0, len(files), self.shards_per_chunk):
            chunk = files[chunk_start : chunk_start + self.shards_per_chunk]
            tokens = torch.cat(
                [torch.load(f, weights_only=True) for f in chunk]
            ).long()

            n_seqs = (len(tokens) - 1) // self.seq_len
            indices = list(range(n_seqs))
            if self.shuffle:
                random.shuffle(indices)

            for idx in indices:
                start = idx * self.seq_len
                yield (
                    tokens[start : start + self.seq_len],
                    tokens[start + 1 : start + self.seq_len + 1],
                )

            del tokens  # release chunk memory before loading the next


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


def get_dataloaders(
    cache_dir: str,
    seq_len: int,
    batch_size: int,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    shards_per_chunk: int = 100,
    n_shards: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders backed by ShardedTokenDataset.

    All .pt shard files found in cache_dir are used. The first
    `train_ratio` fraction (by shard count) go to train; the rest to val.
    Only `shards_per_chunk` shards are held in RAM at any one time per worker.

    Args:
        cache_dir:        directory containing shard_*.pt files
        seq_len:          context window length (should match model max_seq_len)
        batch_size:       per-GPU batch size
        train_ratio:      fraction of shards used for training
        num_workers:      DataLoader workers for train (val always uses 0)
        rank:             DDP rank (0 for single-GPU)
        world_size:       total DDP processes (1 for single-GPU)
        shards_per_chunk: shards loaded into RAM at once (controls peak memory)
        n_shards: total number of shards to use
    """
    all_shards = sorted(Path(cache_dir).glob("shard_*.pt"))
    if not all_shards:
        raise FileNotFoundError(
            f"No shards found in {cache_dir}. Run build_fineweb_cache() first."
        )

    if n_shards is not None:
        all_shards = all_shards[:n_shards]

    n_train = int(len(all_shards) * train_ratio)
    train_shards = all_shards[:n_train]
    val_shards = all_shards[n_train:]

    logger.info(
        f"Found {len(all_shards)} shards — "
        f"{len(train_shards)} train / {len(val_shards)} val "
        f"(chunk size: {shards_per_chunk})"
    )

    train_ds = ShardedTokenDataset(
        train_shards,
        seq_len=seq_len,
        shards_per_chunk=shards_per_chunk,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )
    val_ds = ShardedTokenDataset(
        val_shards,
        seq_len=seq_len,
        shards_per_chunk=shards_per_chunk,
        rank=0,        # all ranks evaluate the same val shards
        world_size=1,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=0,   # avoids worker-state issues on repeated early-terminated iterations
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
