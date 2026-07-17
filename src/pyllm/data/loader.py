"""Reading packed token shards for training.

Two access patterns:
  * ``PackedDataset``  — a map-style torch Dataset yielding contiguous (x, y)
    blocks; good with a DataLoader and for deterministic validation passes.
  * ``get_batch``      — nanoGPT-style random-offset sampling straight from a
    memmap; the simplest thing that works for LM pretraining, no DataLoader.

For a causal LM, the target is the input shifted by one: predict token t+1 from
tokens <= t.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DTYPE = np.uint16


def load_bin(path: str | Path) -> np.memmap:
    """Memory-map a packed .bin as a read-only uint16 array (no full load)."""
    return np.memmap(path, dtype=DTYPE, mode="r")


class PackedDataset(Dataset):
    """Contiguous, non-overlapping (x, y) blocks over a packed token stream."""

    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        self.data = load_bin(bin_path)
        self.seq_len = seq_len
        if len(self.data) < seq_len + 1:
            raise ValueError(
                f"Packed data has {len(self.data)} tokens, need at least seq_len+1={seq_len + 1}."
            )

    def __len__(self) -> int:
        # number of full non-overlapping blocks (each needs seq_len+1 tokens)
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.seq_len
        # .astype(int64) also copies out of the read-only memmap (required by torch)
        x = torch.from_numpy(self.data[i : i + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64))
        return x, y


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random (x, y) blocks from a token stream.

    Random offsets mean the model sees varied context boundaries each step, which
    is exactly what we want for pretraining.
    """
    max_start = len(data) - seq_len - 1
    if max_start < 1:
        raise ValueError("data too short for the requested seq_len.")
    ix = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
