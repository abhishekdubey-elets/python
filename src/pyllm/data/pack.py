"""Tokenize documents and pack them into flat token-id shards on disk.

We concatenate all documents into one long token stream (each ended by EOS as a
document separator) and dump it as a raw ``uint16`` binary. Why:
  * uint16 holds ids 0..65535 — enough for our 50257 vocab, at half the size of
    int32. (Assert vocab < 65536.)
  * a flat stream + memmap (see loader.py) means training reads contiguous blocks
    with near-zero overhead and no per-example padding.

Document-level train/val split (not token-level) avoids leaking the tail of a
training doc into validation.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

DTYPE = np.uint16


class SupportsEncode(Protocol):
    eos_id: int | None

    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...


def tokenize_documents(docs: Sequence[str], tokenizer: SupportsEncode) -> np.ndarray:
    """Encode each doc (with a trailing EOS) and concatenate into one uint16 array."""
    if tokenizer.eos_id is None:
        raise ValueError("tokenizer must define an eos_id to separate documents.")
    chunks: list[np.ndarray] = []
    for doc in docs:
        ids = tokenizer.encode(doc, add_eos=True)
        chunks.append(np.asarray(ids, dtype=DTYPE))
    if not chunks:
        return np.zeros(0, dtype=DTYPE)
    return np.concatenate(chunks)


def write_bin(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(DTYPE).tofile(path)


def pack_dataset(
    docs: Sequence[str],
    tokenizer: SupportsEncode,
    out_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict:
    """Shuffle, split at the document level, tokenize, and write train/val bins.

    Returns a stats dict (also written to ``meta.json``) with token counts.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    n_val = int(len(order) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_arr = tokenize_documents([docs[i] for i in train_idx], tokenizer)
    val_arr = tokenize_documents([docs[i] for i in val_idx], tokenizer)
    write_bin(train_arr, out_dir / "train.bin")
    write_bin(val_arr, out_dir / "val.bin")

    stats = {
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else None,
        "num_docs": len(docs),
        "train_docs": len(train_idx),
        "val_docs": len(val_idx),
        "train_tokens": int(train_arr.size),
        "val_tokens": int(val_arr.size),
    }
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
