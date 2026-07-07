"""A minimal, from-scratch Byte-Pair Encoding implementation — for LEARNING.

This is NOT what we train the real model on (that is the fast Rust-backed
``tokenizers`` library, wrapped in ``tokenizer.py``). This file exists so you can
read the *entire* BPE algorithm in one place, with no magic:

    train:  bytes -> repeatedly merge the most frequent adjacent pair
    encode: bytes -> re-apply the learned merges, lowest-priority-id first
    decode: ids   -> concatenate the byte strings each id stands for

It operates on **raw UTF-8 bytes**, so like byte-level BPE it can encode any
string with zero out-of-vocabulary tokens. Complexity of the naive trainer is
O(num_merges * len(text)); production trainers use incremental pair counts to go
much faster, but the *result* is identical. (Inspired by Karpathy's minbpe.)
"""

from __future__ import annotations


def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair occurs. e.g. [1,2,1,2] -> {(1,2):2,(2,1):1}."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with the single ``new_id``."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ReferenceBPE:
    """A tiny, readable BPE tokenizer over UTF-8 bytes."""

    def __init__(self) -> None:
        # merges: an ordered map (pair) -> new_id. Order == merge priority.
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: id -> the byte string it expands to. Ids 0..255 are the raw bytes.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn merges until the vocabulary reaches ``vocab_size`` (>= 256)."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte alphabet).")
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = _get_stats(ids)
            if not stats:
                break  # nothing left to merge (text too short for this vocab_size)
            # pick the most frequent pair; ties broken by first-seen (max is stable-ish)
            pair = max(stats, key=lambda p: stats[p])
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]!r}) count={stats[pair]}")

    def encode(self, text: str) -> list[int]:
        """Encode text by greedily applying merges in the order they were learned."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # find the pair whose merge was learned EARLIEST (lowest new_id) and apply it
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no learned merge applies anymore
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Invert encoding: expand each id to its bytes, concat, decode as UTF-8."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
