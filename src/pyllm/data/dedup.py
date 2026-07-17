"""Deduplication: exact (content hash) and near-duplicate (MinHash + LSH).

Why dedup matters: duplicated training data wastes capacity, biases the model
toward repeated content, and — worst — leaks eval examples into training
(contamination). Code corpora are especially duplicated (vendored deps, forks,
copy-pasted boilerplate).

* Exact dedup: hash the (normalized) bytes, keep first occurrence. O(n).
* Near-dup: MinHash estimates Jaccard similarity of k-shingle sets cheaply;
  LSH banding buckets likely-similar docs so we only compare within buckets.
"""

from __future__ import annotations

import hashlib
import random
import zlib
from collections.abc import Iterable, Sequence

_MERSENNE_PRIME = (1 << 61) - 1  # a large prime for the hash family
_MAX32 = (1 << 32) - 1


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_exact(texts: Iterable[str]) -> list[int]:
    """Return indices of the first occurrence of each distinct text."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, t in enumerate(texts):
        h = content_hash(t)
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def _shingles(text: str, k: int) -> set[int]:
    """Set of hashed k-character shingles (a cheap surrogate for the token set)."""
    if len(text) < k:
        return {zlib.crc32(text.encode("utf-8"))}
    return {zlib.crc32(text[i : i + k].encode("utf-8")) for i in range(len(text) - k + 1)}


class MinHasher:
    """MinHash signatures using a family of hashes  h_i(x) = (a_i*x + b_i) mod p.

    Two documents' fraction of matching signature entries is an unbiased estimate
    of the Jaccard similarity of their shingle sets.
    """

    def __init__(self, num_perm: int = 64, k: int = 5, seed: int = 1) -> None:
        self.num_perm = num_perm
        self.k = k
        rng = random.Random(seed)
        self.a = [rng.randrange(1, _MERSENNE_PRIME) for _ in range(num_perm)]
        self.b = [rng.randrange(0, _MERSENNE_PRIME) for _ in range(num_perm)]

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = _shingles(text, self.k)
        sig = []
        for a, b in zip(self.a, self.b):
            m = min(((a * s + b) % _MERSENNE_PRIME) & _MAX32 for s in shingles)
            sig.append(m)
        return tuple(sig)


def estimated_jaccard(sig1: Sequence[int], sig2: Sequence[int]) -> float:
    matches = sum(x == y for x, y in zip(sig1, sig2))
    return matches / len(sig1)


def dedup_near(
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    bands: int = 16,
    k: int = 5,
    seed: int = 1,
) -> list[int]:
    """Return indices to keep after removing near-duplicates (greedy, first-wins).

    LSH: split each signature into ``bands`` bands; documents sharing a full band
    land in the same bucket and become candidate pairs. We then confirm with the
    estimated Jaccard against threshold before dropping.
    """
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands.")
    rows = num_perm // bands
    hasher = MinHasher(num_perm=num_perm, k=k, seed=seed)
    sigs = [hasher.signature(t) for t in texts]

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    kept: list[int] = []
    dropped: set[int] = set()

    for i, sig in enumerate(sigs):
        # candidate neighbors = anything sharing a band bucket with doc i
        candidate_ids: set[int] = set()
        band_keys = []
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows])
            band_keys.append(key)
            candidate_ids.update(buckets.get(key, ()))

        is_dup = any(
            j not in dropped and estimated_jaccard(sig, sigs[j]) >= threshold
            for j in candidate_ids
        )
        if is_dup:
            dropped.add(i)
        else:
            kept.append(i)
            for key in band_keys:
                buckets.setdefault(key, []).append(i)
    return kept
