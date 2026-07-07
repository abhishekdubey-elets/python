"""Tests for the data pipeline (Phase 5)."""

from __future__ import annotations

import numpy as np

from pyllm.data import (
    MinHasher,
    PackedDataset,
    clean_document,
    dedup_exact,
    dedup_near,
    estimated_jaccard,
    is_valid_python,
    normalize_text,
    pack_dataset,
)


# --- clean ------------------------------------------------------------------ #
def test_is_valid_python():
    assert is_valid_python("def f():\n    return 1\n")
    assert not is_valid_python("def f(:\n  ??\n")


def test_normalize_line_endings_and_trailing_ws():
    assert normalize_text("a  \r\nb\t\r\n") == "a\nb\n"


def test_clean_document_rejects_invalid_and_short():
    assert clean_document("x = 1\ny = 2\nz = x + y\n") is not None
    assert clean_document("not valid python !!!", require_valid=True) is None
    assert clean_document("x=1\n") is None  # too few lines


# --- dedup ------------------------------------------------------------------ #
def test_dedup_exact():
    texts = ["a\nb\nc", "a\nb\nc", "d\ne\nf"]
    assert dedup_exact(texts) == [0, 2]


def test_minhash_estimates_similarity():
    h = MinHasher(num_perm=128, k=3, seed=0)
    base = "def add(a, b):\n    return a + b\n" * 3
    near = base + "# a trailing comment\n"
    far = "class Foo:\n    pass\n" * 3
    assert estimated_jaccard(h.signature(base), h.signature(near)) > 0.7
    assert estimated_jaccard(h.signature(base), h.signature(far)) < 0.3


def test_dedup_near_removes_similar():
    base = "def add(a, b):\n    return a + b\n" * 4
    near = base + "x = 1\n"
    far = "import os\nimport sys\nprint(os, sys)\n" * 2
    kept = dedup_near([base, near, far], threshold=0.7, num_perm=64, bands=16, k=4)
    assert 0 in kept and 2 in kept  # base and far kept
    assert 1 not in kept           # near-dup of base removed


# --- pack + load ------------------------------------------------------------ #
class _DummyTokenizer:
    """Minimal tokenizer: bytes as ids, EOS = 256. vocab < 65536 for uint16."""

    eos_id = 256
    vocab_size = 257

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return ids + [self.eos_id] if add_eos else ids


def test_pack_and_load_roundtrip(tmp_path):
    docs = [f"line {i}\nvalue = {i}\nprint(value)\n" for i in range(20)]
    tok = _DummyTokenizer()
    stats = pack_dataset(docs, tok, tmp_path, val_fraction=0.2, seed=0)
    assert stats["train_docs"] == 16 and stats["val_docs"] == 4
    assert stats["train_tokens"] > 0 and (tmp_path / "train.bin").exists()

    ds = PackedDataset(tmp_path / "train.bin", seq_len=8)
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)
    # target is input shifted by one
    raw = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    assert x.tolist() == raw[:8].tolist()
    assert y.tolist() == raw[1:9].tolist()
