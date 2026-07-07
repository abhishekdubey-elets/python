"""Tests for the tokenizer (Phase 3).

Two layers:
* ``ReferenceBPE`` — the from-scratch teaching implementation.
* ``PyTokenizer`` — the production byte-level BPE trained via the tokenizers lib.

The single most important property for a code tokenizer is **exact round-trip**:
decode(encode(x)) == x for arbitrary text, including indentation, newlines, and
unicode. If that ever breaks, the model would be trained on corrupted targets.
"""

from __future__ import annotations

import pytest

from pyllm.tokenizer import PyTokenizer, ReferenceBPE, train_bpe_tokenizer
from pyllm.tokenizer.tokenizer import EOS_TOKEN

# A small but representative Python corpus (indentation, f-strings, operators,
# a unicode comment, dunder methods) — enough to learn meaningful merges fast.
SAMPLE_CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
    "for i in range(10):\n    if i % 2 == 0:\n        print(f'{i} is even')\n",
    "import os\nimport sys\nfrom pathlib import Path  # imports\n",
    "# unicode: café, naïve, 数学\nresult = [x**2 for x in range(100)]\n",
] * 20  # repeat so pairs clear the min_frequency threshold


# --------------------------------------------------------------------------- #
# ReferenceBPE — the from-scratch algorithm
# --------------------------------------------------------------------------- #
def test_reference_bpe_roundtrip_ascii():
    bpe = ReferenceBPE()
    bpe.train("def foo(): return foo() + foo()\n" * 5, vocab_size=300)
    text = "def foo(): return foo()\n"
    assert bpe.decode(bpe.encode(text)) == text


def test_reference_bpe_roundtrip_unicode():
    bpe = ReferenceBPE()
    bpe.train("café 数学 café 数学 " * 10, vocab_size=280)
    text = "café 数学 test"
    assert bpe.decode(bpe.encode(text)) == text


def test_reference_bpe_learns_merges_and_grows_vocab():
    bpe = ReferenceBPE()
    bpe.train("ababababab", vocab_size=260)
    assert len(bpe.merges) > 0
    assert len(bpe.vocab) == 260
    # "ab" is the dominant pair, so it must have become a single merged token.
    assert (ord("a"), ord("b")) in bpe.merges


def test_reference_bpe_rejects_tiny_vocab():
    with pytest.raises(ValueError, match="vocab_size"):
        ReferenceBPE().train("abc", vocab_size=100)


# --------------------------------------------------------------------------- #
# PyTokenizer — the production byte-level BPE
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tokenizer() -> PyTokenizer:
    # Small vocab so training is instant; round-trip is exact regardless of size.
    return train_bpe_tokenizer(SAMPLE_CORPUS, vocab_size=500, min_frequency=1)


def test_special_tokens_present_with_low_ids(tokenizer):
    assert tokenizer.eos_id is not None
    assert tokenizer.pad_id is not None
    # Special tokens are added first, so they occupy the lowest ids.
    assert tokenizer.eos_id == tokenizer.token_to_id(EOS_TOKEN)
    assert tokenizer.eos_id < 10


def test_vocab_size_is_capped_by_request(tokenizer):
    # vocab_size is a TARGET CEILING: on a tiny corpus the trainer runs out of
    # frequent pairs before reaching it, so actual <= requested. It must still
    # exceed the base (256 bytes + 5 special tokens) by the merges it did learn.
    assert 261 < tokenizer.vocab_size <= 500


@pytest.mark.parametrize(
    "text",
    [
        "def add(a, b):\n    return a + b\n",
        "        deeply.nested.indent = True\n",   # 8-space indent
        "x = {'k': [1, 2, 3], 'u': 'café'}\n",     # unicode + punctuation
        "\t\ttab\tindented\n",                       # tabs
        "",                                          # empty string
        "🐍 python snake emoji in a string\n",       # 4-byte utf-8
    ],
)
def test_bytelevel_roundtrip_is_exact(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_add_eos_appends_eos_id(tokenizer):
    ids = tokenizer.encode("pass\n", add_eos=True)
    assert ids[-1] == tokenizer.eos_id
    # ...and without it, the eos is absent.
    assert tokenizer.encode("pass\n")[-1] != tokenizer.eos_id


def test_save_load_roundtrip(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    reloaded = PyTokenizer.load(path)
    assert reloaded.vocab_size == tokenizer.vocab_size
    text = "def f():\n    return 42\n"
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded.decode(reloaded.encode(text)) == text


def test_python_is_compressed(tokenizer):
    # Sanity: after learning merges, common Python is fewer tokens than bytes.
    code = "def __init__(self):\n        return self\n" * 3
    assert len(tokenizer.encode(code)) < len(code.encode("utf-8"))
