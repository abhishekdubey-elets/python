"""Train a byte-level BPE tokenizer optimized for Python.

Byte-level BPE = the GPT-2 / Llama / StarCoder recipe:
* base alphabet is the 256 bytes  -> zero out-of-vocabulary, ever
* a ByteLevel pre-tokenizer applies the GPT-2 regex split, then maps bytes to
  a reversible set of unicode chars (so the JSON is text-safe)
* a ByteLevel decoder inverts it exactly -> ``decode(encode(x)) == x``

For Python specifically, the trainer naturally learns indentation tokens (runs
of 4/8/12 spaces) and high-frequency tokens (``self``, ``def``, ``return``,
``):``) because they dominate the corpus.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pyllm.tokenizer.tokenizer import DEFAULT_SPECIAL_TOKENS, PyTokenizer


def train_bpe_tokenizer(
    corpus: Iterable[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> PyTokenizer:
    """Train and return a :class:`PyTokenizer`.

    Args:
        corpus: an iterable of text chunks (e.g. file contents). Streamed, so it
            need not fit in memory.
        vocab_size: target total vocab, *including* the 256 bytes and the
            special tokens. Must match ``ModelConfig.vocab_size`` later.
        special_tokens: added first, getting the lowest ids; defaults to
            :data:`DEFAULT_SPECIAL_TOKENS`.
        min_frequency: a pair must occur at least this many times to be merged.
        show_progress: print a progress bar (nice for the full stdlib run).
    """
    if special_tokens is None:
        special_tokens = list(DEFAULT_SPECIAL_TOKENS)

    # BPE model with no <unk>: byte-level means every input is representable.
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space=False: do NOT inject a leading space (would corrupt code).
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        # Seed the vocab with all 256 byte-level symbols so nothing is unseen.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=min_frequency,
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return PyTokenizer(tokenizer)


def iter_python_files(root: str, limit: int | None = None) -> Iterator[str]:
    """Yield the text of ``*.py`` files under ``root`` (recursively).

    Reads with ``errors="ignore"`` so a stray bad byte in one file can't abort a
    long training run. Kept out of the trainer so it's independently testable.
    """
    import os

    count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    yield f.read()
            except (OSError, UnicodeError):
                continue
            count += 1
            if limit is not None and count >= limit:
                return
