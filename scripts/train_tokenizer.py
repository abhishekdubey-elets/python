"""CLI: train a byte-level BPE tokenizer on a directory of Python files.

Example (train on the local Python standard library — a real, PSF-licensed
Python corpus that ships with your interpreter, no download needed):

    python scripts/train_tokenizer.py \
        --input-dir "$(python -c 'import sysconfig;print(sysconfig.get_path(\"stdlib\"))')" \
        --vocab-size 50257 \
        --output tokenizer/pyllm.json \
        --show-progress

Then in Python:  PyTokenizer.load("tokenizer/pyllm.json")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without `pip install -e .` by putting src/ on the path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pyllm.tokenizer.train import iter_python_files, train_bpe_tokenizer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on Python code.")
    p.add_argument("--input-dir", required=True, help="Directory to scan recursively for *.py")
    p.add_argument("--vocab-size", type=int, default=50257, help="Total vocab incl. specials")
    p.add_argument("--output", default="tokenizer/pyllm.json", help="Where to save the tokenizer")
    p.add_argument("--limit", type=int, default=None, help="Max number of files (for quick runs)")
    p.add_argument("--min-frequency", type=int, default=2, help="Min pair frequency to merge")
    p.add_argument("--show-progress", action="store_true")
    args = p.parse_args()

    print(f"Scanning {args.input_dir} for *.py (limit={args.limit}) ...")
    corpus = iter_python_files(args.input_dir, limit=args.limit)

    tok = train_bpe_tokenizer(
        corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=args.show_progress,
    )
    tok.save(args.output)

    print(f"Done. vocab_size={tok.vocab_size}  eos_id={tok.eos_id}  saved -> {args.output}")

    # Quick smoke test: round-trip a snippet of Python.
    sample = "def add(a, b):\n    return a + b  # 4-space indent\n"
    ids = tok.encode(sample)
    ok = tok.decode(ids) == sample
    print(f"round-trip ok={ok}  ({len(sample)} chars -> {len(ids)} tokens)")


if __name__ == "__main__":
    main()
