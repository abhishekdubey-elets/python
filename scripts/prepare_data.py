"""CLI: turn a directory of .py files into packed train/val token shards.

Pipeline: read -> clean/validate -> exact dedup -> (optional) near-dedup ->
tokenize+pack. Requires a trained tokenizer (see train_tokenizer.py).

Example:
    python scripts/prepare_data.py --input-dir data/raw \
        --tokenizer tokenizer/pyllm.json --out-dir data/packed --near-dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pyllm.data import clean_document, dedup_exact, dedup_near, pack_dataset  # noqa: E402
from pyllm.data.pack import SupportsEncode  # noqa: E402
from pyllm.tokenizer import PyTokenizer  # noqa: E402


def _read_py_files(root: str, limit: int | None) -> list[str]:
    import os

    out: list[str] = []
    for dp, _dn, fns in os.walk(root):
        for name in fns:
            if not name.endswith(".py"):
                continue
            try:
                with open(os.path.join(dp, name), encoding="utf-8", errors="ignore") as f:
                    out.append(f.read())
            except OSError:
                continue
            if limit and len(out) >= limit:
                return out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Clean, dedup, and pack Python files.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--tokenizer", required=True, help="Path to a trained tokenizer JSON")
    p.add_argument("--out-dir", default="data/packed")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--near-dedup", action="store_true", help="Also remove near-duplicates")
    args = p.parse_args()

    raw = _read_py_files(args.input_dir, args.limit)
    print(f"read {len(raw)} files")

    cleaned = [d for d in (clean_document(t) for t in raw) if d is not None]
    print(f"after clean/validate: {len(cleaned)}")

    keep = dedup_exact(cleaned)
    docs = [cleaned[i] for i in keep]
    print(f"after exact dedup: {len(docs)}")

    if args.near_dedup:
        keep = dedup_near(docs)
        docs = [docs[i] for i in keep]
        print(f"after near dedup: {len(docs)}")

    tok: SupportsEncode = PyTokenizer.load(args.tokenizer)
    stats = pack_dataset(docs, tok, args.out_dir, val_fraction=args.val_fraction)
    print(f"packed -> {args.out_dir}: {stats}")


if __name__ == "__main__":
    main()
