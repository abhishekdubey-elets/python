"""CLI: generate Python from a trained checkpoint.

Example:
    python scripts/generate.py --checkpoint checkpoints/ckpt.pt \
        --tokenizer tokenizer/pyllm.json --prompt "def fibonacci(n):" \
        --max-new-tokens 128 --temperature 0.8 --top-p 0.95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from pyllm.inference import generate  # noqa: E402
from pyllm.model import PyLLM  # noqa: E402
from pyllm.tokenizer import PyTokenizer  # noqa: E402
from pyllm.training import load_checkpoint, model_config_from_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a pyllm checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="def ")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    model_cfg = model_config_from_checkpoint(ckpt)
    model = PyLLM(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = PyTokenizer.load(args.tokenizer)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)

    out = generate(
        model, ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        greedy=args.greedy, eos_id=tok.eos_id,
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
