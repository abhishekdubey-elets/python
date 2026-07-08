# pyllm

A compact, Python-specialized **decoder-only Transformer (~125M–300M params)**, built
from scratch for learning. Inspired by nanoGPT (simplicity) and modern
Llama/Qwen/Gemma architecture (RoPE, RMSNorm, SwiGLU, GQA, weight tying, KV cache).

This repo is being built **phase by phase**. Each module is explained in `docs/`
before it is implemented, and ships with tests. See `docs/` for the "why" behind
every decision.

## Status

| Phase | What | State |
|------:|------|-------|
| 1 | Architecture & roadmap | ✅ (design) |
| 2 | Project skeleton & config system | ✅ |
| 3 | Tokenizer | ✅ |
| 4 | Model (RMSNorm, RoPE, SwiGLU, attention, block, model) | ✅ |
| 5 | Data pipeline | ✅ |
| 6 | Training loop | ✅ |
| 7 | Inference (KV cache, sampling, streaming) | ✅ |
| 8 | Evaluation | ✅ |
| 9 | Optimizations & polish | ✅ |

**All 9 phases complete — 90 tests passing.**

## Quickstart

```bash
# from the python-llm/ directory
python -m venv .venv
.venv\Scripts\activate         # Windows (PowerShell/cmd)
# source .venv/bin/activate    # macOS/Linux

pip install -e ".[dev]"
pytest                          # Phase 2: config tests should pass
```

Load a config in Python:

```python
from pyllm import ModelConfig

cfg = ModelConfig.from_yaml("configs/model_125m.yaml")
print(cfg.head_dim, cfg.ffn_hidden)          # 64 2048
print(f"{cfg.estimate_params()/1e6:.1f}M params")
```

## Full workflow (CLI)

```bash
# 1. train a tokenizer on a Python corpus (e.g. the local stdlib)
python scripts/train_tokenizer.py --input-dir <py-dir> --vocab-size 50257 \
    --output tokenizer/pyllm.json

# 2. clean, dedup, and pack data into train/val shards
python scripts/prepare_data.py --input-dir <py-dir> \
    --tokenizer tokenizer/pyllm.json --out-dir data/packed --near-dedup

# 3. train (use configs/toy.yaml on CPU; 125m/300m need a GPU)
python scripts/train.py --model-config configs/toy.yaml \
    --train-config configs/train_default.yaml --data-dir data/packed

# 4. generate
python scripts/generate.py --checkpoint checkpoints/ckpt.pt \
    --tokenizer tokenizer/pyllm.json --prompt "def fibonacci(n):" --top-p 0.95
```

## Layout

```
configs/     YAML configs (toy / 125m / 300m / train). The source of truth.
src/pyllm/   The installable package: config, tokenizer, model, training, ...
scripts/     Thin CLI entrypoints (argparse -> call into src/pyllm).
tests/       pytest; mostly shape + invariant tests.
docs/        The "why": one markdown per concept.
```

## Design principles

1. Config is data, code is logic.
2. Every module is independently testable.
3. Shapes are documented on every forward.
4. No hidden global state.
5. Minimal dependencies (PyTorch + a tokenizer lib; everything else we write).
6. Correctness before speed (optimizations behind config flags).
7. Architecture / training / data are cleanly separated.
