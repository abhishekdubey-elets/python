# 01 — Config system & repository layout

> Phase 2. The goal of this phase is a small, installable, tested skeleton that
> every later phase drops into. No model math yet — just the harness.

## 1. Intuition

A trained model is only reproducible if you can answer, exactly: *what network
was this, and how was it trained?* If those answers live as scattered constants
(`d_model = 768` typed in `attention.py`, `1e-5` typed in `rmsnorm.py`), you can
never reconstruct a run, sweep architectures, or resume training safely.

So we make the answer a **single object you can serialize**: a config. The model
is a pure function of `ModelConfig`; the run is a pure function of
`ModelConfig + TrainConfig + DataConfig`. Save those three and you can rebuild
anything.

## 2. Why dataclasses (not dicts, not argparse namespaces)

| Option | Problem |
|---|---|
| Plain `dict` | No validation, no autocomplete, typos are silent (`cfg["d_moel"]` → `KeyError` at runtime, far from the cause). |
| `argparse.Namespace` | Same as dict, plus it couples config to the CLI. |
| Global constants | Can't serialize, can't have two configs at once, can't sweep. |
| **`@dataclass`** | Typed fields, IDE autocomplete, `__eq__`/`__repr__` for free, one obvious place for validation (`__post_init__`), trivially convertible to/from YAML. |

## 3. The "derive, don't duplicate" rule

Some values are *functions of others* and must never be typed independently:

```
head_dim   = d_model // n_heads          # and we assert d_model % n_heads == 0
n_kv_heads = n_heads                      # if not given (=> plain MHA)
ffn_hidden = round_up(2/3 * mult * d_model, multiple_of)   # SwiGLU sizing
```

These are computed in `__post_init__`. `head_dim` is marked `field(init=False)`:
you cannot pass it in, and it is **not serialized** (it would be redundant and
could contradict `d_model`/`n_heads` on reload). `n_kv_heads` and `ffn_hidden`
*are* serialized once concrete, so a saved config is fully explicit.

### The SwiGLU 2/3 factor (preview)

A classic Transformer MLP has 2 matrices and hidden size `4 * d_model`, giving
`2 * (d_model * 4 d_model) = 8 d_model²` params. SwiGLU uses **3** matrices
(gate, up, down). To spend the *same* parameter budget we shrink the hidden size
by 2/3: `3 * (d_model * (2/3 * 4 d_model)) = 8 d_model²`. Same budget, better
quality. We then round up to a multiple (256) so matrix dims are hardware
friendly. Full derivation in the Phase 4 MLP doc — here we just *compute* it.

## 4. Validate at construction, fail loud

`ModelConfig(d_model=768, n_heads=10)` raises immediately:

```
ValueError: d_model (768) must be divisible by n_heads (10).
```

Compare the alternative: this misconfig would otherwise blow up as an opaque
tensor-reshape error inside the attention forward, mid-training. **The best time
to catch a bad config is the microsecond you create it.**

## 5. Why the `src/` layout

The package lives in `src/pyllm/`, not top-level. This forces
`pip install -e .`, which means:

* `import pyllm` works from *any* directory (no "only runs from repo root" bugs).
* Tests import the **installed** package — exactly what a user would import.
* There's a clean boundary: `src/` = importable library, `scripts/` = CLI glue,
  `configs/` = data, `docs/` = prose, `tests/` = checks.

## 6. Worked example

```python
from pyllm import ModelConfig, TrainConfig

m = ModelConfig.from_yaml("configs/model_125m.yaml")
m.head_dim              # 64      (derived: 768 // 12)
m.ffn_hidden           # 2048    (derived: round_up(2/3 * 4 * 768, 256))
m.estimate_params()    # ~123.5M

t = TrainConfig.from_yaml("configs/train_default.yaml")
t.tokens_per_step(m.seq_len)   # 8 * 4 * 1024 = 32768 tokens/optimizer-step

m.to_yaml("run_config.yaml")   # snapshot for reproducibility/resume
```

## 7. Parameter-count arithmetic (so you can predict, not guess)

For the 125M config (`d=768`, `L=12`, `vocab=50257`, tied, `h_ffn=2048`, MHA):

```
embedding (tied, also the head):  vocab * d          = 50257*768 ≈ 38.60M
per block:
  attention q,k,v,o:              4 * d*d            = 4*768*768  = 2.36M
  SwiGLU gate,up,down:            3 * d*h            = 3*768*2048 = 4.72M
  2 x RMSNorm weights:            2 * d              = 1536
  -> per block                                       ≈ 7.08M
blocks:                           12 * 7.08M         ≈ 84.95M
final RMSNorm:                    d                  = 768
------------------------------------------------------------------
total                                                ≈ 123.5M
```

Note ~31% of the model is the embedding table — a direct consequence of choosing
`vocab_size = 50257` at this small scale. `estimate_params()` encodes exactly
this arithmetic, and a Phase 4 test will assert the real `nn.Module` matches it.

## 8. Common mistakes

* **Typing `head_dim` by hand** → drifts from `d_model`. Always derive.
* **Serializing derived fields** → reload can contradict the source knobs.
* **Validating late** → cryptic crashes deep in training. Validate in `__post_init__`.
* **`num_workers > 0` on Windows without a `__main__` guard** → dataloader hangs.
  We default to `0`; we'll revisit in Phase 5.
* **fp16 without loss scaling** → NaNs. We default to fp32/bf16 (Phase 6).

## 9. References

* nanoGPT — Karpathy — the spiritual template for simplicity.
* Llama / Llama 2 papers — RoPE, RMSNorm, SwiGLU, the 2/3 FFN sizing.
* Python packaging user guide — the `src/` layout rationale.
