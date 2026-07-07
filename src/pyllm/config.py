"""Configuration objects for pyllm.

Design philosophy (see docs/01_config_and_layout.md):

* **Config is data, not code.** A model or training run is *fully* described by
  these dataclasses. Serialize to YAML -> reproduce exactly later. Nothing about
  the architecture lives as a magic constant inside a module.
* **Validate once, at construction.** ``__post_init__`` checks invariants and
  fails fast with a readable message, instead of a cryptic reshape error deep in
  a forward pass 40 minutes into training.
* **Single source of truth for derived values.** You set the user-facing knobs
  (``d_model``, ``n_heads``); the code derives ``head_dim``, ``ffn_hidden``,
  ``n_kv_heads`` from them. They can never drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T", bound="_YamlMixin")


class _YamlMixin:
    """Give a dataclass lossless (dict|YAML) <-> object conversion.

    We deliberately do NOT use ``dataclasses.asdict`` because it also emits
    ``init=False`` derived fields (e.g. ``head_dim``). We only serialize the
    user-facing + concrete-derived knobs, and let ``__post_init__`` recompute
    the purely-derived ones on load. That keeps YAML files clean and reload
    unambiguous.
    """

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
        data = data or {}
        known = {f.name for f in fields(cls) if f.init}  # type: ignore[arg-type]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown config keys for {cls.__name__}: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}"
            )
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict[str, Any]:
        # Only ``init=True`` fields; excludes purely-derived (init=False) fields.
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


@dataclass(eq=True)
class ModelConfig(_YamlMixin):
    """Everything needed to *construct* the network. No training info here.

    Fields with a default of ``None`` are "derive if not given": you may set
    them explicitly, otherwise ``__post_init__`` fills them from the other knobs.
    """

    # --- core dimensions ------------------------------------------------------
    vocab_size: int = 50257          # GPT-2 tokenizer size (our chosen default)
    d_model: int = 768               # residual-stream / embedding width
    n_layers: int = 12               # number of transformer blocks
    n_heads: int = 12                # query heads
    n_kv_heads: int | None = None    # key/value heads; None => == n_heads (plain MHA)
    seq_len: int = 1024              # max context length used for training

    # --- feed-forward (SwiGLU) ------------------------------------------------
    # SwiGLU uses 3 matrices instead of 2, so to keep the parameter count close
    # to a plain 4x GELU-MLP we scale the hidden size by 2/3, then round up to a
    # hardware-friendly multiple. (Full derivation lands in the Phase 4 MLP doc.)
    ffn_mult: float = 4.0
    ffn_multiple_of: int = 256
    ffn_hidden: int | None = None    # derived from the three fields above if None

    # --- positional encoding (RoPE) ------------------------------------------
    rope_theta: float = 10000.0

    # --- normalization --------------------------------------------------------
    norm_eps: float = 1e-5

    # --- regularization -------------------------------------------------------
    dropout: float = 0.0

    # --- misc knobs -----------------------------------------------------------
    tie_weights: bool = True         # share input embedding with output head
    bias: bool = False               # biases in linear layers (modern LLMs: off)

    # --- derived (never set by the user; recomputed on load) ------------------
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        # ---- validate the raw knobs -----------------------------------------
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"ModelConfig.{name} must be > 0, got {getattr(self, name)}")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {self.n_kv_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            # GQA groups query heads over kv heads; the grouping must be even.
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) for grouped-query attention."
            )

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        # ---- derive ----------------------------------------------------------
        self.head_dim = self.d_model // self.n_heads

        if self.ffn_hidden is None:
            self.ffn_hidden = self._compute_ffn_hidden()
        if self.ffn_hidden <= 0:
            raise ValueError(f"ffn_hidden must be > 0, got {self.ffn_hidden}")

    def _compute_ffn_hidden(self) -> int:
        """Llama-style SwiGLU hidden size: 2/3 * mult * d_model, rounded up."""
        hidden = int(self.ffn_mult * self.d_model)
        hidden = int(2 * hidden / 3)
        m = self.ffn_multiple_of
        return m * ((hidden + m - 1) // m)  # round UP to the next multiple of m

    def estimate_params(self) -> int:
        """Predict the total parameter count from config alone.

        This mirrors what the real ``nn.Module`` will allocate in Phase 4, so we
        can unit-test that the model matches this prediction. Biases and dropout
        contribute ~nothing and are handled for completeness.
        """
        d, h = self.d_model, self.ffn_hidden
        kv_dim = self.n_kv_heads * self.head_dim  # type: ignore[operator]

        # token embedding table (tied => counted once; it doubles as the head)
        embed = self.vocab_size * d

        # per-block: attention projections (q, k, v, o) + SwiGLU (gate, up, down)
        attn = d * d + 2 * (d * kv_dim) + d * d          # q, k, v, o
        mlp = 3 * (d * h)                                # gate, up, down
        norms = 2 * d                                    # two RMSNorm weight vecs
        if self.bias:
            attn += d + 2 * kv_dim + d
            mlp += 2 * h + d
        per_block = attn + mlp + norms

        total = embed + self.n_layers * per_block + d    # + final RMSNorm
        if not self.tie_weights:
            total += self.vocab_size * d                 # separate output head
        return total


@dataclass(eq=True)
class TrainConfig(_YamlMixin):
    """Everything about *how* we train. Knows nothing about the architecture."""

    # --- optimization ---------------------------------------------------------
    lr: float = 3e-4
    min_lr: float = 3e-5              # cosine decay floor
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0           # max global grad norm (0 disables)

    # --- schedule -------------------------------------------------------------
    warmup_steps: int = 100
    max_steps: int = 5000

    # --- batching -------------------------------------------------------------
    micro_batch_size: int = 8        # sequences per forward pass
    grad_accum_steps: int = 4        # micro-batches accumulated per optimizer step

    # --- runtime --------------------------------------------------------------
    device: str = "cpu"
    dtype: str = "float32"           # "float32" | "bfloat16" | "float16"
    compile: bool = False            # torch.compile (off on CPU)
    grad_checkpointing: bool = False
    seed: int = 1337

    # --- eval / logging / checkpoints ----------------------------------------
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"

    _VALID_DTYPES = ("float32", "bfloat16", "float16")

    def __post_init__(self) -> None:
        if self.dtype not in self._VALID_DTYPES:
            raise ValueError(f"dtype must be one of {self._VALID_DTYPES}, got {self.dtype!r}")
        for name in ("warmup_steps", "max_steps", "micro_batch_size", "grad_accum_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"TrainConfig.{name} must be > 0, got {getattr(self, name)}")
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) cannot exceed max_steps ({self.max_steps})."
            )
        for name in ("beta1", "beta2"):
            if not (0.0 < getattr(self, name) < 1.0):
                raise ValueError(f"TrainConfig.{name} must be in (0, 1), got {getattr(self, name)}")

    def tokens_per_step(self, seq_len: int) -> int:
        """Effective tokens per optimizer step (the number that actually matters
        for the learning-rate schedule and 'how much has the model seen')."""
        return self.micro_batch_size * self.grad_accum_steps * seq_len


@dataclass(eq=True)
class DataConfig(_YamlMixin):
    """Where the data and tokenizer live. Knows nothing about the model."""

    data_dir: str = "data"
    train_split: str = "train"
    val_split: str = "val"
    tokenizer_path: str = "tokenizer/pyllm.json"
    num_workers: int = 0             # 0 is safest/most portable, esp. on Windows

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
