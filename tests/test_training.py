"""Tests for training (Phase 6).

The headline test is **overfit a tiny dataset**: the canonical way to prove a
training loop is wired correctly. If the loss can't be driven down on a handful
of tokens, something (loss, backward, optimizer, LR) is broken.
"""

from __future__ import annotations

import numpy as np
import torch

from pyllm.config import ModelConfig, TrainConfig
from pyllm.model import PyLLM
from pyllm.training import (
    Trainer,
    configure_optimizer,
    cosine_warmup_lr,
    load_checkpoint,
    model_config_from_checkpoint,
    save_checkpoint,
)


def _toy_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=64, d_model=64, n_layers=2, n_heads=4, seq_len=16)
    base.update(kw)
    return ModelConfig(**base)


# --- scheduler -------------------------------------------------------------- #
def test_schedule_warmup_then_decay():
    kw = dict(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert cosine_warmup_lr(0, **kw) < cosine_warmup_lr(5, **kw)          # ramping up
    assert abs(cosine_warmup_lr(9, **kw) - 1.0) < 1e-9                    # peak at end of warmup
    assert cosine_warmup_lr(9, **kw) > cosine_warmup_lr(55, **kw)         # decaying after
    assert abs(cosine_warmup_lr(100, **kw) - 0.1) < 1e-9                  # floors at min_lr
    assert abs(cosine_warmup_lr(200, **kw) - 0.1) < 1e-9                  # stays at floor


# --- optimizer groups ------------------------------------------------------- #
def test_optimizer_param_groups():
    model = PyLLM(_toy_cfg())
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    # every no-decay param is 1-D (norms/biases); every decay param is >=2D
    assert all(p.dim() < 2 for p in no_decay_group["params"])
    assert all(p.dim() >= 2 for p in decay_group["params"])


# --- checkpoint round-trip -------------------------------------------------- #
def test_checkpoint_roundtrip(tmp_path):
    cfg = _toy_cfg()
    model = PyLLM(cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=42, model_cfg=cfg)

    ckpt = load_checkpoint(path)
    assert ckpt["step"] == 42
    restored_cfg = model_config_from_checkpoint(ckpt)
    assert restored_cfg == cfg
    # weights load into a fresh model of the restored config
    fresh = PyLLM(restored_cfg)
    fresh.load_state_dict(ckpt["model"])
    for p, q in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p, q)


# --- the overfit test ------------------------------------------------------- #
def test_overfit_tiny_batch():
    torch.manual_seed(0)
    model_cfg = _toy_cfg()
    train_cfg = TrainConfig(
        lr=3e-3,
        min_lr=3e-4,
        warmup_steps=10,
        max_steps=300,
        micro_batch_size=4,
        grad_accum_steps=1,
        eval_interval=1000,   # skip eval during this short run
        log_interval=1000,
        device="cpu",
        dtype="float32",
    )
    # a tiny fixed token stream the model can memorize
    data = np.arange(200, dtype=np.uint16) % model_cfg.vocab_size

    model = PyLLM(model_cfg)
    trainer = Trainer(model, model_cfg, train_cfg, train_data=data)

    initial = trainer.estimate_loss()["train"]
    trainer.train()
    final = trainer.estimate_loss()["train"]

    assert final < initial - 1.0   # substantial drop
    assert final < 1.0             # actually memorized the tiny stream
