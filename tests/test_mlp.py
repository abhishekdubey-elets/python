"""Tests for the SwiGLU MLP (Phase 4)."""

from __future__ import annotations

import torch

from pyllm.config import ModelConfig
from pyllm.model import SwiGLU


def _cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4, seq_len=32)
    base.update(kw)
    return ModelConfig(**base)


def test_shape_preserved():
    mlp = SwiGLU(_cfg())
    x = torch.randn(2, 8, 64)
    assert mlp(x).shape == x.shape


def test_param_count_is_three_matrices():
    cfg = _cfg()
    mlp = SwiGLU(cfg)
    n = sum(p.numel() for p in mlp.parameters())
    assert n == 3 * cfg.d_model * cfg.ffn_hidden  # gate + up + down, no bias


def test_is_nonlinear():
    # A gated SiLU network must NOT be linear: f(2x) != 2 f(x) in general.
    torch.manual_seed(0)
    mlp = SwiGLU(_cfg())
    x = torch.randn(4, 64)
    assert not torch.allclose(mlp(2 * x), 2 * mlp(x), atol=1e-3)
