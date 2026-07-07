"""Tests for causal self-attention (Phase 4).

Key properties:
* shape is preserved
* the SDPA (fused/Flash) path and the manual reference agree numerically
* GQA (n_kv_heads < n_heads) works and preserves shape
* causal masking: output at position t must not depend on tokens > t
* KV-cache incremental decoding matches a single full forward
"""

from __future__ import annotations

import torch

from pyllm.config import ModelConfig
from pyllm.model import Attention
from pyllm.model.rope import RotaryEmbedding


def _cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=8, seq_len=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _rope(cfg: ModelConfig):
    return RotaryEmbedding(cfg.head_dim, cfg.seq_len, cfg.rope_theta)


def test_shape_preserved():
    cfg = _cfg()
    attn = Attention(cfg).eval()
    x = torch.randn(2, 10, cfg.d_model)
    cos, sin = _rope(cfg)(10)
    out, present = attn(x, cos, sin)
    assert out.shape == x.shape
    k, v = present
    assert k.shape == (2, cfg.n_kv_heads, 10, cfg.head_dim)


def test_sdpa_matches_manual():
    cfg = _cfg()
    attn = Attention(cfg).eval()
    x = torch.randn(2, 12, cfg.d_model)
    cos, sin = _rope(cfg)(12)
    attn.use_sdpa = True
    out_sdpa, _ = attn(x, cos, sin)
    attn.use_sdpa = False
    out_manual, _ = attn(x, cos, sin)
    assert torch.allclose(out_sdpa, out_manual, atol=1e-4)


def test_gqa_shapes_and_agreement():
    # 8 query heads sharing 2 kv heads (group size 4).
    cfg = _cfg(n_heads=8, n_kv_heads=2)
    assert cfg.n_kv_heads == 2
    attn = Attention(cfg).eval()
    x = torch.randn(2, 9, cfg.d_model)
    cos, sin = _rope(cfg)(9)
    out, (k, v) = attn(x, cos, sin)
    assert out.shape == x.shape
    assert k.shape[1] == 2  # cache keeps the compact (pre-expand) kv heads
    # SDPA and manual still agree under GQA.
    attn.use_sdpa = False
    out2, _ = attn(x, cos, sin)
    assert torch.allclose(out, out2, atol=1e-4)


def test_causal_masking():
    """Changing token t must not change attention outputs at positions < t."""
    torch.manual_seed(0)
    cfg = _cfg()
    attn = Attention(cfg).eval()
    rope = _rope(cfg)
    x = torch.randn(1, 8, cfg.d_model)
    cos, sin = rope(8)
    out_a, _ = attn(x, cos, sin)

    x2 = x.clone()
    x2[:, 7, :] = torch.randn(cfg.d_model)  # perturb the LAST token only
    out_b, _ = attn(x2, cos, sin)

    # positions 0..6 must be unchanged; position 7 may change.
    assert torch.allclose(out_a[:, :7], out_b[:, :7], atol=1e-5)
    assert not torch.allclose(out_a[:, 7], out_b[:, 7], atol=1e-4)


def test_kv_cache_matches_full_forward():
    """Feeding tokens one at a time with a cache == one full forward."""
    torch.manual_seed(0)
    cfg = _cfg()
    attn = Attention(cfg).eval()
    rope = _rope(cfg)
    T = 6
    x = torch.randn(1, T, cfg.d_model)

    cos, sin = rope(T)
    full, _ = attn(x, cos, sin)

    # Incremental: one timestep at a time, carrying the cache and RoPE offset.
    past = None
    outs = []
    for t in range(T):
        cos_t, sin_t = rope(1, offset=t)
        out_t, past = attn(x[:, t : t + 1], cos_t, sin_t, past_kv=past)
        outs.append(out_t)
    incr = torch.cat(outs, dim=1)
    assert torch.allclose(full, incr, atol=1e-4)
