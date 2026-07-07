"""Tests for RoPE (Phase 4).

The single most important property of RoPE is **relativity**: the attention
score between a query at position m and a key at position n depends only on
(m - n). We test that directly, plus norm-preservation (rotation is orthogonal),
the offset API (needed for KV-cache decoding), and basic shape/validation.
"""

from __future__ import annotations

import pytest
import torch

from pyllm.model import RotaryEmbedding, apply_rotary_emb, rotate_half


def test_rotate_half():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])  # halves: [1,2] and [3,4]
    assert torch.equal(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_apply_preserves_shape_and_dtype():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    cos, sin = rope(seq_len=8)
    qr, kr = apply_rotary_emb(q, k, cos, sin)
    assert qr.shape == q.shape and kr.shape == k.shape


def test_rope_preserves_norm():
    # Rotation is orthogonal, so per-vector L2 norm must be unchanged.
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    q = torch.randn(2, 4, 8, 16)
    cos, sin = rope(seq_len=8)
    qr, _ = apply_rotary_emb(q, q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5)


def test_rope_is_relative():
    """<RoPE(q, m), RoPE(k, n)> must depend only on (m - n)."""
    torch.manual_seed(0)
    head_dim, T = 16, 8
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=32)
    cos, sin = rope(seq_len=T)

    # Same q vector at every position, same k vector at every position.
    qv = torch.randn(head_dim)
    kv = torch.randn(head_dim)
    q = qv.expand(1, 1, T, head_dim).contiguous()
    k = kv.expand(1, 1, T, head_dim).contiguous()
    qr, kr = apply_rotary_emb(q, k, cos, sin)
    qr, kr = qr[0, 0], kr[0, 0]  # (T, head_dim)

    scores = qr @ kr.T  # scores[m, n] = <RoPE(q, m), RoPE(k, n)>
    # Diagonals (constant m - n) must be equal: scores[m, n] == scores[m+1, n+1].
    for m in range(T - 1):
        for n in range(T - 1):
            assert torch.allclose(scores[m, n], scores[m + 1, n + 1], atol=1e-4)


def test_offset_matches_slice():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    cos_off, sin_off = rope(seq_len=4, offset=10)
    cos_full, sin_full = rope(seq_len=64)
    assert torch.allclose(cos_off, cos_full[10:14])
    assert torch.allclose(sin_off, sin_full[10:14])


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError, match="head_dim must be even"):
        RotaryEmbedding(head_dim=15, max_seq_len=8)


def test_exceeding_max_seq_len_rejected():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=8)
    with pytest.raises(ValueError, match="exceed max_seq_len"):
        rope(seq_len=4, offset=6)  # 6 + 4 = 10 > 8
