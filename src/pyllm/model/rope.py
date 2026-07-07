"""Rotary Positional Embeddings (RoPE, Su et al., 2021).

Instead of *adding* a position vector, RoPE *rotates* each query/key vector by an
angle proportional to its position. Pairs of dimensions form 2D subspaces, and
subspace i is rotated by ``pos * theta_i`` with ``theta_i = base^(-2i/d)``.

The payoff: the attention score q_m . k_n depends only on the relative offset
(m - n), so the model gets relative-position awareness for free. Rotation is
orthogonal, so ||RoPE(x)|| == ||x|| (no scale distortion).

We use the "rotate_half" convention (as in Llama/HF): the head_dim is split into
two halves, and ``cos``/``sin`` tables of shape (T, head_dim) are formed by
concatenating the per-pair frequencies with themselves.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Map [x1, x2] (the two halves of the last dim) to [-x2, x1].

    This is the vectorized form of a 90-degree rotation within each 2D subspace,
    which is what lets ``x*cos + rotate_half(x)*sin`` implement the rotation.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate query and key tensors by the precomputed cos/sin.

    Shapes:
        q, k: (B, n_heads, T, head_dim)   (n_heads may differ for q vs k under GQA)
        cos, sin: (T, head_dim)
    Returns rotated (q, k) with the same shapes/dtype as the inputs.
    """
    orig_dtype = q.dtype
    # Do the rotation in fp32 for precision, then cast back (matters in bf16).
    q_f, k_f = q.float(), k.float()
    cos_b = cos.float()[None, None, :, :]  # (1, 1, T, head_dim) broadcast over B, heads
    sin_b = sin.float()[None, None, :, :]
    q_out = q_f * cos_b + rotate_half(q_f) * sin_b
    k_out = k_f * cos_b + rotate_half(k_f) * sin_b
    return q_out.to(orig_dtype), k_out.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Precomputes and serves the RoPE cos/sin tables.

    Args:
        head_dim: per-head dimension (must be even — dims are rotated in pairs).
        max_seq_len: largest position we precompute tables for.
        base: the RoPE ``theta`` (10000 is standard; larger extends context).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # theta_i = base^(-2i/d) for i in 0..d/2-1  -> shape (head_dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()             # positions 0..max-1
        freqs = torch.outer(t, inv_freq)                  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)           # (T, head_dim)

        # Buffers: move with .to(device)/.cuda() but are not trainable params and
        # are not saved in the state_dict (they're deterministic, recomputable).
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Return (cos, sin), each (seq_len, head_dim), for positions
        [offset, offset + seq_len). ``offset`` supports KV-cache decoding, where
        a new token must be rotated at its true absolute position."""
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"Requested positions up to {end} exceed max_seq_len={self.max_seq_len}."
            )
        return self.cos_cached[offset:end], self.sin_cached[offset:end]
