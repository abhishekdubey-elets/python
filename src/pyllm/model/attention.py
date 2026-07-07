"""Multi-head self-attention with RoPE, causal masking, GQA, and a KV cache.

Pipeline for input x of shape (B, T, d_model):
  1. project to queries/keys/values, split into heads
  2. apply RoPE to Q and K (relative positions)
  3. (optional) prepend cached K/V for incremental decoding
  4. (GQA) repeat K/V so each query head has a K/V head to attend to
  5. scaled-dot-product attention with a causal mask
  6. recombine heads and project out

Two attention implementations are provided and tested to agree:
  * ``use_sdpa=True``  -> torch.nn.functional.scaled_dot_product_attention
    (picks a fused/Flash kernel when available)
  * ``use_sdpa=False`` -> an explicit, readable reference (for learning + tests)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pyllm.config import ModelConfig
from pyllm.model.rope import apply_rotary_emb

# A single layer's cache: (keys, values), each (B, n_kv_heads, T, head_dim).
KVCache = tuple[Tensor, Tensor]


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand K/V heads for GQA: (B, n_kv, T, hd) -> (B, n_kv*n_rep, T, hd).

    Each of the n_kv key/value heads is shared by ``n_rep`` query heads. We use
    expand+reshape (a view, no copy until reshape) rather than a real tensor of
    duplicated data where possible.
    """
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, t, hd)
        .reshape(b, n_kv * n_rep, t, hd)
    )


class Attention(nn.Module):
    """Causal self-attention. Supports MHA (n_kv_heads == n_heads) and GQA."""

    def __init__(self, cfg: ModelConfig, use_sdpa: bool = True) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        assert self.n_kv_heads is not None
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # query heads per kv head
        self.use_sdpa = use_sdpa
        self.attn_dropout_p = cfg.dropout

        d = cfg.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        # Q keeps all heads; K/V may have fewer heads under GQA.
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(d, kv_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(d, kv_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        """(B, T, d_model) -> (B, T, d_model), plus the updated (K, V) cache.

        ``cos``/``sin`` are (T, head_dim) for the *current* T positions (the model
        slices them with the right offset when a cache is present).
        """
        B, T, _ = x.shape

        # --- project and reshape to (B, heads, T, head_dim) ------------------
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # --- RoPE on Q and K (V is never rotated) ----------------------------
        q, k = apply_rotary_emb(q, k, cos, sin)

        # --- KV cache: prepend past keys/values (already rotated) ------------
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present: KVCache = (k, v)  # cache stores the compact (pre-GQA-expand) K/V

        # No mask needed when a cache is present (a new query attends to all past
        # keys, all causal); full/prefill passes are causal.
        is_causal = past_kv is None

        # --- GQA: give every query head a key/value head ---------------------
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # --- attention -------------------------------------------------------
        if self.use_sdpa:
            out = self._sdpa(q, k, v, is_causal)
        else:
            out = self._manual(q, k, v, is_causal)

        # --- recombine heads and project out ---------------------------------
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(out)), present

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        p = self.attn_dropout_p if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=is_causal)

    def _manual(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        # scores: (B, n_heads, Tq, Tk)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if is_causal:
            Tq, Tk = q.size(-2), k.size(-2)
            # Lower-triangular mask aligned to the bottom-right so it also works
            # if Tq != Tk (query i may attend keys up to column i + (Tk - Tq)).
            allowed = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq)
            scores = scores.masked_fill(~allowed, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout_p, training=self.training)
        return attn @ v
