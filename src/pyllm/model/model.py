"""The full decoder-only Transformer: embeddings -> N blocks -> norm -> LM head.

Design notes:
* **Weight tying**: the output head reuses the input embedding matrix. At our
  scale that saves ~vocab*d_model params (~38M for the 125M config) and tends to
  improve quality.
* **RoPE, no learned position embeddings**: position is injected inside attention.
* **Scaled residual init** (GPT-2 trick): the output projections of each sublayer
  (``o_proj``, ``down_proj``) are initialized with std scaled by 1/sqrt(2*n_layers)
  so the residual stream's variance doesn't grow with depth.
* **KV cache**: ``forward`` accepts and returns per-layer caches for O(1)/token
  incremental decoding (used in Phase 7).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pyllm.config import ModelConfig
from pyllm.model.attention import KVCache
from pyllm.model.block import Block
from pyllm.model.rmsnorm import RMSNorm
from pyllm.model.rope import RotaryEmbedding


class PyLLM(nn.Module):
    """A compact decoder-only Transformer configured entirely by ModelConfig."""

    def __init__(self, cfg: ModelConfig, use_sdpa: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = False  # toggled by the trainer (Phase 6)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # One shared RoPE table for all layers (positions are identical per layer).
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList([Block(cfg, use_sdpa=use_sdpa) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # tie: share the matrix

        self.apply(self._init_weights)
        # Scaled init for residual-path output projections (GPT-2 §2.3).
        std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, include_embedding: bool = True) -> int:
        """Total trainable parameters. ``parameters()`` counts the tied
        embedding/head matrix once, so this matches ``ModelConfig.estimate_params``."""
        n = sum(p.numel() for p in self.parameters())
        if not include_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        past_kvs: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, list[KVCache] | None]:
        """Args:
            idx: (B, T) input token ids.
            targets: (B, T) next-token labels; if given, also returns the loss.
                Positions with id -100 are ignored (padding/masking).
            past_kvs: per-layer KV caches from a previous call (incremental decode).
            use_cache: if True, collect and return updated per-layer caches.

        Returns ``(logits, loss, new_caches)``.
            logits: (B, T, vocab_size); loss: scalar or None; new_caches: list or None.
        """
        B, T = idx.shape
        # Absolute position offset for RoPE = number of cached timesteps.
        offset = 0 if past_kvs is None else past_kvs[0][0].size(2)
        if offset + T > self.cfg.seq_len:
            raise ValueError(f"sequence length {offset + T} exceeds seq_len={self.cfg.seq_len}")

        x = self.drop(self.tok_emb(idx))                 # (B, T, d_model)
        cos, sin = self.rope(T, offset)                  # (T, head_dim)

        new_caches: list[KVCache] | None = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            if self.grad_checkpointing and self.training and not use_cache:
                # Recompute activations in backward to save memory (Phase 6).
                x, present = checkpoint(block, x, cos, sin, past, use_reentrant=False)
            else:
                x, present = block(x, cos, sin, past)
            if new_caches is not None:
                new_caches.append(present)

        x = self.norm_f(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss, new_caches
