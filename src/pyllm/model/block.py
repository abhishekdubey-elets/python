"""A single Transformer block: pre-norm attention + pre-norm SwiGLU, both with
residual connections.

Pre-norm layout (as in Llama/GPT-NeoX):

    x = x + attn(norm1(x))
    x = x + mlp (norm2(x))

The residual stream (``x``) is never normalized in place — each sublayer reads a
normalized *copy*, and its output is added back. This keeps a clean, identity
gradient path from the loss all the way to the embeddings, which is what lets
deep stacks train stably without the warmup gymnastics post-norm needs.
"""

from __future__ import annotations

from torch import Tensor, nn

from pyllm.config import ModelConfig
from pyllm.model.attention import Attention, KVCache
from pyllm.model.mlp import SwiGLU
from pyllm.model.rmsnorm import RMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, use_sdpa=use_sdpa)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        """(B, T, d_model) -> (B, T, d_model), plus this layer's updated KV cache."""
        attn_out, present = self.attn(self.attn_norm(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, present
