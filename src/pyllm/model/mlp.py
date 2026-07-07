"""SwiGLU feed-forward network (Shazeer, 2020; used in Llama/PaLM/Qwen).

A classic Transformer MLP is  down(act(up(x)))  with 2 matrices and hidden = 4d.
SwiGLU is a *gated* variant with 3 matrices:

    SwiGLU(x) = down( SiLU(gate(x)) * up(x) )        (* is elementwise)

One branch (gate) is squashed by SiLU and multiplies (gates) the other (up).
Gating gives the network a cheap, data-dependent on/off control per hidden unit,
which empirically improves quality per parameter.

The "2/3" hidden-size rule (see ModelConfig._compute_ffn_hidden): to spend the
same parameter budget as a 2-matrix MLP with hidden 4d, we set the 3-matrix
hidden to (2/3)*4d, because  3 * d * (2/3 * 4d) = 8 d^2 = 2 * d * 4d.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from pyllm.config import ModelConfig


class SwiGLU(nn.Module):
    """Gated feed-forward block. Shapes: (B, T, d_model) -> (B, T, d_model)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.ffn_hidden
        assert h is not None  # set by ModelConfig.__post_init__
        self.gate_proj = nn.Linear(d, h, bias=cfg.bias)  # -> SiLU
        self.up_proj = nn.Linear(d, h, bias=cfg.bias)    # -> gated value
        self.down_proj = nn.Linear(h, d, bias=cfg.bias)  # back to d_model
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        # SiLU(gate(x)) gates up(x) elementwise, then project down.
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))
