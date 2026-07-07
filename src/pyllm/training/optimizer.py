"""AdamW with correct weight-decay parameter groups.

Weight decay should apply to matmul weights (embeddings, linear layers) but NOT
to 1-D parameters (RMSNorm gains, biases). Decaying norms/biases toward zero
hurts — they're scales/offsets, not features. This split is standard (nanoGPT,
Llama) and matters more than people expect.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    device: str = "cpu",
) -> torch.optim.AdamW:
    """Build AdamW with two param groups: decay (>=2D) and no-decay (<2D)."""
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # weights (matmuls, embeddings) are >=2D; norms/biases are 1D.
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # Use the fused CUDA kernel when available (faster); harmless to probe on CPU.
    fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_ok and device == "cuda"
    extra = {"fused": True} if use_fused else {}
    return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)
