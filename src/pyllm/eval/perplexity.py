"""Perplexity — the standard intrinsic language-model metric.

Perplexity = exp(average per-token cross-entropy). Intuition: the effective
number of equally-likely choices the model is deciding among at each step. Lower
is better; PPL = 1 is perfect, PPL = vocab_size is random. It's cheap and
correlates with quality, but it does NOT measure whether generated code *runs* —
that's what the pass@k harness (codegen.py) is for.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pyllm.data.loader import get_batch


@torch.no_grad()
def evaluate_perplexity(
    model,
    data: np.ndarray,
    seq_len: int,
    batch_size: int = 8,
    max_batches: int = 100,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Estimate mean loss and perplexity over random blocks of ``data``."""
    model.eval()
    total_loss, n = 0.0, 0
    for _ in range(max_batches):
        x, y = get_batch(data, batch_size, seq_len, device=device, generator=generator)
        _, loss, _ = model(x, y)
        total_loss += loss.item()
        n += 1
    mean_loss = total_loss / max(n, 1)
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}
