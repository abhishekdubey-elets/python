"""Decoding strategies: temperature, top-k, top-p (nucleus), and greedy.

All operate on the last-position logits ``(B, vocab)`` and return sampled token
ids ``(B,)``. They compose: temperature first (reshape the distribution), then a
truncation filter (top-k / top-p) to cut the unreliable tail, then sample.

When to use which:
  * greedy (temperature 0): deterministic; best for exact/structured output.
  * temperature: <1 sharpens (safer), >1 flattens (more diverse).
  * top-k: keep the k most likely tokens — simple, fixed budget.
  * top-p: keep the smallest set whose mass >= p — adapts to how peaked the
    distribution is; usually the best default for open-ended generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

NEG_INF = float("-inf")


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy for deterministic).")
    return logits / temperature


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep only the top-k logits per row; set the rest to -inf."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]  # k-th largest per row
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative prob >= p."""
    if not 0.0 < p < 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    # Remove tokens once cumulative prob has exceeded p, but always keep the first.
    remove = cumsum - probs > p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, NEG_INF)
    # scatter back to the original vocab order
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def sample_next(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pick the next token id ``(B,)`` from last-position logits ``(B, vocab)``."""
    if greedy or temperature == 0:
        return logits.argmax(dim=-1)
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
