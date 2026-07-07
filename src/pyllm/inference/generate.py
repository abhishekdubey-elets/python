"""Autoregressive generation with a KV cache (and a streaming variant).

The cache is what makes generation efficient: we run the full prompt once
(prefill), then feed **one** new token per step, reusing cached keys/values.
Cost per new token is O(context), not O(context^2).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor

from pyllm.inference.sampler import sample_next


@torch.no_grad()
def generate(
    model,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate up to ``max_new_tokens`` tokens after the prompt ``idx`` (B, T).

    Returns the full sequence (prompt + generated), (B, T + n).
    """
    model.eval()
    seq_len = model.cfg.seq_len
    B = idx.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)

    logits, _, past = model(idx, use_cache=True)  # prefill
    for _ in range(max_new_tokens):
        next_tok = sample_next(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)
        if eos_id is not None:
            # keep already-finished rows emitting eos, then update finished mask
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            finished |= next_tok == eos_id
        idx = torch.cat([idx, next_tok[:, None]], dim=1)
        if (eos_id is not None and bool(finished.all())) or idx.size(1) >= seq_len:
            break
        logits, _, past = model(next_tok[:, None], past_kvs=past, use_cache=True)
    return idx


@torch.no_grad()
def generate_stream(
    model,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Yield each newly generated token ``(B,)`` as it is produced (for live output)."""
    model.eval()
    seq_len = model.cfg.seq_len
    cur_len = idx.size(1)
    logits, _, past = model(idx, use_cache=True)
    for _ in range(max_new_tokens):
        next_tok = sample_next(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)
        yield next_tok
        cur_len += 1
        if (eos_id is not None and bool((next_tok == eos_id).all())) or cur_len >= seq_len:
            break
        logits, _, past = model(next_tok[:, None], past_kvs=past, use_cache=True)
