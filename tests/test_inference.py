"""Tests for inference: samplers + KV-cached generation (Phase 7)."""

from __future__ import annotations

import torch

from pyllm.config import ModelConfig
from pyllm.inference import generate, sample_next, top_k_filter, top_p_filter
from pyllm.inference.generate import generate_stream
from pyllm.model import PyLLM


def _model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, seq_len=32)
    return PyLLM(cfg).eval()


# --- samplers --------------------------------------------------------------- #
def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, 0.3]])
    assert sample_next(logits, greedy=True).item() == 1


def test_top_k_filter_keeps_k():
    logits = torch.arange(10, dtype=torch.float32)[None, :]
    filtered = top_k_filter(logits, k=3)
    assert torch.isfinite(filtered).sum().item() == 3  # only 3 survive


def test_top_p_keeps_nucleus():
    # one dominant token with >0.9 mass; top_p=0.9 should keep just it
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    filtered = top_p_filter(logits, p=0.9)
    assert torch.isfinite(filtered).sum().item() == 1


def test_top_k_restricts_sampled_support():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 50)
    seen = {sample_next(logits, temperature=1.0, top_k=5, generator=g).item() for _ in range(200)}
    assert len(seen) <= 5


# --- generation ------------------------------------------------------------- #
def test_generate_shapes_and_greedy_determinism():
    model = _model()
    idx = torch.randint(0, 64, (2, 4))
    out1 = generate(model, idx, max_new_tokens=6, greedy=True)
    out2 = generate(model, idx, max_new_tokens=6, greedy=True)
    assert out1.shape == (2, 10)
    assert torch.equal(out1, out2)  # greedy is deterministic


def test_generate_matches_no_cache_reference():
    """KV-cached greedy generation must equal a brute-force no-cache loop."""
    model = _model()
    idx = torch.randint(0, 64, (1, 5))
    cached = generate(model, idx, max_new_tokens=8, greedy=True)

    # reference: recompute full forward each step, no cache
    ref = idx.clone()
    for _ in range(8):
        logits, _, _ = model(ref)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ref = torch.cat([ref, nxt], dim=1)
    assert torch.equal(cached, ref)


def test_generate_stops_at_eos():
    model = _model()
    idx = torch.randint(0, 64, (1, 3))
    # force eos to be whatever greedy would pick first, to guarantee an early stop
    logits, _, _ = model(idx)
    eos = logits[:, -1, :].argmax(dim=-1).item()
    out = generate(model, idx, max_new_tokens=20, greedy=True, eos_id=eos)
    assert out[0, -1].item() == eos
    assert out.size(1) < 3 + 20  # stopped early


def test_generate_stream_yields_tokens():
    model = _model()
    idx = torch.randint(0, 64, (1, 3))
    toks = list(generate_stream(model, idx, max_new_tokens=5, greedy=True))
    assert len(toks) == 5
    assert all(t.shape == (1,) for t in toks)
