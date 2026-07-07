"""Tests for the full PyLLM model (Phase 4).

Highlights:
* the real parameter count matches ModelConfig.estimate_params() exactly
  (this validates the Phase 2 arithmetic against the actual nn.Module)
* weight tying shares one matrix
* forward produces correct shapes and a finite loss ~ ln(vocab) at init
* the model is causal end-to-end
* KV-cache generation matches a full forward
"""

from __future__ import annotations

import math

import pytest
import torch

from pyllm.config import ModelConfig
from pyllm.model import PyLLM


def _toy(**kw) -> ModelConfig:
    base = dict(vocab_size=512, d_model=128, n_layers=2, n_heads=4, seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize(
    "cfg",
    [
        _toy(),                                   # MHA
        _toy(n_heads=8, n_kv_heads=2),            # GQA
        _toy(tie_weights=False),                  # untied head
        _toy(bias=True),                          # with biases
    ],
)
def test_param_count_matches_estimate(cfg):
    model = PyLLM(cfg)
    assert model.num_params() == cfg.estimate_params()


def test_weight_tying_shares_matrix():
    tied = PyLLM(_toy(tie_weights=True))
    assert tied.lm_head.weight is tied.tok_emb.weight

    untied = PyLLM(_toy(tie_weights=False))
    assert untied.lm_head.weight is not untied.tok_emb.weight
    # untied model has exactly one extra vocab*d_model matrix
    assert untied.num_params() - tied.num_params() == 512 * 128


def test_forward_shapes_and_init_loss():
    torch.manual_seed(0)
    cfg = _toy()
    model = PyLLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, cache = model(idx, targets)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert cache is None
    # A well-initialized LM predicts ~uniform, so loss ~ ln(vocab_size).
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.7


def test_loss_is_none_without_targets():
    model = PyLLM(_toy()).eval()
    idx = torch.randint(0, 512, (1, 8))
    logits, loss, _ = model(idx)
    assert loss is None


def test_model_is_causal():
    torch.manual_seed(0)
    cfg = _toy()
    model = PyLLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 10))
    logits_a, _, _ = model(idx)

    idx2 = idx.clone()
    idx2[0, 9] = (idx[0, 9] + 1) % cfg.vocab_size  # change only the last token
    logits_b, _, _ = model(idx2)

    assert torch.allclose(logits_a[:, :9], logits_b[:, :9], atol=1e-4)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = _toy()
    model = PyLLM(cfg).eval()
    T = 12
    idx = torch.randint(0, cfg.vocab_size, (1, T))

    full_logits, _, _ = model(idx)

    past = None
    step_logits = []
    for t in range(T):
        logits_t, _, past = model(idx[:, t : t + 1], past_kvs=past, use_cache=True)
        step_logits.append(logits_t)
    incr_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(full_logits, incr_logits, atol=1e-4)


def test_exceeding_seq_len_raises():
    model = PyLLM(_toy(seq_len=8))
    with pytest.raises(ValueError, match="exceeds seq_len"):
        model(torch.randint(0, 512, (1, 9)))


def test_grad_checkpointing_forward_backward():
    torch.manual_seed(0)
    model = PyLLM(_toy()).train()
    model.grad_checkpointing = True
    idx = torch.randint(0, 512, (2, 16))
    targets = torch.randint(0, 512, (2, 16))
    _, loss, _ = model(idx, targets)
    loss.backward()  # must not error; grads should exist
    assert model.tok_emb.weight.grad is not None
