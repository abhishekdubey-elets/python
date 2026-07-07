"""Tests for evaluation: perplexity + pass@k harness (Phase 8)."""

from __future__ import annotations

import math

import numpy as np
import torch

from pyllm.config import ModelConfig
from pyllm.eval import (
    check_correctness,
    evaluate_pass_at_k,
    evaluate_perplexity,
    pass_at_k,
)
from pyllm.model import PyLLM


# --- pass@k estimator ------------------------------------------------------- #
def test_pass_at_k_edge_cases():
    assert pass_at_k(n=1, c=1, k=1) == 1.0     # the only sample passes
    assert pass_at_k(n=5, c=0, k=1) == 0.0     # none pass
    assert pass_at_k(n=10, c=10, k=3) == 1.0   # all pass
    # n=2, c=1, k=1 => probability a single random draw passes = 1/2
    assert abs(pass_at_k(n=2, c=1, k=1) - 0.5) < 1e-9


# --- code execution harness ------------------------------------------------- #
def test_check_correctness_pass_and_fail():
    good = "def add(a, b):\n    return a + b\nassert add(2, 3) == 5\n"
    bad = "def add(a, b):\n    return a - b\nassert add(2, 3) == 5\n"
    assert check_correctness(good)["passed"] is True
    assert check_correctness(bad)["passed"] is False


def test_evaluate_pass_at_k_end_to_end():
    problems = [{"test": "def check(f):\n    assert f(2) == 4\n", "entry_point": "double"}]
    samples = [[
        "def double(x):\n    return x * 2",   # correct
        "def double(x):\n    return x + 1",   # wrong
    ]]
    res = evaluate_pass_at_k(problems, samples, k=1)
    # 1 of 2 correct => pass@1 == 0.5
    assert abs(res["pass@1"] - 0.5) < 1e-9


# --- perplexity ------------------------------------------------------------- #
def test_perplexity_is_exp_of_loss():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, seq_len=16)
    model = PyLLM(cfg).eval()
    data = np.arange(500, dtype=np.uint16) % cfg.vocab_size
    res = evaluate_perplexity(model, data, seq_len=16, batch_size=4, max_batches=5)
    assert math.isfinite(res["perplexity"])
    assert abs(res["perplexity"] - math.exp(res["loss"])) < 1e-4
