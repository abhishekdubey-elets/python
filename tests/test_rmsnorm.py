"""Tests for RMSNorm (Phase 4).

We test the *properties* that define RMSNorm, not hard-coded numbers:
* it preserves shape and dtype
* with weight=1 it makes each vector's RMS == 1 (that's the whole job)
* it is scale-invariant: RMSNorm(c*x) == RMSNorm(x)  (a defining property)
* the learnable weight scales the output as expected
"""

from __future__ import annotations

import torch

from pyllm.model import RMSNorm


def test_shape_and_dtype_preserved():
    norm = RMSNorm(dim=32)
    x = torch.randn(4, 10, 32)
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_output_has_unit_rms_when_weight_is_one():
    torch.manual_seed(0)
    norm = RMSNorm(dim=64, eps=1e-8)
    x = torch.randn(8, 64) * 5.0  # arbitrary scale
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()  # RMS of each row
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_scale_invariance():
    torch.manual_seed(0)
    norm = RMSNorm(dim=64, eps=1e-8)
    x = torch.randn(8, 64)
    assert torch.allclose(norm(x), norm(3.7 * x), atol=1e-4)


def test_weight_scales_output():
    norm = RMSNorm(dim=16, eps=1e-8)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(5, 16)
    base = RMSNorm(dim=16, eps=1e-8)(x)  # weight=1 reference
    assert torch.allclose(norm(x), 2.0 * base, atol=1e-5)


def test_bf16_input_upcasts_internally_and_returns_bf16():
    norm = RMSNorm(dim=32)
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()
