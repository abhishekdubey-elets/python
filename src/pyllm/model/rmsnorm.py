"""RMSNorm — root-mean-square layer normalization (Zhang & Sennrich, 2019).

Modern LLMs (Llama, Qwen, Gemma, Mistral) use RMSNorm instead of LayerNorm. It
keeps only the *re-scaling* half of LayerNorm — no mean subtraction, no bias —
which is cheaper and empirically just as stable:

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

The reduction is done in fp32 for numerical stability, then cast back to the
input dtype (important once we train in bf16).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension.

    Args:
        dim: size of the feature dimension being normalized (``d_model`` or,
            in some designs, ``head_dim``).
        eps: small constant added inside the sqrt for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # One learnable gain per feature, initialized to 1.0 => starts as an
        # identity re-scaling. There is deliberately no bias (that's the point).
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        # rsqrt(mean(x^2) + eps): divide each vector by its RMS over the last dim.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """(..., dim) -> (..., dim), same shape and dtype as the input."""
        input_dtype = x.dtype
        # Do the reduction AND the weight multiply in fp32 for precision, then
        # cast once at the very end so the output dtype always matches the input
        # (even when `weight` is fp32 but `x` is bf16).
        normed = self._norm(x.float())
        return (normed * self.weight).to(input_dtype)
