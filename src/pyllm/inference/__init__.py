"""Inference subpackage: samplers and KV-cached generation."""

from pyllm.inference.generate import generate, generate_stream
from pyllm.inference.sampler import (
    apply_temperature,
    sample_next,
    top_k_filter,
    top_p_filter,
)

__all__ = [
    "generate",
    "generate_stream",
    "sample_next",
    "apply_temperature",
    "top_k_filter",
    "top_p_filter",
]
