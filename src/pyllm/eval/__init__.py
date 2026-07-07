"""Evaluation subpackage: perplexity and functional (pass@k) code eval."""

from pyllm.eval.codegen import (
    build_program,
    check_correctness,
    evaluate_pass_at_k,
    pass_at_k,
)
from pyllm.eval.perplexity import evaluate_perplexity

__all__ = [
    "evaluate_perplexity",
    "pass_at_k",
    "build_program",
    "check_correctness",
    "evaluate_pass_at_k",
]
