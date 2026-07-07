"""Functional code evaluation: pass@k over HumanEval/MBPP-style problems.

Unlike perplexity, this measures whether generated code *actually runs and passes
tests* — the thing we ultimately care about for a code model.

pass@k (Chen et al., 2021, Codex): generate n samples per problem, count c that
pass; the unbiased estimate of "at least one of k random samples passes" is

    pass@k = 1 - C(n-c, k) / C(n, k)   (computed stably below).

Limitations to keep honest about (see docs/11_evaluation.md):
  * benchmarks like HumanEval are small and can be *contaminated* (present in
    training data) — a high score may overstate ability.
  * tests are partial; passing them != correct.
  * executing model output is a SECURITY RISK. This runs each program in a
    subprocess with a timeout only — NOT a real sandbox. For untrusted models,
    run inside a container/VM with no network and dropped privileges.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k given n samples, c correct."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n - c < k:
        return 1.0
    # 1 - prod_{i=n-c+1..n} (1 - k/i)
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def build_program(solution_code: str, test_code: str, entry_point: str) -> str:
    """Assemble a runnable program: solution + test harness + the check call."""
    return f"{solution_code}\n\n{test_code}\n\ncheck({entry_point})\n"


def check_correctness(program: str, timeout: float = 5.0) -> dict:
    """Run a program in a subprocess; passed == exit code 0 within the timeout.

    WARNING: minimal isolation (separate process + timeout). Do not run untrusted
    code with this outside a proper sandbox.
    """
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(program)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {"passed": proc.returncode == 0, "stderr": proc.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"passed": False, "stderr": "timeout"}
    except Exception as e:  # pragma: no cover - defensive
        return {"passed": False, "stderr": f"{type(e).__name__}: {e}"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def evaluate_pass_at_k(
    problems: list[dict],
    samples: list[list[str]],
    k: int,
    timeout: float = 5.0,
) -> dict:
    """Compute mean pass@k over problems.

    Args:
        problems: each dict has ``test`` and ``entry_point`` (and produced the
            corresponding solutions).
        samples: ``samples[i]`` is the list of candidate solution strings (full
            function definitions) for ``problems[i]``.
    """
    if len(problems) != len(samples):
        raise ValueError("problems and samples must align 1:1")
    scores = []
    for problem, sols in zip(problems, samples):
        n = len(sols)
        c = 0
        for sol in sols:
            program = build_program(sol, problem["test"], problem["entry_point"])
            if check_correctness(program, timeout=timeout)["passed"]:
                c += 1
        scores.append(pass_at_k(n, c, k))
    return {f"pass@{k}": float(np.mean(scores)) if scores else 0.0, "n_problems": len(problems)}
