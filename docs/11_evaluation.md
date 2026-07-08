# 11 — Evaluation

> Phase 8. Deliverable: perplexity + a pass@k functional-correctness harness.

## 1. Perplexity (`perplexity.py`)

`PPL = exp(mean per-token cross-entropy)` — the effective number of equally-likely
choices the model faces per step. Cheap, correlates with quality, great for loss
curves and comparing checkpoints. **But** it does not measure whether generated
code *runs*. Low perplexity ≠ correct programs.

## 2. pass@k (`codegen.py`)

The metric that matters for code: generate n samples per problem, count c that
pass the unit tests, and report the unbiased estimator

    pass@k = 1 − C(n−c, k) / C(n, k)

(computed stably as `1 − Π (1 − k/i)`). pass@1 ≈ "does it work first try";
pass@10/100 ≈ "does it work if you sample a few". Tested against known values
(n=c=k=1 → 1; c=0 → 0; n=2,c=1,k=1 → 0.5).

## 3. Running generated code — honestly

`check_correctness` runs each candidate in a **subprocess with a timeout**. This
is NOT a real sandbox — it's minimal isolation. **Executing model-generated code
is a security risk.** For untrusted models, run inside a container/VM with no
network and dropped privileges. We flag this loudly in the code.

## 4. Benchmarks and their limits

HumanEval / MBPP are the standard code benchmarks, but:
* **Small** (164 / ~1000 problems) → noisy, high variance.
* **Contamination** — if benchmark solutions are in your training data, scores
  are inflated. Keep eval sets out of training (Phase 5 dedup).
* **Partial tests** — passing the given tests ≠ correct in general.
* Prefer them as *relative* signals across your own checkpoints, plus **human
  spot-checks** and real code-completion examples.

## 5. References

* Chen et al., 2021 — *Evaluating LLMs Trained on Code* (Codex, pass@k, HumanEval).
* Austin et al., 2021 — MBPP.
