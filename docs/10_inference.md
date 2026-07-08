# 10 — Inference

> Phase 7. Deliverable: samplers + KV-cached generation (matches a no-cache
> reference exactly).

## 1. Decoding strategies (`sampler.py`)

All operate on last-position logits and compose as: **temperature → truncate
(top-k / top-p) → sample**.

| Strategy | What | When |
|---|---|---|
| greedy (temp 0) | argmax | deterministic, structured output |
| temperature | scales logits: <1 sharpen, >1 flatten | tune diversity |
| top-k | keep k most likely | simple, fixed budget |
| top-p (nucleus) | keep smallest set with cumulative prob ≥ p | best default; adapts to peakedness |

Temperature reshapes the distribution; the filters cut the unreliable tail so a
rare-but-unlucky token can't derail generation.

## 2. KV-cached generation (`generate.py`)

Naive generation recomputes attention over the whole prefix each step → O(T²).
Since past keys/values don't change, we **cache** them: run the prompt once
(prefill), then feed one new token per step and append to the cache → O(T) total.

Two correctness details (both tested): the cache holds **RoPE-rotated** keys, and
each new token is rotated at its **true absolute position** (`rope` offset =
cache length). `test_generate_matches_no_cache_reference` asserts cached greedy
decoding is bit-identical to a brute-force no-cache loop.

**Batched generation** tracks a per-row `finished` mask so sequences that emit
EOS stop cleanly while others continue. **Streaming** (`generate_stream`) yields
each token as produced, for live output.

## 3. Complexity

Per token: one forward over a single position attending to the cache — O(context)
compute, O(context) cache memory (this is what GQA shrinks).

## 4. Common mistakes

* Re-rotating cached keys, or rotating new tokens at position 0.
* Forgetting the seq_len bound during long generation.
* Applying top-p before temperature (order changes the nucleus).
* Not masking finished rows in batched decode.

## 5. References

* Holtzman et al., 2020 — *The Curious Case of Neural Text Degeneration* (top-p).
* Fan et al., 2018 — top-k sampling.
