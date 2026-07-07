# 06 — Attention (MHA, GQA, KV cache, Flash/SDPA)

> Phase 4d. Deliverable: a tested `Attention` module supporting MHA and GQA, with
> a KV cache and both a fused (SDPA) and a manual reference implementation.

## 1. Intuition

Attention lets each token *look at* other tokens and pull in the information it
needs. For token *i*, it forms a **query**, compares it against every token's
**key** to get weights, and returns a weighted sum of their **values**. It's the
only place in the model where positions talk to each other.

## 2. Scaled dot-product attention

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

The `1/√dₕ` scaling keeps the dot products from growing with head dimension
(large scores → peaky softmax → vanishing gradients). **Multi-head**: do this in
`n_heads` parallel subspaces of size `head_dim` so different heads can specialize
(syntax, long-range dependencies, matching brackets, …), then concatenate.

## 3. Causal masking

A language model may only attend to the past. We add `-inf` to scores above the
diagonal before the softmax, so future positions get weight 0. Our manual mask
uses `tril(diagonal = Tk − Tq)` so it's correct even when query and key lengths
differ (as during cached decoding).

## 4. GQA — grouped-query attention

At inference the **KV cache** (see §5) dominates memory, and it scales with the
number of K/V heads. GQA keeps all `n_heads` query heads but uses fewer
`n_kv_heads`; each K/V head is shared by a group of query heads.

```
MHA:  n_kv_heads = n_heads        (biggest cache, most expressive)
GQA:  1 < n_kv_heads < n_heads    (our 300M config: 16 q-heads, 4 kv-heads -> 4x smaller cache)
MQA:  n_kv_heads = 1              (smallest cache, some quality loss)
```

`repeat_kv` expands the compact K/V up to `n_heads` right before attention. The
**cache stores the compact (pre-expansion) K/V**, which is where the memory
saving comes from. Trade-off: GQA trades a little quality for a large cut in
inference memory/bandwidth — usually well worth it.

## 5. The KV cache

During generation we produce one token at a time. Without a cache, step *t*
recomputes attention over all *t* tokens → O(T²) total. Since past keys/values
don't change, we **cache** them: each step computes K/V for the *new* token only,
appends to the cache, and attends. That's O(1) work per step (O(T) total).

Two correctness details, both tested:
* **Cache rotated K.** We apply RoPE *before* caching, so cached keys carry their
  absolute-position rotation and never need re-rotating.
* **RoPE offset.** The new token is rotated at position = current cache length
  (`rope(1, offset=t)`), not position 0.

Our test `test_kv_cache_matches_full_forward` feeds tokens one-by-one and asserts
the result equals a single full forward — the ground truth for Phase 7.

## 6. Flash / SDPA compatibility

`torch.nn.functional.scaled_dot_product_attention` dispatches to a fused kernel
(FlashAttention / memory-efficient attention) when available, avoiding
materializing the full `T×T` score matrix — a big memory/speed win at long
context. We default to it (`use_sdpa=True`) but keep the explicit `_manual` path
for learning, and `test_sdpa_matches_manual` asserts they agree to 1e-4. This is
principle P6 in action: the optimization is toggleable and validated against a
reference.

## 7. Complexity

Compute is O(B·n_heads·T²·head_dim); memory is O(T²) for the naive path, or
O(T) with a Flash kernel. Attention is the term that makes long context
expensive — hence Flash + GQA.

## 8. Common mistakes

* Rotating V with RoPE (only Q/K).
* Forgetting the `1/√dₕ` scale.
* Off-by-one in the causal mask when Tq ≠ Tk (cached decoding).
* Re-rotating cached keys, or rotating the new token at position 0 instead of the
  cache offset.
* Expanding GQA K/V *before* caching (defeats the memory saving).

## 9. References

* Vaswani et al., 2017 — *Attention Is All You Need*.
* Ainslie et al., 2023 — *GQA*; Shazeer, 2019 — *MQA*.
* Dao et al., 2022 — *FlashAttention*.
