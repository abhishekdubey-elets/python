# 04 — RoPE (Rotary Positional Embeddings)

> Phase 4b. Deliverable: a tested `RotaryEmbedding` + `apply_rotary_emb`.

## 1. Intuition

Self-attention sees a *set*, not a *sequence* — it's permutation-invariant. So
"dog bites man" and "man bites dog" look identical unless we inject position.

RoPE injects it by **rotating** each query/key vector by an angle proportional to
its position, rather than *adding* a position vector. The elegant consequence:
the attention score between positions m and n ends up depending only on the
**relative** offset (m − n).

## 2. Evolution of position encoding

| Method | Kind | Extrapolates? | Relative? |
|---|---|---|---|
| Learned absolute (GPT-2) | additive | no (unseen positions) | no |
| Sinusoidal (Transformer) | additive | somewhat | no |
| **RoPE** | **rotary (multiplicative)** | yes (tune base) | **yes** |

## 3. The math

Split `head_dim` into pairs; pair *i* is a 2D subspace rotated by `pos · θᵢ`,
with frequencies

$$\theta_i = \text{base}^{-2i/d}, \quad i = 0 \dots d/2 - 1.$$

Fast-spinning pairs (large θ) capture local structure; slow ones capture
long-range. In 2D, rotating vector `(a, b)` by angle `φ`:

$$(a\cos\varphi - b\sin\varphi,\; a\sin\varphi + b\cos\varphi).$$

**Why the score is relative.** A rotation by angle `mφ` is a matrix `R(mφ)`.
Because rotations compose as `R(a)ᵀR(b) = R(b − a)`, the query·key score is

$$(R(m\varphi)q)^\top (R(n\varphi)k) = q^\top R((n-m)\varphi)\,k,$$

which depends on `n − m` only. Absolute position cancels — algebraically, for
free. And since rotations are orthogonal, `‖RoPE(x)‖ = ‖x‖`.

## 4. The "rotate_half" implementation

We use Llama/HF's convention. Build `cos`/`sin` of shape `(T, head_dim)` by
concatenating the per-pair frequencies with themselves, then:

```
RoPE(x) = x * cos + rotate_half(x) * sin
rotate_half([x1, x2]) = [-x2, x1]     # x1, x2 are the two halves of the last dim
```

This applies the 2D rotation to every pair simultaneously via elementwise ops.
Applied to **Q and K only** (not V), per head, on `head_dim`.

## 5. The `offset` argument (why it exists)

`forward(seq_len, offset)` returns tables for positions `[offset, offset+seq_len)`.
During KV-cache decoding (Phase 7) we feed one new token at a time; it must be
rotated at its *true* absolute position, not position 0. `offset` provides that.

## 6. Complexity

Tables are precomputed once: O(max_seq_len · head_dim) memory. Applying RoPE is
O(B·H·T·head_dim) elementwise — cheap.

## 7. Defining properties (our tests)

* **Relativity:** `<RoPE(q,m), RoPE(k,n)>` depends only on `m − n`
  (checked: score diagonals are constant).
* **Norm preservation:** `‖RoPE(x)‖ = ‖x‖`.
* **Offset = slice:** `forward(n, offset=k)` equals the `[k:k+n]` slice of the
  full table.

## 8. Common mistakes

* Rotating **V** as well as Q/K (only Q/K should be rotated).
* Mismatched `cos`/`sin` convention vs `rotate_half` (interleaved vs
  half-split) — they must agree or positions get scrambled.
* Odd `head_dim` (pairs require it to be even) — we validate this.
* Requesting positions beyond `max_seq_len` — we raise instead of indexing OOB.

## 9. References

* Su et al., 2021 — *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
* Llama / Llama 2 — RoPE at scale.
* Chen et al., 2023 — position interpolation (extending context via `base`).
