# 03 — RMSNorm

> Phase 4a. Deliverable: a tested `RMSNorm` module.

## 1. Intuition

Stacking many layers, activations tend to drift in scale — growing or shrinking
layer over layer — which destabilizes training. Normalization rescales each
token's feature vector back to a controlled magnitude so gradients stay healthy
and you can use a higher learning rate.

## 2. LayerNorm vs RMSNorm

**LayerNorm** does two things per vector:

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta \quad(\text{re-center, then re-scale, + learnable } \gamma,\beta)$$

**RMSNorm** keeps only the re-scaling:

$$y = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}\,\gamma = \frac{x}{\text{RMS}(x)}\,\gamma$$

The finding (Zhang & Sennrich, 2019): the mean-subtraction contributes little to
why normalization helps — the re-scaling is what matters. Dropping the mean and
the bias makes RMSNorm cheaper and, in practice, just as stable. It's the norm in
Llama, Qwen, Gemma, Mistral.

| | LayerNorm | RMSNorm |
|---|---|---|
| subtract mean | yes | **no** |
| divide by | std | RMS |
| learnable | γ (scale) + β (shift) | γ (scale) only |
| cost | higher | lower |

## 3. Worked example

`x = [3, -1, 1, -3]`, d = 4. `mean(x²) = (9+1+1+9)/4 = 5`, `RMS = √5 ≈ 2.236`.
`RMSNorm(x) ≈ [1.342, -0.447, 0.447, -1.342]` (with γ = 1). Its RMS is now 1.

## 4. Two implementation details that matter

* **Pre-norm placement.** We normalize *before* each sublayer and add the
  residual *after*: `x = x + attn(norm(x))`. This gives a clean gradient highway
  down the residual stream. (Wired up in the Block step.)
* **fp32 for the reduction AND the weight multiply.** `mean(x²)` loses precision
  in bf16. We compute the norm and apply `weight` in fp32, then cast **once at
  the end** to the input dtype. A subtle bug we hit and fixed: casting to bf16
  *before* multiplying by an fp32 `weight` silently promotes the result back to
  fp32, breaking the "output dtype == input dtype" contract. Cast last.

## 5. Complexity

O(B·T·d): one reduction and one elementwise scale. Negligible vs attention/MLP.

## 6. Defining properties (these are our tests)

* **Unit RMS:** with γ=1, every output vector has RMS 1.
* **Scale invariance:** `RMSNorm(c·x) == RMSNorm(x)` — magnitude is normalized away.
* **Shape/dtype preserved.**

## 7. Common mistakes

* Normalizing over the wrong axis (must be the last/feature dim, per token).
* Forgetting `eps` → divide-by-zero on all-zero vectors.
* Casting to low precision before the weight multiply (the bug above).
* Adding a bias — RMSNorm deliberately has none.

## 8. References

* Zhang & Sennrich, 2019 — *Root Mean Square Layer Normalization*.
* Ba et al., 2016 — *Layer Normalization*.
* Xiong et al., 2020 — *On Layer Normalization in the Transformer* (pre- vs post-norm).
