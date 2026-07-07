# 07 — The Block and the full Transformer

> Phase 4e. Deliverable: a tested `Block` and `PyLLM` model whose real parameter
> count matches `ModelConfig.estimate_params()` exactly.

## 1. The block: pre-norm residuals

Each block is two sublayers, each wrapped in a normalize → sublayer → add-residual
pattern:

```
x = x + attn( RMSNorm(x) )
x = x + mlp ( RMSNorm(x) )
```

**Pre-norm vs post-norm.** The original Transformer normalized *after* the
residual add (post-norm) and needed learning-rate warmup and care to train deep.
Pre-norm (normalize the sublayer's *input*, add the raw output back) leaves an
uninterrupted identity path down the residual stream, so gradients reach the
embeddings cleanly. Every modern LLM uses pre-norm.

Think of the residual stream as a shared "communication bus": each sublayer reads
a normalized view of it and writes an increment back. Nothing overwrites the bus.

## 2. The model, end to end

```
tokens ─▶ Embedding ─▶ [Block × n_layers] ─▶ RMSNorm ─▶ LM head ─▶ logits
                            ▲
                   RoPE cos/sin (shared)
```

* **Token embedding** maps ids → vectors. There is **no position embedding table**
  — position is injected by RoPE inside attention.
* **Final RMSNorm** before the head (standard; stabilizes the output distribution).
* **LM head** projects d_model → vocab_size. With **weight tying** it *is* the
  embedding matrix, transposed — saving ~vocab·d_model params (~38.6M at 125M).

## 3. Loss: next-token prediction

We shift labels by one and average token-level cross-entropy:

$$\mathcal{L} = -\frac{1}{N}\sum_t \log p_\theta(x_{t+1}\mid x_{\le t})$$

`ignore_index=-100` lets us mask padding/prompt tokens later. **Sanity check at
init:** an untrained model predicts ~uniformly, so the loss should be ≈
`ln(vocab_size)`. For the 125M model we measured **10.97 vs ln(50257)=10.83** —
exactly the expected ballpark, confirming the head/embedding/softmax are wired up
right.

## 4. Initialization

* Linear/Embedding: `N(0, 0.02)`.
* **Scaled residual init (GPT-2 §2.3):** the sublayer *output* projections
  (`o_proj`, `down_proj`) get std `0.02/√(2·n_layers)`. Each block adds to the
  residual stream; without down-scaling these outputs, the stream's variance
  grows with depth. This keeps it stable at initialization.

## 5. Parameter count matches the prediction

The whole point of config-as-data: we can predict params from config, and the
real module agrees. `test_param_count_matches_estimate` checks
`model.num_params() == cfg.estimate_params()` across MHA, GQA, untied, and
biased variants. At scale, the 125M config gives **123,551,232**, estimated and
actual identical (of which 84.95M is non-embedding). `parameters()` counts the
tied matrix once, so the accounting lines up.

## 6. KV cache in the model

`forward(idx, past_kvs=..., use_cache=True)` threads a per-layer cache through the
stack and returns the updated caches, with the RoPE offset derived from the cache
length. `test_kv_cache_matches_full_forward` asserts step-by-step cached decoding
equals a single full forward — the foundation for Phase 7 generation.

## 7. Gradient checkpointing (hook)

`model.grad_checkpointing = True` wraps each block in `torch.utils.checkpoint`,
trading recompute for memory during training (activations are recomputed in the
backward pass instead of stored). Off during caching/inference. The trainer
(Phase 6) flips this based on `TrainConfig`.

## 8. Common mistakes

* Adding a position-embedding table *and* RoPE (double-counting position).
* Forgetting the final norm before the head.
* Breaking weight tying by re-creating `lm_head.weight` after tying.
* Not down-scaling residual output projections → activation blow-up in deep stacks.

## 9. References

* Radford et al., 2019 (GPT-2) — pre-norm, scaled init, weight tying.
* Touvron et al., 2023 (Llama) — the RMSNorm + RoPE + SwiGLU decoder recipe.
* Press & Wolf, 2017 — weight tying.
