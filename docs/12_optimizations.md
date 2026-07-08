# 12 — Optimizations & polish

> Phase 9. Deliverable: performance toggles (each validated against a reference)
> and an end-to-end integration test.

## 1. Principle: every optimization is a toggle, validated

Per design principle P6, we never trade correctness for speed silently. Each
optimization is a config flag, and a test proves the fast path matches the
readable reference.

| Optimization | Where | Flag | Validated by |
|---|---|---|---|
| Flash / fused attention | `attention.py` | `use_sdpa` | `test_sdpa_matches_manual` |
| Grouped-query attention | `attention.py` | `n_kv_heads < n_heads` | `test_gqa_shapes_and_agreement` |
| KV cache | `model.py` / `generate.py` | `use_cache` | `test_kv_cache_matches_full_forward`, `test_generate_matches_no_cache_reference` |
| bf16 mixed precision | `trainer.py` | `dtype` | dtype round-trip in RMSNorm |
| Gradient checkpointing | `model.py` | `grad_checkpointing` | `test_grad_checkpointing_forward_backward` |
| Gradient accumulation | `trainer.py` | `grad_accum_steps` | overfit test |
| Fused AdamW | `optimizer.py` | auto on CUDA | param-group test |
| torch.compile | `trainer.py` | `compile` | best-effort, wrapped in try |

## 2. What each buys you

* **Flash/SDPA:** avoids materializing the T×T score matrix → big memory/speed
  win at long context.
* **GQA:** shrinks the KV cache (the inference memory bottleneck) — 4× in our
  300M config.
* **KV cache:** O(T) generation instead of O(T²).
* **bf16:** ~2× throughput and half the activation memory, no loss scaler.
* **Gradient checkpointing:** fit bigger models/batches by recomputing
  activations in backward.
* **torch.compile:** kernel fusion; large speedups on GPU (often unavailable/slow
  on CPU, hence best-effort).

## 3. The integration test

`test_integration.py` runs the whole stack in miniature: tokenizer → clean →
pack → train (loss drops) → generate → perplexity, on a toy model in seconds. If
it passes, the subsystems compose correctly.

## 4. Scaling up (when you have a GPU)

1. Expand the corpus (The Stack / curated GitHub Python) via `prepare_data.py`.
2. Train the 50257-vocab tokenizer on the full corpus.
3. Use `configs/model_125m.yaml` (or `300m`), set `device=cuda`,
   `dtype=bfloat16`, `compile=true`, and raise `micro_batch_size` /
   `grad_accum_steps` to a large effective batch.
4. Watch the loss curve; sample completions periodically; evaluate pass@k.

## 5. Where to go next

Instruction/FIM fine-tuning (the reserved FIM tokens are ready), longer context
via RoPE `base` scaling, GQA/MQA ablations, quantized inference, DDP multi-GPU.
The clean module boundaries mean each is an additive change, not a rewrite.
