# 09 — Training

> Phase 6. Deliverable: optimizer, schedule, checkpointing, and a Trainer that
> overfits a tiny batch (the canonical correctness test).

## 1. The objective

Next-token prediction with cross-entropy: maximize the log-probability the model
assigns to the actual next token, averaged over positions. Every position in a
sequence is a training example simultaneously (teacher forcing), which is why LM
pretraining is so sample-efficient per forward pass.

## 2. Optimizer: AdamW with param groups (`optimizer.py`)

AdamW = Adam with **decoupled** weight decay. The crucial detail: apply weight
decay to matmul weights (dim ≥ 2), but **not** to 1-D params (RMSNorm gains,
biases). Decaying scales/offsets toward zero hurts. `betas=(0.9, 0.95)` is the
LLM-standard (a slightly lower β₂ than the vision default). Fused kernel is used
on CUDA when available.

## 3. Schedule: warmup + cosine decay (`scheduler.py`)

Linear warmup for the first `warmup_steps` (Adam's moment estimates are noisy
early; big steps destabilize), then cosine decay to `min_lr`. Standard for LLMs.

## 4. The loop (`trainer.py`)

Per optimizer step:
1. set the LR from the schedule,
2. **accumulate** grads over `grad_accum_steps` micro-batches — simulates a large
   batch that wouldn't fit in memory (effective batch = micro × accum × seq_len),
3. **clip** the global grad norm (guards against loss spikes),
4. `optimizer.step()`,
5. periodically estimate val loss, log, and checkpoint.

**Mixed precision:** bf16 autocast (no loss scaler needed — bf16 keeps fp32's
exponent range); fp16 additionally needs a `GradScaler`. **Gradient
checkpointing** (`grad_checkpointing`) trades compute for memory. Both are config
flags.

## 5. Checkpointing & resume (`checkpoint.py`)

A checkpoint stores model + optimizer (Adam moments!) + step + **config**. Saving
the config makes the checkpoint self-describing — `model_config_from_checkpoint`
rebuilds the exact architecture with no external context. Tested by a full
save→load→weights-equal round-trip.

## 6. The overfit test

The single most important training test: on a tiny fixed token stream, the loss
must fall to near zero. Ours drops below 1.0 in 300 steps. If a training loop
*can't* overfit, the loss/backward/optimizer wiring is broken — this catches it
before you waste a real run.

## 7. Common mistakes

* Weight-decaying norms/biases. * Forgetting to zero grads. * Clipping *before*
unscaling fp16 grads. * LR too high with no warmup → early divergence.
* Counting a micro-batch as a full step in the schedule.

## 8. References

* Loshchilov & Hutter, 2019 — AdamW. * Kingma & Ba, 2015 — Adam.
* nanoGPT — the training-loop template.
