# 05 — SwiGLU MLP

> Phase 4c. Deliverable: a tested `SwiGLU` feed-forward module.

## 1. Intuition

Each Transformer block alternates *mixing across positions* (attention) with
*transforming each position independently* (the MLP). The MLP is where most of a
model's parameters and "knowledge" live. SwiGLU is a **gated** MLP that gives
better quality per parameter than the classic version.

## 2. Classic MLP vs SwiGLU

Classic (GPT-2): two matrices, a pointwise activation, hidden = 4·d:

    y = down( GELU( up(x) ) )

SwiGLU: **three** matrices; one branch gates the other:

    y = down( SiLU(gate(x)) ⊙ up(x) )          SiLU(z) = z·σ(z),  ⊙ elementwise

The `SiLU(gate(x))` branch produces a smooth, data-dependent 0..~1 mask that
turns hidden units up or down per token. That multiplicative interaction is
strictly more expressive than a single pointwise nonlinearity.

## 3. The 2/3 rule (full derivation)

We want SwiGLU to cost the same as the 2-matrix MLP it replaces.

* Classic params: `up` is d×4d, `down` is 4d×d → `2 · d · 4d = 8d²`.
* SwiGLU params (hidden h): `gate`, `up` are d×h, `down` is h×d → `3 · d · h`.

Set equal: `3dh = 8d²  ⇒  h = 8d/3 = (2/3)·4d`. Then round up to a hardware
multiple (256). That's exactly `ModelConfig._compute_ffn_hidden` — for d=768 it
gives 2048; for d=1024 it rounds 2730 up to 2816.

## 4. Complexity

O(B·T·d·h). With h ≈ 8d/3 this is the largest single matmul cost in a block,
which is why the 2/3 trick (keeping it at parity, not larger) matters.

## 5. Properties tested

* shape preserved (d_model → d_model)
* parameter count is exactly `3 · d · h` (three matrices, no bias by default)
* the module is genuinely nonlinear (`f(2x) ≠ 2f(x)`)

## 6. Common mistakes

* Using `h = 4d` with three matrices → 50% more MLP params than intended.
* Applying the activation to the wrong branch (SiLU goes on `gate`, not `up`).
* Adding biases — modern LLMs omit them (we gate on `cfg.bias`, default off).

## 7. References

* Shazeer, 2020 — *GLU Variants Improve Transformer*.
* Llama / PaLM — SwiGLU with the 2/3 sizing at scale.
