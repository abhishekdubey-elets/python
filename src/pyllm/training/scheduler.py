"""Learning-rate schedule: linear warmup then cosine decay to a floor.

Warmup: start tiny and ramp up over the first ``warmup_steps``. Early on, Adam's
running moments are unreliable and big steps can destabilize training; warmup
avoids that. Cosine decay: smoothly anneal to ``min_lr`` so late training takes
small, refining steps. This is the de-facto standard for LLM pretraining.
"""

from __future__ import annotations

import math


def cosine_warmup_lr(
    step: int, lr: float, min_lr: float, warmup_steps: int, max_steps: int
) -> float:
    """Return the learning rate for a given step (0-indexed)."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps  # linear ramp (never exactly 0)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return min_lr + coeff * (lr - min_lr)


class CosineWarmupScheduler:
    """Stateful convenience wrapper that sets an optimizer's LR each step."""

    def __init__(self, optimizer, lr, min_lr, warmup_steps, max_steps) -> None:
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_num = 0

    def get_lr(self, step: int) -> float:
        return cosine_warmup_lr(step, self.lr, self.min_lr, self.warmup_steps, self.max_steps)

    def step(self) -> float:
        lr = self.get_lr(self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step_num += 1
        return lr
