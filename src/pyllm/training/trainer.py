"""The training loop: gradient accumulation, clipping, mixed precision, LR
schedule, periodic evaluation, logging, and checkpointing.

Kept deliberately explicit (nanoGPT-style) so every step is visible:

    for each optimizer step:
        set LR from the cosine-warmup schedule
        accumulate grads over `grad_accum_steps` micro-batches   (larger effective batch)
        clip the global grad norm
        optimizer.step()
        periodically: estimate val loss, log, checkpoint
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from pyllm.config import ModelConfig, TrainConfig
from pyllm.data.loader import get_batch
from pyllm.training.checkpoint import save_checkpoint
from pyllm.training.optimizer import configure_optimizer
from pyllm.training.scheduler import cosine_warmup_lr

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        save_checkpoints: bool = False,
    ) -> None:
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.device = train_cfg.device
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.save_checkpoints = save_checkpoints

        self.model.grad_checkpointing = train_cfg.grad_checkpointing
        if train_cfg.compile:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:  # compile is best-effort (often unavailable on CPU)
                print(f"[trainer] torch.compile disabled: {e}")

        self.optimizer = configure_optimizer(
            self.model,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(train_cfg.beta1, train_cfg.beta2),
            device=self.device,
        )
        self.step = 0
        self.history: dict[str, list] = {"step": [], "train_loss": [], "val_loss": []}

        # Mixed precision. bf16 needs no loss scaling; fp16 does.
        self.amp_dtype = _DTYPES[train_cfg.dtype]
        self.use_amp = train_cfg.dtype in ("bfloat16", "float16")
        use_scaler = train_cfg.dtype == "float16"
        self.scaler = torch.amp.GradScaler(self.device, enabled=use_scaler)

        # Seed for reproducible batch sampling.
        self.generator = torch.Generator().manual_seed(train_cfg.seed)

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type=self.device.split(":")[0], dtype=self.amp_dtype)
        return nullcontext()

    def _batch(self, data: np.ndarray):
        return get_batch(
            data,
            self.cfg.micro_batch_size,
            self.model_cfg.seq_len,
            device=self.device,
            generator=self.generator,
        )

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Average loss over ``eval_iters`` random batches per available split."""
        self.model.eval()
        out: dict[str, float] = {}
        splits = {"train": self.train_data}
        if self.val_data is not None and len(self.val_data) > self.model_cfg.seq_len + 1:
            splits["val"] = self.val_data
        for name, data in splits.items():
            losses = torch.zeros(self.cfg.eval_iters)
            for i in range(self.cfg.eval_iters):
                x, y = self._batch(data)
                with self._autocast():
                    _, loss, _ = self.model(x, y)
                losses[i] = loss.item()
            out[name] = losses.mean().item()
        self.model.train()
        return out

    def train(self) -> dict[str, list]:
        self.model.train()
        while self.step < self.cfg.max_steps:
            lr = cosine_warmup_lr(
                self.step, self.cfg.lr, self.cfg.min_lr, self.cfg.warmup_steps, self.cfg.max_steps
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(self.cfg.grad_accum_steps):
                x, y = self._batch(self.train_data)
                with self._autocast():
                    _, loss, _ = self.model(x, y)
                    loss = loss / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1

            if self.step % self.cfg.log_interval == 0:
                print(f"step {self.step:>6} | loss {step_loss:.4f} | lr {lr:.2e}")
            if self.step % self.cfg.eval_interval == 0 or self.step == self.cfg.max_steps:
                losses = self.estimate_loss()
                self.history["step"].append(self.step)
                self.history["train_loss"].append(losses.get("train"))
                self.history["val_loss"].append(losses.get("val"))
                print(f"  eval @ {self.step}: {losses}")
                if self.save_checkpoints:
                    self._save("ckpt.pt")
        return self.history

    def _save(self, name: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / name
        # unwrap torch.compile if needed
        model = getattr(self.model, "_orig_mod", self.model)
        save_checkpoint(path, model, self.optimizer, self.step, self.model_cfg, self.cfg)
