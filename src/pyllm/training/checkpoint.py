"""Checkpoint save/load for exact, resumable training.

A checkpoint captures everything needed to reconstruct the run: model weights,
optimizer state (Adam moments!), step counter, and both configs. Saving the
config alongside the weights is what makes a checkpoint self-describing — you can
rebuild the exact model from the file without external context (Phase 2 payoff).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from pyllm.config import ModelConfig, TrainConfig


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig | None = None,
    **extra: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg) if train_cfg is not None else None,
        **extra,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint dict. (weights_only=False: our payload has config dicts.)"""
    return torch.load(path, map_location=map_location, weights_only=False)


def model_config_from_checkpoint(ckpt: dict) -> ModelConfig:
    """Rebuild the ModelConfig stored in a checkpoint (drops derived fields)."""
    data = dict(ckpt["model_cfg"])
    data.pop("head_dim", None)  # derived; recomputed in __post_init__
    return ModelConfig.from_dict(data)
