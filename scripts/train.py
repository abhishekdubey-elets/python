"""CLI: train a model from config files on packed token shards.

Example:
    python scripts/train.py --model-config configs/toy.yaml \
        --train-config configs/train_default.yaml --data-dir data/packed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pyllm.config import ModelConfig, TrainConfig  # noqa: E402
from pyllm.data import load_bin  # noqa: E402
from pyllm.model import PyLLM  # noqa: E402
from pyllm.training import Trainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train a pyllm model.")
    p.add_argument("--model-config", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--data-dir", required=True, help="Dir with train.bin / val.bin")
    args = p.parse_args()

    model_cfg = ModelConfig.from_yaml(args.model_config)
    train_cfg = TrainConfig.from_yaml(args.train_config)

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("vocab_size") and meta["vocab_size"] != model_cfg.vocab_size:
            raise SystemExit(
                f"vocab mismatch: data={meta['vocab_size']} model={model_cfg.vocab_size}"
            )

    train_data = load_bin(data_dir / "train.bin")
    val_path = data_dir / "val.bin"
    val_data = load_bin(val_path) if val_path.exists() else None

    model = PyLLM(model_cfg)
    print(f"model: {model.num_params()/1e6:.1f}M params on {train_cfg.device}")

    trainer = Trainer(model, model_cfg, train_cfg, train_data, val_data, save_checkpoints=True)
    trainer.train()
    print("done.")


if __name__ == "__main__":
    main()
