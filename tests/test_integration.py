"""End-to-end integration test (Phase 9): the whole pipeline in miniature.

tokenizer -> clean -> pack -> train (loss drops) -> generate -> perplexity.

If this passes, every subsystem composes correctly. It runs in seconds on CPU
with a toy model, exercising the exact code paths a real run would use.
"""

from __future__ import annotations

import numpy as np
import torch

from pyllm.config import ModelConfig, TrainConfig
from pyllm.data import clean_document, load_bin, pack_dataset
from pyllm.eval import evaluate_perplexity
from pyllm.inference import generate
from pyllm.model import PyLLM
from pyllm.tokenizer import train_bpe_tokenizer
from pyllm.training import Trainer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
    "for i in range(10):\n    print(i)\n",
    "x = [i * i for i in range(5)]\n",
] * 40


def test_end_to_end_pipeline(tmp_path):
    torch.manual_seed(0)

    # 1) clean
    docs = [d for d in (clean_document(c) for c in CORPUS) if d is not None]
    assert docs

    # 2) tokenizer (tiny vocab, byte-level BPE)
    tok = train_bpe_tokenizer(docs, vocab_size=300, min_frequency=1)

    # 3) pack to shards
    stats = pack_dataset(docs, tok, tmp_path, val_fraction=0.2, seed=0)
    assert stats["train_tokens"] > 0 and stats["val_tokens"] > 0
    train_data = load_bin(tmp_path / "train.bin")
    val_data = load_bin(tmp_path / "val.bin")

    # 4) model sized to the tokenizer's actual vocab
    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, seq_len=32
    )
    model = PyLLM(model_cfg)

    # 5) train briefly; loss must drop
    train_cfg = TrainConfig(
        lr=3e-3, min_lr=3e-4, warmup_steps=10, max_steps=120,
        micro_batch_size=8, grad_accum_steps=1,
        eval_interval=1000, log_interval=1000, device="cpu", dtype="float32",
    )
    trainer = Trainer(model, model_cfg, train_cfg, train_data, val_data)
    before = trainer.estimate_loss()["train"]
    trainer.train()
    after = trainer.estimate_loss()["train"]
    assert after < before

    # 6) generate from a prompt and decode to text
    prompt = torch.tensor([tok.encode("def ")], dtype=torch.long)
    out = generate(model, prompt, max_new_tokens=20, temperature=0.8, top_k=20,
                   generator=torch.Generator().manual_seed(0))
    text = tok.decode(out[0].tolist())
    assert isinstance(text, str) and len(text) >= len("def ")

    # 7) perplexity is finite
    ppl = evaluate_perplexity(model, val_data, seq_len=32, batch_size=4, max_batches=5)
    assert np.isfinite(ppl["perplexity"])
