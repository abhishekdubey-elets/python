"""pyllm: a compact, Python-specialized decoder-only Transformer, from scratch.

The public surface is intentionally tiny for now. As we build each phase we will
re-export the pieces that form the stable API (model, tokenizer, trainer, ...).
"""

from pyllm.config import DataConfig, ModelConfig, TrainConfig

__version__ = "0.1.0"

__all__ = ["ModelConfig", "TrainConfig", "DataConfig", "__version__"]
