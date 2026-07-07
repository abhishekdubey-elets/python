"""The production tokenizer wrapper: a thin, typed layer over a trained
byte-level BPE ``tokenizers.Tokenizer``.

Why wrap it at all instead of using ``tokenizers.Tokenizer`` directly?
* A stable, minimal API the rest of the codebase depends on (encode/decode/save/
  load + special-token ids). If we ever swap the backend, only this file changes.
* Convenience the raw library doesn't give: ``add_eos``, named special-token id
  properties, and clear errors.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer

# Special tokens. We reserve the FIM (fill-in-the-middle) tokens now — adding
# them later would force retraining the whole tokenizer — even though we won't
# use infilling until much later. EOS doubles as the document separator.
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"

DEFAULT_SPECIAL_TOKENS: list[str] = [
    EOS_TOKEN,
    PAD_TOKEN,
    FIM_PREFIX,
    FIM_MIDDLE,
    FIM_SUFFIX,
]


class PyTokenizer:
    """Reversible text <-> token-id mapping for pyllm.

    The underlying model is byte-level BPE, so ``decode(encode(x)) == x`` exactly
    for *any* string (whitespace, unicode, indentation all preserved), and no
    input is ever out-of-vocabulary.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    # --- vocabulary ----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Total number of tokens, including special tokens. Must equal the
        model's ``ModelConfig.vocab_size``."""
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def id_to_token(self, idx: int) -> str | None:
        return self._tok.id_to_token(idx)

    # --- special-token ids (None if the tokenizer lacks them) ----------------
    @property
    def eos_id(self) -> int | None:
        return self._tok.token_to_id(EOS_TOKEN)

    @property
    def pad_id(self) -> int | None:
        return self._tok.token_to_id(PAD_TOKEN)

    # --- encode / decode -----------------------------------------------------
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Text -> token ids. Optionally append the EOS/document-separator id."""
        ids = self._tok.encode(text).ids
        if add_eos:
            eos = self.eos_id
            if eos is None:
                raise ValueError("This tokenizer has no EOS token to append.")
            ids = ids + [eos]
        return ids

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tok.encode_batch(list(texts))]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Token ids -> text. By default special tokens are dropped from output."""
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save to a single self-contained JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> PyTokenizer:
        return cls(Tokenizer.from_file(str(path)))
