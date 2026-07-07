# 02 — Tokenizer

> Phase 3. Deliverable: a byte-level BPE tokenizer optimized for Python, with a
> from-scratch reference implementation, exact-round-trip tests, and a real
> 50,257-token tokenizer trained on the Python standard library.

## 1. Intuition

A model consumes integers, not text. The tokenizer is the reversible bridge
`text ↔ list[int]`. It quietly decides three things that shape everything
downstream:

* **Vocabulary size** — how many rows the embedding table has (and, tied, the
  output head). At 50,257 that's ~31% of a 125M model.
* **Sequence length** — better compression = more code per context window.
* **Robustness** — can it encode *any* input, or does it choke on something?

## 2. The four approaches

| Approach | Base unit | OOV? | Notes |
|---|---|---|---|
| Word-level | words | **yes, fatal** | every new identifier is unseen — unusable for code |
| Char-BPE | unicode chars | yes (unseen chars) | needs an `<unk>` token |
| **Byte-level BPE** | 256 bytes | **never** | GPT-2/Llama/StarCoder; our choice |
| Unigram (SentencePiece) | prunes from big vocab | never | principled; more complex, best for multilingual NL |

**We use byte-level BPE.** The base alphabet is the 256 possible bytes, so *any*
byte sequence is representable — no `<unk>`, ever. That robustness matters for
code, which is full of novel identifiers, rare unicode in strings/comments, and
odd byte sequences.

## 3. The BPE algorithm (worked example)

BPE = "repeatedly merge the most frequent adjacent pair."

Start: `a a a b d a a a b a c`
1. `aa` is most frequent → merge to `Z`: `Z a b d Z a b a c`
2. `ab`… → `Y`: `Z Y d Z Y a c`
3. `ZY`… → `X`: `X d X a c`

Each merge adds one vocab entry and is recorded with a priority. **Encoding** a
new string re-applies the learned merges in priority order; **decoding**
concatenates the byte strings each id expands to. See
[`bpe_reference.py`](../src/pyllm/tokenizer/bpe_reference.py) — the whole thing
is ~60 lines with no magic. Complexity of the naive trainer is
`O(num_merges × len(text))`; production trainers (the Rust `tokenizers` lib we
actually use) keep incremental pair counts to run much faster for an identical
result.

## 4. Why code tokenization differs from natural language

1. **Whitespace is syntax.** In Python, indentation *is* structure. Byte-level
   BPE doesn't discard it — it *learns* it. On the stdlib, our tokenizer turned
   4, 8, and 12 spaces each into a **single token**:

   ```
    4 spaces -> 1 token   ('ĠĠĠĠ')          # Ġ is ByteLevel's visual for a space
    8 spaces -> 1 token
   12 spaces -> 1 token
   ```

   One indent level = one token: compression *and* structure the model can latch
   onto.

2. **Skewed frequency.** `self`, `return`, `import`, `def`, `):` dominate Python,
   so the merge budget is spent making them cheap. Measured on our tokenizer:
   `self`, `return`, `import` are each a single token.

3. **Digits & operators.** Code-specific tokenizers (e.g. GPT-4's `cl100k`)
   often split numbers into individual digits and use a custom pre-tokenization
   regex. We use the GPT-2 ByteLevel regex by default; swapping in a code regex
   is a one-line change in [`train.py`](../src/pyllm/tokenizer/train.py) and a
   good future experiment.

Net result on a real snippet: **~2.78 bytes/token** on typical Python.

## 5. Design choices in our implementation

* **`unk_token=None`** — byte-level means unseen input is impossible, so there's
  no unknown token.
* **`add_prefix_space=False`** — do NOT inject a leading space; that would
  corrupt code (indentation, string contents).
* **ByteLevel pre-tokenizer + ByteLevel decoder** — together they guarantee
  `decode(encode(x)) == x` for arbitrary text (tested on tabs, 4-byte emoji,
  unicode, empty string).
* **Special tokens, added first (lowest ids):** `<|endoftext|>` (EOS + document
  separator, id 0), `<|pad|>`, and the three **FIM** tokens
  (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`). We reserve FIM now
  because adding tokens later would force retraining the tokenizer, even though
  fill-in-the-middle training comes much later.

## 6. `vocab_size` is a *ceiling*, not a guarantee

The trainer stops early if the corpus runs out of pairs above `min_frequency`.
On our tiny test corpus, asking for 500 yields 381. On the full stdlib, 50,257
is reached comfortably. **The tokenizer's real `vocab_size` must match
`ModelConfig.vocab_size`** — always read it back after training and set the model
config from it (a Phase 4 assertion will enforce this).

## 7. Training the real tokenizer

We trained on the **local Python standard library** — a real, PSF-licensed
Python corpus (~19k files) that ships with the interpreter, so no download:

```bash
python scripts/train_tokenizer.py \
  --input-dir "<python stdlib path>" \
  --vocab-size 50257 \
  --output tokenizer/pyllm.json
# -> vocab_size=50257  eos_id=0  round-trip ok=True   (~4 min on CPU)
```

The output is a single self-contained JSON (`tokenizer/pyllm.json`) loadable via
`PyTokenizer.load(...)`. (It is git-ignored as a build artifact; regenerate with
the script above. For real training we'd later expand the corpus well beyond the
stdlib — see Phase 5.)

## 8. Common mistakes

* **Injecting a prefix space** (`add_prefix_space=True`) — silently corrupts code
  round-trips. We disable it and test exact round-trip.
* **Mismatched vocab sizes** — tokenizer vs `ModelConfig.vocab_size` must be
  equal or the embedding/label ids go out of range.
* **Printing ByteLevel tokens on a non-UTF-8 console** — Windows `cp1252` can't
  render `Ġ`; set `PYTHONIOENCODING=utf-8`. (Does not affect training/round-trip,
  only display.)
* **Training the tokenizer on data that overlaps your eval set** — a
  contamination vector; keep tokenizer-training and eval corpora disjoint
  (Phase 5/8).

## 9. References

* Sennrich et al., 2016 — BPE for NMT (the original).
* Radford et al., 2019 (GPT-2) — byte-level BPE.
* Kudo, 2018 — Unigram LM / SentencePiece.
* Karpathy — `minbpe` (the spirit of `bpe_reference.py`).
* BigCode / StarCoder — tokenizer design for code at scale.
