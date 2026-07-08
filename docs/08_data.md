# 08 — Data pipeline

> Phase 5. Deliverable: clean → dedup → pack → load, all tested.

## 1. Why data is (most of) the model

For a small model, data quality dominates. Garbage, duplicates, and leaked eval
examples hurt a 125M model far more than a 7B one — it has no spare capacity to
average out noise. The pipeline is a stack of cheap, conservative filters.

## 2. Sources (for a real Python corpus)

Permissively-licensed GitHub repos, the standard library, official docs, unit
tests, tutorials, bug-fix commits, notebooks, and competitive-programming
solutions. Respect licensing (train only on permissive licenses) and keep any
eval benchmarks **out** of training (contamination). For this project we
demonstrate on the local stdlib (PSF-licensed, already on disk).

## 3. Stages (`src/pyllm/data/`)

1. **clean.py** — `normalize_text` (line endings, NFC, trailing ws), quality
   gate (`passes_quality`: min lines, max line length, alpha fraction), Python
   validity (`is_valid_python` via `ast.parse`), and a license allow-list.
   *Tuning note:* the alpha-fraction floor is 0.15 — code has many
   spaces/operators/digits, so a natural-language-style 0.25 wrongly rejects
   valid code.
2. **dedup.py** — exact dedup by SHA-256, and near-dup via **MinHash + LSH**:
   MinHash estimates Jaccard similarity of k-shingle sets cheaply; LSH banding
   buckets likely-similar docs so we compare only within buckets, then confirm
   against a threshold. O(n) exact, ~O(n) near with banding.
3. **pack.py** — tokenize each doc (with a trailing EOS separator), concatenate
   into one `uint16` stream, split at the **document** level (no token-level
   leakage), write `train.bin`/`val.bin` + `meta.json`. uint16 because our vocab
   < 65536.
4. **loader.py** — `PackedDataset` (contiguous (x, y) blocks over a memmap) and
   `get_batch` (nanoGPT-style random-offset sampling). Target = input shifted by
   one.

## 4. Common mistakes

* Token-level train/val split → the tail of a training doc leaks into val.
* Skipping dedup → memorization + inflated eval.
* Storing tokens as int64 → 4× disk/bandwidth for nothing.
* Forgetting the EOS separator → documents bleed into each other.

## 5. References

* Lee et al., 2022 — *Deduplicating Training Data Makes LMs Better*.
* Broder, 1997 — MinHash. Kocetkov et al., 2022 — *The Stack* (code data).
