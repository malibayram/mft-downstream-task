# Vendored Dependencies Report

## Overview

The repository vendors `sentence-transformers` and `mteb-tr`. This report documents _why_ we vendored them and _what_ we changed.

**Key Files:** `sentence_transformers/`, `mteb-tr/`.

## 1. sentence-transformers

**Why vendored?**
To support custom tokenizers (like our `TurkishTokenizer`) that are not essentially HuggingFace AutoTokenizers, specifically in the `Model` class loading phase.

**Modifications:**

- **File:** `sentence_transformers/models/Transformer.py`
- **Change:** Added a `try-except` block around `AutoTokenizer.from_pretrained`.
- **Effect:** If the HF tokenizer fails to load (or if we intentionally don't want to load it because we are using pre-computed `input_ids`), the code proceeds without crashing. This is critical for our "Tokenizer Bypass" strategy where `input_ids` are generated offline by `prepare_dataset.py`.

## 2. mteb-tr

**Why vendored?**
To have a stable, local version of the Turkish MTEB tasks for consistent evaluation.

**Modifications:**

- None significant; primarily used as a library for task definitions.

## Paper Integration Points

- **Appendix / Reproducibility:**
  - Briefly mention that a "modified version of sentence-transformers was used to support custom tokenizer integration."
  - This serves as a technical detail for anyone trying to reproduce the code.
