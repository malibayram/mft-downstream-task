# Dataset Preparation Report

## Overview

The dataset preparation pipeline (`prepare_dataset.py`) creates a fixed training corpus for embedding distillation. It ensures that both the student (MFT/Tabi) and teacher models see identical data, pre-processed to strict length constraints.

**Key File:** `prepare_dataset.py`

## Dataset Specifications

- **Source:** `alibayram/cosmos-corpus-0-05-with-embeddings` (Contains text + pre-computed teacher embeddings).
- **Teacher Model:** Used to generate `teacher_embedding_final` column (vectors).
- **Max Sequence Length:** 2048 tokens.

## Processing Pipeline

1.  **Dual Tokenization:**
    - Every example is encoded twice: once with `TurkishTokenizer` (MFT) and once with `TabiBERT` tokenizer.
    - This produces `mft_input_ids` and `tabi_input_ids`.

2.  **Strict Filtering:**
    - Any document where _either_ the MFT sequence OR the Tabi sequence exceeds 2048 tokens is dropped.
    - **Rationale:** Ensures fair comparison. If a document is too long for one tokenizer but not the other, truncation would mean they see different content. Dropping ensures identical content coverage.

3.  **Output Format:**
    - `text`: Original raw text.
    - `mft_input_ids`: List[int].
    - `tabi_input_ids`: List[int].
    - `teacher_embedding_final`: List[float] (Target for distillation).

## Paper Integration Points

- **Methodology / Experimental Setup:**
  - Describe the dataset source.
  - Explicitly mention the "Dual Tokenization & Filter" strategy to guarantee apples-to-apples comparison on context window usage.
  - Cite the max sequence length (2048).
