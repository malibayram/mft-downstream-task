# Experiment Construction & Pipeline

## Overview

This document details the exact steps taken to construct the experiments for the "Tokens with Meaning" paper, including data preparation, model initialization, and training protocols.

**Key Files:** `prepare_dataset.py`, `train.py`, `evaluate_sts_tr.py`, `mteb-tr/mteb_tr_cli.py`.

## 1. Dataset Preparation

We constructed a unified training dataset (`alibayram/cosmos-corpus-encoded`) containing pre-tokenized sequences for all target tokenizers to ensure consistent training data across experiments.

### Source Data

- **Corpus**: `alibayram/cosmos-corpus-0-05-with-embeddings`
- **Content**: Turkish text with aligned teacher embeddings (used for distillation).

### Tokenization Strategy

We pre-encoded the text using four distinct tokenizers to facilitate efficient training without runtime tokenization overhead.

1.  **MFT (Ours)**: `turkish_tokenizer` (Morphology-First Tokenizer)
2.  **Tabi (Baseline)**: `alibayram/TabiBERT-tokenizer-32k`
3.  **CosmosGPT2 (Baseline)**: `alibayram/cosmosGPT2-tokenizer-32k`
4.  **Mursit (Baseline)**: `alibayram/newmindaiMursit-tokenizer-32k`

### Processing Pipeline (`prepare_dataset.py`)

The dataset preparation involved two stages:

**Stage 1: Initial Encoding (MFT & Tabi)**

- Loaded raw text from source.
- Encoded "mft_input_ids" and "tabi_input_ids".
- Filtered sequences > 2048 tokens.
- Saved to `alibayram/cosmos-corpus-encoded`.

**Stage 2: Extended Encoding (Cosmos & Mursit)**

- Loaded the stage 1 dataset (`alibayram/cosmos-corpus-encoded`).
- Loaded tokenizers from the initialized random models (`alibayram/cosmosGPT2-random-init` and `alibayram/newmindaiMursit-random-init`) to ensure vocabulary alignment (Vocab Size: 32,768).
- Appended "cosmos_input_ids" and "mursit_input_ids".
- Maintained the same row alignment and length filtering.
- Updated `alibayram/cosmos-corpus-encoded` with all 4 input columns.

## 2. Model Initialization

To isolate the effect of tokenization, we initialized all embedding models randomly with identical architectural constraints and seed.

### Base Architecture

- **Model**: `google/embeddinggemma-300m` architecture.
- **Modifications**: Resized embedding layer to exactly **32,768** to match all tokenizers.
- **Seed**: Fixed random seed (`42`) for weight initialization.

### Initialization Script (`create_random_models.py` / `random_init.py`)

We programmatically created random-initialized versions for all baselines using the same seed (`42`):

| Model ID                                | Tokenizer Source                          | Special Config           |
| :-------------------------------------- | :---------------------------------------- | :----------------------- |
| `alibayram/mft-random-init`             | `turkish_tokenizer` (Custom)              | No special tokens needed |
| `alibayram/tabi-random-init`            | `alibayram/TabiBERT-tokenizer-32k`        | Standard config          |
| `alibayram/cosmosGPT2-random-init`      | `alibayram/cosmosGPT2-tokenizer-32k`      | Pad Token ID: 0          |
| `alibayram/newmindaiMursit-random-init` | `alibayram/newmindaiMursit-tokenizer-32k` | Pad Token ID: 0          |

This ensured that `MFT`, `Tabi`, `Cosmos`, and `Mursit` models all started from the _exact same_ distribution of random weights, differing only in their tokenizer and embedding layer mapping.

## 3. Training Protocol

We employed a **distillation-based training** strategy (despite the focus on random-init comparison in the results, the pipeline supports distillation) using the `EmbeddingDistillationTrainer`.

### Training Configuration (`train.py`)

- **Objective**: Cosine Similarity Loss against Teacher Embeddings (`teacher_embedding_final`).
- **Optimization**:
  - Batch Size: 256
  - Learning Rate: 5e-5
  - Scheduler: Warmup (100 steps) + Linear Decay
  - Epochs: 1
  - Precision: bf16

### Model Specifics

The training script (`train.py`) iterates through the defined models, pulling the specific pre-encoded column for each:

- **MFT**: Uses `mft_input_ids`
- **Tabi**: Uses `tabi_input_ids`
- **Cosmos**: Uses `cosmos_input_ids`
- **Mursit**: Uses `mursit_input_ids`

## 4. Evaluation Protocols

Evaluation was performed on three downstream benchmarks using dedicated scripts to ensure consistency across all models.

### 4.1 Semantic Textual Similarity (STS)

- **Script**: `evaluate_sts_tr.py`
- **Dataset**: `figenfikri/stsb_tr` (Turkish STS benchmark)
- **Metric**: Pearson & Spearman Correlation
- **Execution**:
  ```bash
  python evaluate_sts_tr.py --model "alibayram/cosmosGPT2-random-init" "alibayram/newmindaiMursit-random-init"
  ```

### 4.2 Massive Text Embedding Benchmark (MTEB-TR)

- **Script**: `mteb-tr/mteb_tr_cli.py`
- **Benchmark**: `MTEB(Turkish)` (Comprehensive suite including Classification, Clustering, Retrieval, STS)
- **Execution**:
  ```bash
  python mteb-tr/mteb_tr_cli.py "alibayram/cosmosGPT2-random-init" --output-folder results/
  python mteb-tr/mteb_tr_cli.py "alibayram/newmindaiMursit-random-init" --output-folder results/
  ```

### 4.3 TurBLiMP (Linguistic Minimal Pairs)

- **Script**: `evaluate_turblimp.py`
- **Dataset**: `TurBLiMP/data/base` (Grammaticality judgment pairs)
- **Metric**: Average Cosine Similarity (Sensitivity)
- **Execution**:
  ```bash
  python evaluate_turblimp.py --model_path "alibayram/cosmosGPT2-random-init" --data_dir TurBLiMP/data/base --output_dir turblimp_results
  python evaluate_turblimp.py --model_path "alibayram/newmindaiMursit-random-init" --data_dir TurBLiMP/data/base --output_dir turblimp_results
  ```

### 4.4 Result Aggregation

Results from these scripts are aggregated into summary tables (e.g., `STS_BENCHMARK_RESULTS.md`) for final comparison.

## Paper Integration Points

- **Methodology / Experimental Setup:**
  - Cite the "Dual/Quad Tokenization" strategy in the Data Preparation section.
  - Emphasize strict length filtering (2048) to guarantee identical context across all 4 models.
  - Detail the random initialization (Seed 42) to preemptively address "lucky seed" counter-arguments.
  - Reference the specific evaluation scripts (`evaluate_sts_tr.py`, etc.) if providing a "Reproducibility" footnote.
