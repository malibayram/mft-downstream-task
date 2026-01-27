# STS Evaluation Report

## Overview

We evaluated all models on the **STS Benchmark for Turkish (STSb-TR)**. This measures the ability of the embeddings to capture semantic similarity, which is the direct objective of the distillation.

**Source of Truth:** `STS_BENCHMARK_RESULTS.md`

## Key Findings

### 1. MFT Outperforms Tabi significantly

- **MFT Models Average:** 72.72% Pearson.
- **Tabi Models Average:** 63.91% Pearson.
- **Delta:** MFT is **+8.80 points** better on average.

### 2. Best Model

- **mft-downstream-task-embeddingmagibu** acheived **74.41% Pearson** (Test Split).

### 3. Random Initialization Check

- **mft-random-init** (47.09%) vs **tabi-random-init** (40.53%).
- Even with random weights, the MFT tokenization structure provides a better starting point/inductive bias for Turkish, or preserves more information in the token sequence.

## Data for Paper

**Table: STS Test Set Performance (Pearson Correlation x100)**

| Model Architecture         | Tokenizer       | Pearson ($\rho$) | Spearman ($r$) |
| :------------------------- | :-------------- | :--------------- | :------------- |
| **EmbeddingMagibu (200M)** | **MFT (Ours)**  | **74.41**        | **73.08**      |
|                            | Tabi (Baseline) | 66.29            | 64.97          |
| **EmbeddingGemma (300M)**  | **MFT (Ours)**  | **71.02**        | **70.00**      |
|                            | Tabi (Baseline) | 61.54            | 60.56          |
| _Random Init_              | _MFT_           | _47.09_          | _45.96_        |
|                            | _Tabi_          | _40.53_          | _38.60_        |

## Paper Integration Points

- **Results Section (New Subsection: "Downstream Semantic Similarity"):**
  - Insert the table above.
  - Include `sts_benchmark_chart_test.png` (or a cleaner generated version).
  - Narrative: "Under identical distillation budgets, MFT-based models consistently achieve higher correlation with human similarity judgments compared to standard BPE baselines."
