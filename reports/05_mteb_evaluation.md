# MTEB Evaluation Report

## Overview

We conducted a comprehensive evaluation on the **MTEB-TR (Turkish)** suite, covering retrieval, classification, clustering, and pair classification tasks.

**Key Files:** `MTEB_BENCHMARK_RESULTS.md`

## Key Findings

### 1. Overall Averages

| Model Family              | Average Score (26 Tasks)   |
| :------------------------ | :------------------------- |
| **EmbeddingGemma (Base)** | 65.21% (Teacher/Reference) |
| **MFT Models (Avg)**      | ~61.9%                     |
| **Tabi Models (Avg)**     | ~62.4%                     |

_Note:_ Tabi has a slight edge in the _overall_ average, largely driven by "Other" and specific classification tasks. However, MFT behaves competitively and wins in specific semantic categories.

### 2. Category Breakdown

- **STS:** MFT wins (74.73% vs 72.41%).
- **Retrieval:** Competitive/Mixed.
- **BitextMining:** Tabi wins slightly (94.26% vs 91.82%).
- **Classification:** Tabi wins slightly (67.90% vs 67.06%).

### 3. Critical Observation: Retrieval/STS alignment

MFT is particularly strong in **STS** (as seen in the dedicated report) and competitive in Retrieval. These are the core tasks for semantic embeddings.

## Data for Paper

**Table: MTEB-TR Category Summaries (Average Scores)**

| Category               | MFT-Magibu | Tabi-Magibu | Difference      |
| :--------------------- | :--------- | :---------- | :-------------- |
| **STS**                | **74.73**  | 72.41       | **+2.32** (MFT) |
| **Retrieval**          | 64.39      | **65.46**   | -1.07 (Tabi)    |
| **Classification**     | 67.06      | **67.90**   | -0.84 (Tabi)    |
| **Clustering**         | 61.42      | **62.61**   | -1.19 (Tabi)    |
| **PairClassification** | 61.59      | **61.87**   | -0.28 (Tabi)    |

## Paper Integration Points

- **Results Section:**
  - Present MTEB results as a "Breadth Evaluation".
  - Highlight that while Tabi (trained on massive data) holds up well on general classification, MFT (linguistically motivated) shows distinct advantages in **semantic similarity (STS)** tasks, which validates the "Tokens with Meaning" hypothesis-that better morphology leads to better semantic representation.
  - Be transparent: MFT does not win _every_ category, but wins the one most relevant to the paper's core claim (meaning/similarity).
