# Repository Overview

## Overview

This report outlines the repository structure, identifying the primary source of truth for the academic paper and the key artifacts available for integration.

**Key Files:** `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/main.tex`, `tokenizer.bib`.

## Repository Structure

### Paper Source of Truth

The compiled paper resides in `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/`.

- **Core File:** `main.tex` (Entry point).
- **Chapters:** `abstract.tex`, `introduction.tex`, `methodology.tex`, `results_and_analysis.tex`, `related_work.tex`, `future_work.tex`, `conclusion.tex`.
- **Bibliography:** `tokenizer.bib`.

### Downstream Evaluation Artifacts

These files contain the raw data and charts for the results section:

- **STS:** `STS_BENCHMARK_RESULTS.md`, `sts_benchmark_results.json`, `sts_benchmark_chart_*.png`.
- **MTEB:** `MTEB_BENCHMARK_RESULTS.md`, `results/**`, `mteb_average_scores.png`.
- **Version History:** `VERSION_BENCHMARK_RESULTS.md`, `version_history_*.png`.
- **Training:** `TRAINING_DETAILS.md` (Reproducibility).

### Tokenizer Implementation

Core code artifacts to document in the Methodology section:

- **Code:** `turkish_tokenizer.py`, `turkish_decoder.py`.
- **Resources:** `kokler.json`, `ekler.json`, `bpe_tokenler.json`.

### Vendored Dependencies

Modified external libraries:

- `sentence_transformers/`: Patched for custom tokenizer support (bypass).
- `mteb-tr/`: Local evaluation framework.

## Paper Integration Points

- **General:** Use this overview to locate the correct file when writing specific sections of the paper.
- **Bibliography:** Ensure all citations are centralized in `tokenizer.bib`.
