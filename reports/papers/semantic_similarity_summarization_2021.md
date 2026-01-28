# Semantic Similarity Evaluation for Summarization (Beken Fikri et al., 2021) - Notes

Source: `papers/Beken Fikri et al. - 2021 - Semantic Similarity ... .pdf` (also in `papers/papers.bib` as `beken_fikri_semantic_2021`).

## What it is

Introduces semantic similarity models for Turkish and argues semantic similarity correlates better with human judgments than ROUGE for abstractive summarization; it also discusses Turkish STS data (translated STSb).

## Key ideas to cite in our paper

- Correlation-based evaluation (Pearson/Spearman) is meaningful for semantic similarity tasks.
- Agglutinative morphology can make lexical-overlap metrics (ROUGE) less reliable; semantic similarity is a better lens.

## How it relates to our work

- Directly supports the **STS evaluation framing** we add in the downstream section (Pearson/Spearman on Turkish STS).
- Provides a Turkish-focused motivation for semantic similarity evaluation beyond token purity.

## Paper insertion points

- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/related_work.tex`: cite when motivating downstream semantic similarity evaluation.
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`: cite alongside STS reporting.
