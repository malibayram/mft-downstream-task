# TabiBERT (Türker et al., 2026) — Notes

Source: `papers/Türker et al. - 2026 - TabiBERT ... .pdf` (also in `papers/papers.bib` as `turker_tabibert_2026`).

## What it is
TabiBERT is a monolingual Turkish encoder based on ModernBERT and introduces the TabiBench benchmark.

## Key ideas to cite in our paper
- Turkish-specific encoders/benchmarks matter for fair evaluation and reproducibility.
- TabiBERT supports longer context windows and reports strong benchmark performance within Turkish NLP.

## How it relates to our work
- Provides a **Turkish-specific baseline** ecosystem and supports the critique that English-centric tokenizers can be unfair baselines.
- Useful for positioning our “Tabi tokenizer baseline” in downstream evaluations (where applicable).

## Paper insertion points
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/related_work.tex`: cite to broaden beyond “English-heavy general tokenizers” and acknowledge Turkish-specific model/tokenizer ecosystems.
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`: mention as an example of Turkish-focused evaluation efforts; avoid claiming direct comparability unless we actually run their benchmarks.

