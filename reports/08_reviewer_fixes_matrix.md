# Reviewer Fixes Matrix (Jan 2026) → Planned Edits

This file is a working checklist mapping each reviewer point to specific paper edits.

## Missing downstream task performance (major)

- Add **Downstream Performance** subsection to `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`.
  - STS summary + figure + key deltas (MFT vs Tabi) + random-init sanity check.
  - MTEB summary + overall + category averages + short notes on where Tabi wins vs where MFT leads.
  - Version benchmark: short robustness paragraph in main text + detailed table in Appendix if needed.

## Intro contains results (structural)

- Refactor `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/introduction.tex`:
  - Move quantitative claims (TR%, Pure%, model-by-model comparisons) to Results.
  - Keep: problem setup, research questions, contributions, and brief metric definitions only.

## Methodology needs subsections + algorithm box

- Refactor `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/methodology.tex`:
  - Add titled subsections: Dictionary Construction, Normalization, Encoding, BPE fallback, Decoding, Edge cases.
  - Add an algorithm/pseudocode box for the full pipeline.
  - Remove evaluative statements (performance claims) from Methodology.

## Linguistic examples need Leipzig glossing + consistency

- Add consistent example formatting (segmentation line, gloss line, translation line).
- Update all examples in `methodology.tex` and `results_and_analysis.tex` to include glossing.

## Citation style and bibliography hygiene

- Fix “\cite as sentence constituent” style in `related_work.tex`.
- Merge `papers/papers.bib` into `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/tokenizer.bib`.
- Replace weak/non-rigorous sources if present (avoid LinkedIn-style citations).
- Normalize title casing in BibTeX where feasible.

## Scope / generalization claims

- Reframe language independence as “language-agnostic framework” and explicitly note what requires language-specific resources.
- Add balanced related work acknowledging contradictory findings (referenced by reviewers via URLs) and position as open debate when no clean citation metadata is available locally.

## Transparency about data sources

- Add explicit sources for:
  - Turkish web/corpus text used in examples/evaluation
  - corpora used for root extraction and BPE training (names + URLs if needed)

## Efficiency metrics

- Avoid unsupported “Rust is faster” claims unless measured.
- Optionally add a tokenizer throughput microbenchmark; report as tokenization throughput (not model training speed).

