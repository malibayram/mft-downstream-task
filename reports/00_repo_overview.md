# Repo Overview (for paper integration)

## Paper source of truth

The compiled paper lives under `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/`.
Primary files to edit:
- `main.tex` (entry point / preamble)
- `abstract.tex`, `introduction.tex`, `methodology.tex`, `results_and_analysis.tex`, `related_work.tex`, `future_work.tex`, `conclusion.tex`
- `tokenizer.bib`

## Downstream evaluation artifacts (to integrate)

- `STS_BENCHMARK_RESULTS.md` + `sts_benchmark_results.json` + `sts_benchmark_chart_*.png`
- `MTEB_BENCHMARK_RESULTS.md` + `results/**` + `mteb_average_scores.png`
- `VERSION_BENCHMARK_RESULTS.md` + `version_history_*.png`
- `TRAINING_DETAILS.md` (reproducibility details; should be referenced but not over-emphasized)

## Tokenizer implementation artifacts (to document)

- `turkish_tokenizer.py` (main tokenizer)
- `turkish_decoder.py` (decoder rules)
- Dictionaries: `kokler.json`, `ekler.json`, `bpe_tokenler.json`

## Vendored code (targeted documentation only)

- `sentence_transformers/`: local fork includes a tokenizer load bypass / custom tokenizer support.
- `mteb-tr/`: evaluation framework and task definitions used for MTEB-style results.

