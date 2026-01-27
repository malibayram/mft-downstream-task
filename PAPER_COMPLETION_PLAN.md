# Paper Completion Plan: *Tokens with Meaning* — Revision + Downstream Evaluation Integration

## 0) Goal (what “done” means)

1. **Complete and polish the paper** in `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/` so it is internally consistent, reproducible, and includes the repo’s experimental evidence.
2. **Read and cite every paper in `papers/`** and mention each in the paper text (at least once, in an appropriate section).
3. **Add downstream task evidence** (STS + MTEB + version tracking) to address reviewer concerns, with clear methodology, results tables/figures, and limitations.
4. **Create small Markdown “reports” per folder/file first**, then convert the key points into LaTeX edits (paper sections, tables, figures, citations).

Deliverable: updated LaTeX sources + updated `tokenizer.bib` + any new `*.tex` tables/figures needed, and the paper compiles cleanly.

---

## 1) Scope & important clarifications (updated with your decisions + reviews)

### 1.0 Confirmed decisions (from you)
- **Downstream distillation experiments will be part of the main Results** (not Appendix-only), with the Appendix used only for long version-history tables if needed.
- **Tokenizer comparison is the objective**; we will not center narrative on architecture effects (Gemma vs Magibu), beyond minimal reporting as a control.
- If it improves clarity/presentation, it’s OK to **re-run report-generation scripts** and/or add a small helper script for cleaner charts/tables.

### 1.0.1 Wording alignment (per your request)
We will **de-emphasize “distillation” terminology** in the paper and frame the new evidence as:
- **Downstream sentence embedding evaluation** (STS and MTEB-style retrieval/classification tasks).
- Training details (teacher-guided / embedding-alignment) will be described *only as needed for reproducibility* and will not be positioned as the key contribution.

Important: we will not claim “the result will be similar for pretraining/fine-tuning” as a fact unless we add evidence. We can safely phrase it as a **hypothesis/expectation** and explicitly label it as such (or keep it as Future Work).

### 1.1 “Read all files detailly”
This repo includes large vendored codebases (notably `sentence_transformers/` and `mteb-tr/`). A literal line-by-line “deep reading” of those would be extremely time-consuming and mostly irrelevant to the paper.

Proposed interpretation (recommended):
- **Project-owned code** (top-level scripts + tokenizer implementation + results artifacts) gets deep reading and detailed reporting.
- **Vendored libraries** get **targeted reading**: we document (a) what they are, (b) how we use them, and (c) any modifications we made (e.g., tokenizer bypass in `sentence_transformers/models/Transformer.py`).

Assumption (unless you object): **targeted reading for vendored libraries** + deep reading for project-owned code and produced artifacts.

### 1.2 Re-running experiments vs. using existing artifacts
Some scripts depend on HuggingFace datasets/models and require network/HF tokens (e.g., `prepare_dataset.py`, `train.py`, `evaluate_sts_tr.py` for remote models).

Proposed approach:
- Use existing artifacts already in-repo (`*_BENCHMARK_RESULTS.md`, `*.json`, `results/**`, `*.png`, `*.pdf`) as the “source of truth”.
- Optionally re-run selected scripts only if you want fresh reproduction checks.

Assumption (per your note): **only re-run if it improves tables/figures**, otherwise integrate existing artifacts.

---

## 2) Reviewer-driven revision priorities (new; ties plan directly to feedback)

This section translates your Jan 2026 reviews into concrete edits so the revision reads like a response to reviewers.

### 2.1 Must-fix (blocked acceptance)
- **Add downstream task performance** (address Reviewer Wx9w + multiple others): integrate STS + MTEB results into `results_and_analysis.tex`.
- **Move quantitative findings out of Introduction** (Reviewer 1): refactor `introduction.tex` so it sets up problem/RQs/contributions; keep numbers in Results.
- **Restructure Methodology with clear subsections + algorithm box/pseudocode** (Reviewer 1, 3): add titled subsections and a step-by-step algorithm box; remove evaluative claims from Methodology.
- **Define TR% and Pure% clearly where first used (and briefly in abstract)** (Reviewer 4): add crisp definitions, including how “morpheme alignment/purity” is computed.
- **Fix citation style (“[17] demonstrated…”) + remove weak sources** (Reviewer 2, 4): adopt “FirstAuthor et al.” phrasing; remove/replace non-rigorous citations (e.g., LinkedIn) with peer-reviewed sources.

### 2.2 Strongly recommended (to strengthen credibility)
- **Linguistic examples with Leipzig glosses** (Reviewer jLKM): add morpheme-by-morpheme glosses + full-word meaning for *every* showcased example; standardize formatting.
- **Clarify data sources used for dictionary/BPE training** (Reviewer jLKM, Reviewer 1): cite corpora by name at first mention and describe extraction method.
- **Clarify edge cases**: uppercase sequences (e.g., `HTTPServer`), acronyms, compounds handling; clarify “special category” and “distinct surface forms”.
- **Efficiency metrics** (Reviewer 3/4): measure and report tokenization throughput/latency (and clearly label what is measured).

### 2.3 Scope/claims corrections
- **Language independence claim**: reframe to “language-agnostic framework” and explicitly state what requires language-specific resources (root/affix lists), and what transfers. Keep non-Turkish benchmarks as Future Work unless we add new experiments.
- **Balance related work**: add citations representing *contradictory findings* (Reviewer 4) and broaden beyond agglutinative-only where possible (e.g., inflectional languages; include Slovak morphological tokenizer reference from review).

---

## 3) Narrative goals for the downstream section (paper-objective aligned)

#### 1.3.1 Core claims we will support using repo artifacts
1. **STS:** MFT-distilled models achieve materially higher Pearson/Spearman on `figenfikri/stsb_tr` than Tabi-distilled models (source: `STS_BENCHMARK_RESULTS.md`, `sts_benchmark_results.json`).
2. **MTEB-TR:** Even if Tabi gets isolated wins on some tasks/aggregates, the tokenizer-first story prioritizes **semantic similarity + retrieval relevance**, where MFT is stronger overall in our evidence (source: `MTEB_BENCHMARK_RESULTS.md`, `results/**`).
3. **Random-init sanity check:** With random initialization, **Tabi does not catch MFT** on STS and does not surpass it on overall MTEB averages; this supports that improvements are not just noise (source: the random-init rows in both STS/MTEB reports).

#### 3.2 How we will phrase “Tabi/BPE doesn’t catch MFT” (and avoid over-claiming)
We will emphasize **quality under fixed training budget** (supported by artifacts) rather than asserting wall-clock speed:
- Primary statement (supported): **“Under the same downstream training budget, the Tabi baseline does not catch the proposed MFT tokenizer on STS/retrieval-focused quality.”**
- Secondary statement (only if we add evidence): tokenization throughput/efficiency via a microbenchmark, explicitly labeled as *tokenization speed*, not “training speed”.

---

## 4) Inventory (what we will touch and why)

### 2.1 Paper sources (primary)
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/main.tex` (main entry)
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/abstract.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/introduction.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/methodology.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/related_work.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/future_work.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/conclusion.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/tokenizer.bib`

### 4.2 “Downstream” experiment artifacts to integrate
- Protocol/writeups: `TRAINING_DETAILS.md`, `STS_BENCHMARK_RESULTS.md`, `MTEB_BENCHMARK_RESULTS.md`, `VERSION_BENCHMARK_RESULTS.md`
- Raw/structured results: `sts_benchmark_results.json`, `sts_benchmark_results_p.json`, `results/**`
- Figures: `sts_benchmark_chart*.png`, `mteb_average_scores.png`, `version_history_*.png`

### 4.3 Key experiment code (to document)
- Dataset: `prepare_dataset.py`, `upload_dataset.py`
- Training: `train.py`, `embedding_trainer.py`
- Evaluation/reporting: `evaluate_sts_tr.py`, `generate_sts_tables.py`, `generate_mteb_report.py`
- Baselines/setup: `random_init.py`, `setup_remote.sh`, `requirements.txt`
- Tokenizer + dictionaries: `turkish_tokenizer.py`, `turkish_decoder.py`, `kokler.json`, `ekler.json`, `bpe_tokenler.json`
- Integration test: `test_custom_tokenizer.py`

### 4.4 Papers that must be cited + mentioned (from `papers/`)
- `papers/Asgari et al. - 2025 - MorphBPE ... .pdf`
- `papers/Beken Fikri et al. - 2021 - Semantic Similarity Based Evaluation ... .pdf`
- `papers/Rahimov - 2025 - miLLi Model ... .pdf`
- `papers/Türker et al. - 2026 - TabiBERT ... .pdf`
- `papers/papers.bib` (BibTeX entries we can reuse/merge)

### 4.5 Vendored/auxiliary code (targeted documentation)
- `sentence_transformers/` (we will document the tokenizer bypass + how it enables offline `input_ids` training)
- `mteb-tr/` (we will document how we use it for MTEB-style evaluation + which tasks/metrics are reported)

---

## 5) Reporting workflow (Markdown first, then LaTeX)

I will create a `reports/` folder and produce short, focused Markdown notes before editing the paper. Proposed structure:

- `reports/00_repo_overview.md`
- `reports/01_tokenizer_implementation.md` (tokenizer pipeline + dictionaries + decoding)
- `reports/02_dataset_preparation.md` (`prepare_dataset.py`, dataset schema, filtering, sequence limits)
- `reports/03_training_and_setup.md` (`train.py`, `embedding_trainer.py`, hyperparams, hardware) — written for reproducibility, but kept secondary in paper narrative
- `reports/04_sts_evaluation.md` (`evaluate_sts_tr.py`, charts, JSON format)
- `reports/05_mteb_evaluation.md` (`generate_mteb_report.py`, `results/**`, aggregate metrics)
- `reports/06_version_tracking.md` (`VERSION_BENCHMARK_RESULTS.md`, what “revision” means, charts)
- `reports/07_vendored_dependencies.md` (what we rely on inside `sentence_transformers/` + `mteb-tr/`, and what we changed)
- `reports/08_reviewer_fixes_matrix.md` (each reviewer point → exact file/section change)
- `reports/papers/`:
  - `reports/papers/morphbpe_2025.md`
  - `reports/papers/semantic_similarity_summarization_2021.md`
  - `reports/papers/milli_2025.md`
  - `reports/papers/tabibert_2026.md`

Each report will follow the same mini-template:
- **What this file/folder is**
- **Key parameters/assumptions**
- **Outputs produced**
- **How it supports paper claims**
- **Exact LaTeX section(s) to update**

---

## 6) Paper integration plan (what will be added/changed)

### 6.1 Structural refactor to match reviewer expectations
1. **Title update** to include typology/language, e.g.:
   - “Tokens with Meaning: A Hybrid Tokenization Approach for Morphologically Rich (Agglutinative) Languages”
   - or explicitly “... for Turkish and Agglutinative Languages”
2. **Introduction refactor**:
   - remove/relocate quantitative results
   - add research questions + contributions list
3. **Methodology refactor**:
   - add subsections: *Dictionary Construction*, *Normalization Rules*, *Encoding Algorithm*, *BPE Fallback*, *Decoding*, *Edge Cases*
   - add an **Algorithm box/pseudocode** and one worked example with Leipzig gloss
4. **Related work cleanup**:
   - keep it as literature survey only; move speculative content to Discussion/Future Work
   - add contradictory findings + broaden beyond agglutinative-only where possible

### 6.2 Linguistic examples upgrade (Leipzig glossing)
For every example word/sentence:
- Add segmented form, gloss line, and free translation line (Leipzig conventions).
- Ensure consistent formatting (italicization, quotes, examples numbering).

### 6.3 Transparency & definitions
- Define **TR%** and **Pure%** early (abstract + first mention), including computation details.
- Cite and describe data sources for:
  - root extraction corpora (names + links/citations)
  - BPE training corpora + whether SentencePiece uses BPE or Unigram (explicit)
- Clarify whether tokenization is lossless; if not, state what information is abstracted and how decoding behaves.

### 6.4 Experiments: add downstream performance as **main results**
1. Read `main.tex` and each section `*.tex` end-to-end.
2. Identify gaps/inconsistencies:
   - Missing citations / uncited claims
   - Missing dataset/benchmark definitions
   - Terminology drift (MFT vs hybrid tokenizer naming)
   - Figures/tables referenced but not introduced (or vice versa)
3. Create a short “paper gap list” report: `reports/00_repo_overview.md`.

### 4.2 Literature: integrate and mention each `papers/` PDF
For each PDF in `papers/`:
1. Extract: problem, method, datasets, metrics, key findings, and how it relates to our approach.
2. Add/merge BibTeX entry into `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/tokenizer.bib`.
3. Add at least one explicit mention and citation in the paper text:
   - Likely `related_work.tex` for all four
   - Possibly `results_and_analysis.tex` (if directly comparable)
   - Possibly `methodology.tex` (if method inspiration/contrast)

Planned placement (initial proposal, will confirm after reading each PDF):
- MorphBPE (morphology-aware + BPE hybrid): `related_work.tex` + “hybrid tokenization” framing.
- miLLi (local linguistic insights for robust tokenization): `related_work.tex` + “linguistic priors” discussion.
- TabiBERT (Turkish foundation model + benchmark): `related_work.tex` + benchmark context; also cite where we describe Tabi tokenizer baseline.
- Semantic similarity for summarization evaluation (2021): cite when motivating correlation-based evaluation (ties cleanly to STS-like correlation reporting in our downstream experiments).

The previous submission lacked downstream task evidence. This revision adds **downstream evaluation** aligned with the paper’s goal: showing that more linguistically meaningful tokens translate to better semantic representations on real tasks.

Plan:
1. Add a short **Experimental Setup** subsection (keep “distillation” minimal):
   - what models are compared (MFT vs Tabi tokenization variants)
   - what tasks/benchmarks (STS + MTEB-TR)
   - how models are trained/evaluated (only essentials for reproducibility)
2. Add a **Downstream Performance** subsection in `results_and_analysis.tex`:
   - STS results summary (from `STS_BENCHMARK_RESULTS.md`)
   - MTEB results summary (from `MTEB_BENCHMARK_RESULTS.md`)
   - Version history analysis (from `VERSION_BENCHMARK_RESULTS.md`) as:
     - a short robustness/reproducibility note in the main text, and
     - an Appendix section only if the detailed history tables are too long
3. Add figures/tables:
   - Convert key Markdown tables into LaTeX `table` environments (or `\input{}` generated `*.tex` tables).
   - Include existing charts `sts_benchmark_chart_test.png`, `mteb_average_scores.png`, `version_history_pearson.png`, etc.
4. Add a limitations paragraph:
   - dependency on teacher model choice
   - remaining confounds (backbone differences are reported but not the focus)
   - what the random-init baseline demonstrates

### 4.4 Implementation details: tokenizer algorithm + dictionaries
In `methodology.tex` (or a dedicated implementation subsection):
- Tie the *code-level pipeline* (`turkish_tokenizer.py` + dictionaries) to the conceptual framework:
  - root finding / longest-match heuristic
  - affix segmentation + equivalence classes
  - phonological normalization and reverse decoding
  - BPE fallback behavior
  - casing/whitespace special tokens

Where helpful, add one compact algorithm/pseudocode block or flow figure (only if the paper benefits and space allows).

---

## 7) Concrete execution steps (what I will do once you approve)

### Phase A — Reporting (Markdown reports)
1. Create `reports/` and write `reports/00_repo_overview.md`.
2. Write the 6–8 focused reports listed in section 3.
3. For each report, include “paper insertion points” (exact `*.tex` file + subsection title).
4. Write `reports/08_reviewer_fixes_matrix.md` mapping each review point → exact edit.

### Phase B — Paper edits (LaTeX integration)
1. Update `related_work.tex` to include and discuss all `papers/` items.
2. Update `methodology.tex` with a “Downstream Distillation Setup” subsection:
   - dataset, offline encoding, bypass patch, training details
3. Update `results_and_analysis.tex` with tokenizer-first framing:
   - **STS:** headline table + chart, explicit “MFT vs Tabi” delta, and random-init sanity check.
   - **MTEB-TR:** overall average + category averages + short category takeaways (do not over-focus on backbone differences).
   - **Version benchmark:** 3–5 sentence robustness summary; move detailed version history table(s) to Appendix if needed.
4. Refactor `introduction.tex` and `methodology.tex` per Section 6.1 (move results out of Intro; add algorithm box; add subsections).
5. Add Leipzig gloss formatting and update examples throughout (Section 6.2).
4. Update `tokenizer.bib`:
   - merge entries from `papers/papers.bib`
   - ensure every new citation key is referenced at least once
5. Ensure `main.tex` compiles cleanly and references resolve.

### Phase C — Reproducibility & polish
1. Confirm all reported numbers match their source artifacts (`*.md` / `*.json` / `results/**`).
2. Add a short “Reproducibility” paragraph (commands + artifact locations).
3. Run a final LaTeX build check; fix broken references/figures.

Optional (only if it improves presentation):
- Re-run `generate_sts_tables.py` / `generate_mteb_report.py` to regenerate cleaner tables.
- Add a small helper script to emit LaTeX tables directly from `sts_benchmark_results.json` and `results/**` so paper tables stay consistent with raw results.

---

## 8) Acceptance checklist (quick review criteria)

- All four `papers/*.pdf` are (a) cited in BibTeX and (b) explicitly mentioned in the paper text.
- Downstream experiments are described with enough detail to reproduce (inputs, code entrypoints, hyperparams, metrics).
- Figures/tables compile and are referenced in text.
- No dangling citations; bibliography compiles.
- Claims in the paper are supported by either:
  - cited literature, or
  - results artifacts in this repo.

---

## 9) Quick confirmations (so I implement the exact narrative you want)

1. OK to phrase the key claim as: **“Under the same downstream training budget, Tabi does not catch MFT on STS/retrieval-focused quality”** (and only discuss “speed/adaptation” if we add a microbenchmark or strong citations)?
2. For MTEB, do you prefer the headline metric to be **overall average** (simple) or a **retrieval+STS subset average** (more aligned with your objective)?
3. Should detailed version history go to the **Appendix by default** (recommended), with only a short robustness summary in the main Results?
4. OK to update the **title** to include Turkish/agglutinative typology (as suggested by Reviewer jLKM)?
