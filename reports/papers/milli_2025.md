# miLLi (Rahimov, 2025) - Notes

Source: `papers/Rahimov - 2025 - miLLi Model ... .pdf` (also in `papers/papers.bib` as `rahimov_milli_2025`).

## What it is

miLLi 1.0 is a hybrid tokenizer for Azerbaijani combining:

- a **root dictionary**,
- **BPE** fallback,
- and a **phonological restoration** mechanism that maps allomorphic surface forms back to canonical roots.

## Key ideas to cite in our paper

- For agglutinative languages, purely statistical tokenization may treat allomorphs as unrelated units, harming morphological generalization.
- A restoration/normalization layer can improve root consistency without abandoning statistical coverage.

## How it relates to our work

- Very aligned with our **normalization + unified identifiers** principle (e.g., surface variants → shared ID).
- Helpful support for the “language-agnostic framework + language-specific resources” framing:
  - miLLi explicitly depends on language-specific lexicons (Azerbaijani), similar to our root/affix resources.

## Paper insertion points

- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/related_work.tex`: cite as another dictionary+BPE hybrid for a Turkic/agglutinative language (Azerbaijani).
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/methodology.tex`: use it to motivate normalization/restoration design choices and limitations.
