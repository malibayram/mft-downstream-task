# MorphBPE (Asgari et al., 2025) — Notes

Source: `papers/Asgari et al. - 2025 - MorphBPE ... .pdf` (also in `papers/papers.bib` as `asgari_morphbpe_2025`).

## What it is
MorphBPE proposes a **morphology-aware extension of BPE** that incorporates morphological structure while staying compatible with standard LLM pipelines.

## Key ideas to cite in our paper
- Vanilla BPE often violates morpheme boundaries, especially for morphologically rich languages.
- MorphBPE introduces **morphology-aware evaluation metrics** (e.g., morpheme/token alignment and consistency-style measures).
- Reported benefits include lower training loss and faster convergence in their experiments across multiple languages (English, Russian, Hungarian, Arabic).

## How it relates to our work
- Strong “closest neighbor” in spirit: both aim to keep subword segmentation **linguistically meaningful**.
- Useful for positioning our approach in Related Work:
  - MorphBPE modifies the BPE merging procedure.
  - Our approach uses **dictionary-driven segmentation + normalization** with a BPE fallback stage.

## Paper insertion points
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/related_work.tex`: add contrast between “BPE-with-constraints” (MorphBPE) vs “morphology-first pipeline + fallback”.
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/methodology.tex`: cite as a similar hybridization goal and motivate why a linguistic layer can be beneficial.

