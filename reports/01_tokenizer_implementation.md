# Tokenizer Implementation Report

## Overview

The `TurkishTokenizer` is a hybrid morphological tokenizer designed to improve tokenization for agglutinative languages like Turkish. It combines:

1.  **Rule-based Morphological Segmentation:** Identifies roots and suffixes using curated dictionaries.
2.  **Byte Pair Encoding (BPE) Fallback:** Handles unknown stems or foreign words to ensure open-vocabulary coverage.
3.  **Phonologically Aware Decoding:** Reconstructs surface forms by applying Turkish vowel harmony and consonant assimilation rules.

**Key Files:**

- `turkish_tokenizer.py`: Main tokenizer class and pipeline logic.
- `turkish_decoder.py`: Re-synthesis rules for decoding.
- `kokler.json` (Roots), `ekler.json` (Suffixes), `bpe_tokenler.json` (BPE vocabulary).

## Pipeline Steps

### 1. Pre-processing

- **CamelCase Split:** Splits CamelCase words (e.g., `HTTPServer` -> `HTTP Server`) to handle code-switching or technical terms better.
- **Uppercasing Handling:** Identifies uppercase letters and inserts a special `<uppercase>` token before them during tokenization to preserve casing information without separate vocab entries for every case variant.

### 2. Tokenization Logic (`_tokenize_word`)

The tokenizer processes each word using a **Longest-Prefix Matching** strategy with prioritized dictionaries:

1.  **Root Lookup:** effectively "greediest" match against `kokler.json`.
2.  **Suffix Lookup:** If no root match, checks `ekler.json`.
3.  **BPE Lookup:** If no morphological match, falls back to `bpe_tokenler.json`.
4.  **Unknown:** If all fail, marks as `<unknown>`.

**Token Types:**

- `ROOT`: Base stems (e.g., "ev").
- `SUFFIX`: Grammatical affixes (e.g., "ler", "den").
- `BPE`: Subword units for coverage.

### 3. Post-processing

- **Clean-up:** Logic to remove spurious space markers or invalid sequences (e.g., around uppercase markers) to ensure cleaner token streams.

## Decoding Logic (`TurkishDecoder`)

Decoding is not just concatenation. It requires **morpheme synthesis**:

- **Vowel Harmony:** Suffixes have multiple surface forms (e.g., `-ler/-lar`). The decoder chooses the correct form based on the _previous token's_ last vowel (e.g., "ev" -> "-ler", "okul" -> "-lar").
- **Consonant Assimilation:** Handles hard/soft consonant transitions (e.g., "kitap" + "da" -> "kitapta").
- **Recursive Checks:** `_ends_with_ince`, `_ends_with_sert_unsuz` etc. check the phonological properties of the preceding context.

## Paper Integration Points

- **Methodology Section:**
  - Detailed subsections for "Dictionary Construction", "Encoding Algorithm", "BPE Fallback", and "Decoding".
  - Pseudocode for the `_tokenize_word` greedy matching loop.
  - Example of the camel-case splitting + uppercase token strategy (Addressing "Edge Cases").
