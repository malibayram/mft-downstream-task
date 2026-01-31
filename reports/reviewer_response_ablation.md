# Reviewer Response: Ablation Analysis - Component Contributions

## Overview

This document analyzes the contributions of individual components in the MFT tokenizer by examining the vocabulary structure and token ID allocation.

---

## Vocabulary Breakdown by Component

From the JSON configuration files:

| Component           | Token Count                       | ID Range      | Percentage |
| ------------------- | --------------------------------- | ------------- | ---------- |
| **Root Dictionary** | 20,000 unique IDs                 | 0–19,999      | 61.0%      |
| **Affix System**    | 72 unique IDs → 177 surface forms | 20,000–20,071 | 0.2%       |
| **BPE Fallback**    | 12,696 IDs                        | 20,072–32,767 | 38.8%      |
| **Total**           | **32,768**                        | 0–32,767      | 100%       |

---

## Component Analysis

### 1. Root Dictionary (IDs 0–19,999)

**Purpose:** Cover Turkish lexical stems with morphological variants

**Key features:**

- Includes 22,231 entries mapping to 20,000 unique IDs
- Multiple surface forms map to same ID (phonological normalization)
- Contains special tokens: `<uppercase>` (ID 0), `<unknown>` (ID 1), space (ID 2), `<pad>` (ID 5), `<eos>` (ID 6)

**Example normalizations:**

- `kitap` / `kitab` → same root ID (p→b alternation)
- `git` / `gid` → same root ID (t→d voicing)

### 2. Affix System (IDs 20,000–20,071)

**Purpose:** Morpheme-aligned suffix tokenization with allomorph normalization

**Structure:**

- 72 unique affix IDs covering 177 surface allomorphs
- Vowel harmony variants share IDs (e.g., `-lar/-ler` → ID 20,000)
- Consonant alternation variants share IDs (e.g., `-da/-de/-ta/-te` → ID 20,024)

**Example mappings from ekler.json:**

```
"lar": 20000, "ler": 20000           # Plural
"da": 20024, "de": 20024, "ta": 20024, "te": 20024  # Locative
"mış": 20016, "miş": 20016, "muş": 20016, "müş": 20016  # Evidential past
```

### 3. BPE Fallback (IDs 20,072–32,767)

**Purpose:** Handle OOV words, foreign terms, neologisms

**Statistics:**

- 12,696 unique BPE tokens
- Trained on corpus with morphological segments masked
- Covers residual stems and subwords not in root/affix dictionaries

---

## Estimated Contribution to Coverage

Based on typical Turkish text:

| Component | Estimated Token Usage |
| --------- | --------------------- |
| Roots     | ~60-70% of tokens     |
| Affixes   | ~20-25% of tokens     |
| BPE       | ~10-15% of tokens     |

---

## Ablation Scenarios (Hypothetical)

### Without Phonological Normalization

- Vocabulary would grow from 20,000 → 22,231+ root entries
- Allomorphic variants treated as separate tokens
- Expected impact: Lower TR% due to fragmented representations

### Without Affix Merging

- 177 affix entries instead of 72
- Each allomorph has unique ID
- Expected impact: More vocabulary entries, possibly similar downstream performance

### Without BPE Fallback

- OOV words produce `<unknown>` tokens
- Expected impact: Significant coverage loss on foreign words, neologisms

### Without Uppercase Token

- Case variants would need separate vocabulary entries OR lose case info
- Expected impact: 2x vocabulary inflation or information loss

---

## Recommendation for Paper

The paper could add a table showing vocabulary composition:

> **Table X: MFT Vocabulary Composition**
> | Component | IDs | Purpose |
> |-----------|-----|---------|
> | Roots (normalized) | 20,000 | Turkish lexical stems with phonological variants |
> | Affixes (merged) | 72 | Grammatical suffixes with allomorph normalization |
> | BPE subwords | 12,696 | Open-vocabulary fallback for OOV coverage |
> | **Total** | **32,768** | |

This clarifies the numerical inconsistencies reviewers noted.
