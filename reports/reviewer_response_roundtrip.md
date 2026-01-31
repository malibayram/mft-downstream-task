# Reviewer Response: Round-Trip Reconstruction Evaluation

## Executive Summary

The decoder is **not yet lossless** in all cases. Current testing reveals issues with vowel harmony rules in suffix selection. This is documented transparently for the reviewers.

---

## Test Methodology

We tested round-trip reconstruction on sample sentences:

```
Original → encode() → decode() → Reconstructed
```

### Test Results

| Original                                | Decoded                                 | Match |
| --------------------------------------- | --------------------------------------- | ----- |
| Kalktığımızda hep birlikte yürüdük.     | kalkdüğımüzde hep birlikte yürüdik.     | ✗     |
| İstanbul Türkiye'nin en büyük şehridir. | istanbul türkiye'nün en büyük şehridir. | ✗     |
| Kitaplarımızdan bazılarını okudum.      | kitaplarımızdan bazılarını okudım.      | ✗     |
| Öğretmenlerimize teşekkür ettik.        | öğretmenlarımıze teşekkür ettik.        | ✗     |
| Çocuklar parkta oynuyor.                | çocuklar parkda oynuyor.                | ✗     |

**Exact-match accuracy: 0%**

---

## Error Categories

### 1. Vowel Harmony Errors (Most Common)

The decoder's vowel harmony rules don't always select the correct allomorph:

- `yürüdük` → `yürüdik` (wrong vowel in past tense)
- `okudum` → `okudım` (wrong vowel in 1st person singular)
- `nin` → `nün` (wrong genitive allomorph)

### 2. Consonant Alternation Errors

Hard-soft consonant rules not fully implemented:

- `parkta` → `parkda` (should be `ta` after `k`, a hard consonant)
- `Kalkt` → `kalkd` (similar issue)

### 3. Capitalization Normalization

Uppercase is tracked but reconstructed as lowercase:

- `İstanbul` → `istanbul`
- `Türkiye` → `türkiye`

---

## Root Cause Analysis

The decoder in [turkish_decoder.py](file:///Users/alibayram/Desktop/mft-downstream-task/turkish_decoder.py) implements vowel harmony rules but has edge cases:

1. **`_ends_with_ince()`** function doesn't handle all vowel patterns
2. **`_ends_with_sert_unsuz()`** is applied but consonant context tracking is incomplete
3. **Capitalization** is stored via `<uppercase>` token but not restored in reconstruction

---

## Implications for the Paper

The paper claims the decoder enables "lossless reconstruction." **This claim should be softened** to:

- "Near-lossless reconstruction for most common morphological patterns"
- Or include quantitative accuracy metrics

---

## Recommended Fixes

1. **Improve vowel harmony rules** in `_select_correct_suffix()`
2. **Add more test cases** for consonant alternations
3. **Restore capitalization** during decoding
4. **Report reconstruction accuracy** on a held-out corpus

---

## Quantitative Metrics (From Testing)

| Metric                     | Value                   |
| -------------------------- | ----------------------- |
| Test sentences             | 5                       |
| Exact match rate           | 0%                      |
| Character-level similarity | ~90% (estimated)        |
| Main error type            | Vowel harmony selection |
