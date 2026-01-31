# Reviewer Response: Computational Efficiency Analysis

## Summary

This document provides tokens-per-word, sequence length, and efficiency metrics for the MFT tokenizer compared to baseline considerations.

---

## Tokens-per-Word Analysis

### Test Results on Sample Sentences

| Sentence                                              | Words | Tokens | Tok/Word |
| ----------------------------------------------------- | ----- | ------ | -------- |
| Kalktığımızda hep birlikte yürüdük.                   | 4     | 13     | 3.25     |
| İstanbul Türkiye'nin en büyük şehridir.               | 5     | 12     | 2.40     |
| Kitaplarımızdan bazılarını okudum.                    | 3     | 14     | 4.67     |
| Öğretmenlerimize teşekkür ettik.                      | 3     | 10     | 3.33     |
| Çocuklar parkta oynuyor.                              | 3     | 8      | 2.67     |
| Bilgisayarlarımızdaki programlar çalışmıyor.          | 3     | 14     | 4.67     |
| Üniversitedeki öğrenciler sınava hazırlanıyor.        | 4     | 12     | 3.00     |
| Teknolojinin gelişmesiyle birlikte hayatımız değişti. | 5     | 13     | 2.60     |

### Aggregate Statistics

| Metric                           | Value     |
| -------------------------------- | --------- |
| **Total words**                  | 30        |
| **Total tokens**                 | 96        |
| **Average tokens-per-word**      | **3.20**  |
| **Average tokens-per-character** | **0.313** |

---

## Vocabulary Composition

| Component    | Entries           | Unique IDs | ID Range      |
| ------------ | ----------------- | ---------- | ------------- |
| **Roots**    | 22,231            | 20,000     | 0–19,999      |
| **Suffixes** | 177 surface forms | 72         | 20,000–20,071 |
| **BPE**      | 12,696            | 12,696     | 20,072–32,767 |
| **Total**    | —                 | **32,768** | —             |

### Special Tokens

| Token         | ID  | Purpose                  |
| ------------- | --- | ------------------------ |
| `<uppercase>` | 0   | Case preservation marker |
| `<unknown>`   | 1   | OOV fallback             |
| ` ` (space)   | 2   | Whitespace token         |
| `<pad>`       | 5   | Padding                  |
| `<eos>`       | 6   | End of sequence          |

---

## Efficiency Considerations

### Token Count Trade-off

MFT produces **more tokens** than some baselines because it:

1. Explicitly segments affixes (e.g., `kitap + lar + ımız + dan` = 4 tokens)
2. Uses dedicated tokens for case (`<uppercase>`)
3. Preserves morpheme boundaries over compression

### Computational Implications

| Factor               | Impact                                             |
| -------------------- | -------------------------------------------------- |
| **Attention cost**   | Higher token count → O(n²) attention scales worse  |
| **Semantic quality** | Morpheme alignment → better representation reuse   |
| **Vocabulary size**  | 32,768 = standard transformer embedding table size |

### Comparison Context

While sequence length increases, this is offset by:

- Better sample efficiency (stable root representations)
- Improved semantic similarity scores (as shown in STS results)
- More interpretable tokenization for analysis

---

## Training Efficiency Notes

From the training configuration:

- **Architecture:** google/embeddinggemma-300m
- **Batch size:** configured per training script
- **Epochs:** embedding distillation with two-phase training
- **Memory:** Standard for 300M parameter model

---

## Recommendation for Paper

Add a sentence in the Discussion/Trade-offs section:

> "The proposed tokenizer produces on average 3.2 tokens per word, compared to [baseline X] at [Y] tokens per word. While this increases sequence length, the gains in TR% and downstream STS correlation suggest favorable sample efficiency."
