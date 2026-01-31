# Reviewer Response: Greedy Longest-Prefix Matching Analysis

## Overview

The MFT tokenizer uses a **greedy longest-prefix matching** strategy implemented in [turkish_tokenizer.py](file:///Users/alibayram/Desktop/mft-downstream-task/turkish_tokenizer.py). This document analyzes failure modes and potential disambiguation strategies.

---

## Algorithm Description

From the source code, the matching order is:

1. **Roots dictionary** (longest prefix first)
2. **Suffixes dictionary** (longest prefix first)
3. **BPE fallback** (longest prefix first)
4. **Unknown token** (single character)

```python
# From _tokenize_word() in turkish_tokenizer.py
rid, rtok = self._longest_prefix_lookup(substr, self.roots, self.max_root_len)
if rid is not None:
    result.append({"token": rtok, "id": rid, "type": TokenType.ROOT})
    pos += len(rtok)
    continue

sid, stok = self._longest_prefix_lookup(substr, self.suffixes, self.max_suffix_len)
# ... continues with BPE fallback
```

---

## Known Failure Modes

### 1. Morphologically Ambiguous Segmentations

**Example:** The word "alarm" could be parsed as:

- `al + ar + m` (root + aorist + 1st person) - grammatically invalid
- `alarm` (foreign loanword root) - correct

The greedy approach may prefer the longer morphological parse even when incorrect.

### 2. Derivational Chain Ambiguity

**Example:** "güzelleştirmek" (to beautify):

- Correct: `güzel + leş + tir + mek`
- Possible greedy error: if "güzelle" existed as a root

### 3. Compound Word Boundaries

**Example:** "evkadını" (housewife):

- Could be: `ev + kadın + ı` (house + woman + possessive)
- Or: `evkadın + ı` (if "evkadın" is in compound dictionary)

---

## Mitigation Strategies in Current Implementation

### 1. Explicit Compound Dictionary

The root dictionary includes common compounds as single entries:

- `akarsu` (stream = flowing + water)
- `çamaşırhane` (laundromat)

### 2. Priority Ordering

Roots are checked before suffixes, preventing suffix-only parses.

### 3. BPE Fallback

Unknown or ambiguous forms fall back to statistical BPE, avoiding forced morphological analysis.

---

## Quantitative Analysis

From vocabulary files:

- **Max root length:** computed dynamically
- **Max suffix length:** computed dynamically
- **Special handling:** CamelCase splitting via `_camel_split_with_positions()`

---

## Potential Improvements (Not Yet Implemented)

1. **Scoring-based disambiguation:** Assign probabilities to parses based on corpus frequency
2. **Morphological constraint checking:** Validate that suffix sequences are grammatically valid
3. **Context-aware selection:** Use surrounding tokens to resolve ambiguity
4. **User-adjustable fallback threshold:** Allow tuning the greedy vs. BPE tradeoff

---

## Conclusion

The greedy strategy is computationally efficient (O(n) per word) but can produce linguistically implausible segmentations for ambiguous cases. The current design prioritizes:

- **Coverage** (via BPE fallback)
- **Consistency** (same word always gets same tokenization)
- **Speed** (no backtracking or beam search)

For a full journal-ready paper, adding a failure analysis on annotated morphological data would quantify these edge cases.
