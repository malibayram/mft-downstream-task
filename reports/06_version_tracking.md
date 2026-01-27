# Version Tracking & Robustness Report

## Overview

We tracked the performance of models across multiple code revisions to ensure stability and reproducibility.

**Source of Truth:** `VERSION_BENCHMARK_RESULTS.md`

## Key Findings

- **Stability:** The top models (`mft-downstream-task-embeddingmagibu`) consistently score in the **75-76%** range on STS Pearson across multiple recent revisions.
- **No Flukes:** The improvement over baselines is not a one-off random seed luck; it is sustained across iterations.
- **Latest Best:** Revision `7d2932fe` achieved **76.10%** Pearson.

## Paper Integration Points

- **Results Section (or Appendix):**
  - Add a brief "Robustness" statement: "We tracked performance across X experiment runs and code revisions. The reported improvements are stable, with standard deviations < 1.0%."
  - (Optional) If reviewers asked for error bars, this data supports that our numbers are reliable.
