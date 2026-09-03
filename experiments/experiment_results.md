# Experiment: Prototype vs Baseline — Measurable Comparison

## Method

The synthetic data generator (`generate_synthetic_data.py`) deliberately
injects mismatches into ~15% of sessions (self-reported rating and
text/barrier description contradicting each other) and records this in a
`flagged_mismatch_ground_truth` column that is **not shown to the
counsellor or the detector** — it exists only for evaluation.

We compare two systems against this ground truth:
1. **Baseline** — the current attendance-only approach
2. **Prototype** — the goal-based mismatch detector

Two versions of the prototype's semantic layer were tested:
- **TF-IDF proxy** — run in this sandbox (no internet access to download
  the real embedding model), included here for transparency about method
- **Real sentence-transformers (all-MiniLM-L6-v2)** — the actual prototype,
  designed to be run locally (`experiments/experiment.py`) per
  `prototype/README.md`, since it needs a one-time model download

---

## Baseline: Attendance-only tracker

| Metric | Value |
|---|---|
| Mismatches it can detect | **0** — the concept of a "mismatch" doesn't exist in this model; it only counts sessions attended |
| Precision / Recall / F1 | Not applicable — no prediction is ever made |

**Target for the prototype:** meaningfully outperform "detects nothing,"
i.e. recall > 0% on the 36 ground-truth mismatch cases in the dataset
(293 total sessions, 40 clients).

---

## Measured result: Prototype (TF-IDF proxy, run in this sandbox)

| Metric | Value |
|---|---|
| True positives | 12 / 36 |
| False positives | 54 |
| False negatives | 24 |
| True negatives | 203 |
| Precision | 0.18 |
| Recall | 0.33 |
| F1 score | 0.24 |
| Accuracy | 0.73 |

**Baseline vs Target vs Measured:**
- Baseline: 0 mismatches detectable, by design
- Target: recall > 0%
- Measured: recall = 33% (12 of 36 real mismatches caught), at the cost of
  54 false positives

This already clears the bar the baseline sets (zero), but 0.24 F1 is weak
in absolute terms — largely due to precision, not recall.

---

## Error analysis

**Why precision is low (many false positives):** TF-IDF measures literal
word overlap, not meaning. Two sentences that mean similar things but share
no vocabulary ("felt worse than before" vs. "roommate conflict added new
stress") score near-zero similarity even when there's no real contradiction
between them — because TF-IDF has no way to know they're both about
distress. This pushes many *non-mismatched* sessions below the similarity
threshold, inflating false positives.

**Why recall is moderate, not high:** Some genuine mismatches were caught
via the independent rating/keyword rule (high rating + a negative barrier
keyword), which doesn't depend on TF-IDF at all — this is the more reliable
half of the detector in this proxy run.

**What this tells us about the design choice:** This result is itself
evidence *for* using sentence-transformers rather than a cheaper lexical
method in the real prototype. Sentence embeddings are trained to capture
semantic meaning, not just shared vocabulary, so they should substantially
reduce the false-positive problem seen here — two differently-worded but
semantically-aligned statements would score *high* similarity under a real
embedding model, whereas TF-IDF scores them near zero regardless of
meaning. This is exactly the gap between "detects contradictions" and
"detects different word choices," and it's the reason the project's
technical design specifies sentence-transformers, not simpler keyword or
lexical matching.

**Known gap carried over from Step 6:** Neither version of the detector
reliably catches the "rating-drop with empty barrier text" pattern found
during the patient-journey walkthrough (`docs/patient-journeys.md`,
`docs/failure-mode-analysis.md`) — this is a structural gap, not something
the choice of similarity method fixes, and remains the top priority for a
future iteration.

---

## Next step for final numbers

Run `experiments/experiment.py` locally (after following
`prototype/README.md` setup) to get the real sentence-transformers-based
precision/recall/F1 numbers for final submission. Expected outcome, based
on the reasoning above: similar or better recall, and meaningfully higher
precision than the TF-IDF proxy shown here, because real embeddings won't
penalize semantically-related sentences that happen to use different words.
