# Experiment: Prototype vs Baseline — Measurable Comparison

Headline numbers below are from the **current detector** (rating-drop +
keyword checks; semantic similarity **off**) run with the real
`all-MiniLM-L6-v2` stack available but unused. Ground truth is the
`flagged_mismatch_ground_truth` column injected by
`generate_synthetic_data.py` (~15% of sessions). That column is not shown
to the counsellor or the detector.

**Current prototype vs attendance-only baseline**

| | Baseline | Prototype (current) |
|---|---|---|
| What it can detect | Attendance count only | Rating-drop + keyword mismatch |
| Precision | n/a | **0.17** |
| Recall | 0 (36 mismatches invisible) | **0.47** (17 / 36) |
| F1 | n/a | **0.25** |
| Confusion | n/a | TP 17, FP 85, FN 19, TN 172 |

The prototype clears the only bar the baseline sets (detecting *anything*),
but 0.25 F1 is weak in absolute terms because of false positives. Severity
tiers exist so a counsellor can look at high-priority flags first rather
than treating all 85 extra flags as equal. Reproduce with
`experiments/experiment.py`.

How we got here is in the iteration history below. Each round is left in
place as an evidence trail, not as competing "current" numbers.

---

## Method

Dataset: 293 sessions, 40 clients, 36 ground-truth mismatches.

Two systems are scored against that ground truth:

1. **Baseline** — attendance-only (`baseline/attendance_baseline.py`)
2. **Prototype** — `prototype/mismatch_detector.py`

The semantic layer was tried two ways: a TF-IDF proxy (this sandbox had
no model download) and the real MiniLM model locally.

---

## Iteration history

### Round 1 — TF-IDF proxy (no embedding model)

Lexical overlap only. Used to get *some* number in an offline sandbox.

| Metric | Value |
|---|---|
| True positives | 12 / 36 |
| False positives | 54 |
| Precision | 0.18 |
| Recall | 0.33 |
| F1 | 0.24 |

TF-IDF treats "different words" as "contradiction," which inflates false
positives. At the time this was taken as an argument *for* real sentence
embeddings. The next two rounds tested that belief.

A separate structural gap was already known from the patient-journey
walkthrough (client C027): a sharp rating drop with empty barrier text
could not be caught by any text-vs-text comparison.

### Round 2 — Rating-drop check added (Review-1)

`_rating_drop_mismatch` compares the current rating to the client's own
last-three-session average. Independent of barrier text.

| Metric | Round 1 | Round 2 |
|---|---|---|
| True positives | 12 | 17 |
| False positives | 54 | 85 |
| Precision | 0.18 | 0.17 |
| Recall | 0.33 | **0.47** |
| F1 | 0.24 | 0.25 |

Five extra true mismatches were caught. Precision dipped slightly. That
trade-off is why severity tiers were added: more flags are only usable if
they can be triaged.

These Round 2 numbers came from the TF-IDF proxy plus the new drop check.
The automated tests in `tests/test_mismatch_detector.py` cover the drop
logic without depending on either TF-IDF or MiniLM.

### Round 3 — Real MiniLM, then semantic check turned off

`experiments/experiment.py` and `experiments/threshold_sweep.py` were run
with `all-MiniLM-L6-v2`. Including the semantic similarity check **hurt**
F1 at every threshold tried (about **0.12–0.16** vs **0.25** without it).

Cause: short, topically different phrases score as "dissimilar" even when
there is no emotional contradiction ("feeling better this week" vs
"missed a session due to a deadline"). The check conflated *different
subject* with *contradicts what was said*.

**Decision recorded in code:** `USE_SEMANTIC_CHECK = False` in
`mismatch_detector.py`. The function is kept, not deleted, so the
negative result stays inspectable. A polarity-based (not topical)
semantic check is future work.

With the semantic check off, the real-model run matches Round 2:

| Metric | Semantic check on (real MiniLM) | Semantic check off (current) |
|---|---|---|
| Precision | lower (see sweep) | **0.17** |
| Recall | mixed / not better | **0.47** |
| F1 | 0.12–0.16 | **0.25** |

`experiments/experiment_results_summary.md` is the machine-written
snapshot of this current configuration.

---

## What the baseline comparison does and does not claim

- Attendance-only tracking cannot represent a "mismatch" at all. Any
  recall above 0% beats it.
- 0.47 recall / 0.17 precision is not a clinical-grade detector. It is a
  prototype that flags 17 of 36 planted contradictions and over-flags
  85 sessions. That is disclosed, not dressed up.
- Stakeholder reaction to this false-positive load is **not** inferred
  here. Real counsellor feedback will be pasted in separately when
  collected (`docs/stakeholder-validation.md` is still a simulated
  walkthrough).
