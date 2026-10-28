# Collaborative Outcome Tracker

A Streamlit prototype for a university counselling centre. It replaces
attendance-only outcome tracking with **client-defined goal tracking**, and
flags sessions where a client's rating or wording does not line up with
the rest of their record — for a counsellor to review, never as an
automatic clinical conclusion.

Stack: Python, Streamlit, pandas, Altair, optional sentence-transformers,
optional Ollama (local SLM).

**Contents:** [Detector](#current-detector-what-actually-flags) ·
[Results](#measured-comparison-vs-attendance-baseline) ·
[Quick start](#quick-start) · [Data schema](#data-schema) ·
[Interfaces](#interfaces) · [Configuration](#configuration) ·
[Tests & error boundaries](#tests--error-boundaries) ·
[Known limitations](#known-limitations) · [Layout](#repository-layout)

## Current detector (what actually flags)

Two rules, both inspectable:

1. **Rating-drop** — current self-reported rating is ≥ 3 points below the
   client's own last-three-session average (independent of barrier text).
2. **Keyword** — rating ≥ 7 paired with negative barrier language
   (`relapse`, `conflict`, …).

A **semantic similarity check** (sentence-transformers `all-MiniLM-L6-v2`)
is implemented but **off by default** (`USE_SEMANTIC_CHECK = False` in
`prototype/mismatch_detector.py`). A real-model evaluation found it hurt
precision: F1 dropped from 0.25 to 0.12–0.16. See
`experiments/experiment_results.md`.

Flags are tiered **low / medium / high** so a busy caseload can be
triaged:

| Tier | When |
|---|---|
| `none` | no rule fired |
| `low` | keyword rule only (crude substring match) |
| `medium` | rating-drop of 3 up to (not including) 5 as the only signal |
| `high` | two rules agree, or a rating-drop of 5 or more |

Ollama only writes a one-sentence explanation of an already-made flag; if
Ollama is not running (Streamlit Cloud), explanations are skipped after the
first failed contact. It can never change whether or how a session is flagged.

## Measured comparison vs attendance baseline

293 synthetic sessions, 36 planted mismatches.

| | Attendance baseline | Current prototype |
|---|---|---|
| Precision | n/a | 0.17 |
| Recall | 0 | 0.47 |
| F1 | n/a | 0.25 |

Reproduce: from inside `experiments/`, run `python3 experiment.py`.
Read these numbers together with [Known limitations](#known-limitations) —
the precision figure is affected by how the synthetic labels were generated.

## Quick start

```bash
python3 generate_synthetic_data.py      # from the repo root; seeded, reproducible
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

Full local + Ollama notes: `prototype/README.md`.

## Data schema

There is **no database server**. The prototype's data store is three CSV
files in `synthetic_data/`, written by `generate_synthetic_data.py` (seeded
with 42, so every run produces identical files). They stand in for the
centre's record system. All names are synthetic (first name + last initial).

```
clients (40 rows) 1 ──< goals (53 rows) 1 ──< sessions (293 rows)
```

A client has 1–2 goals (70% / 30%); a goal has 3–8 sessions, 6–10 days apart.
`sessions` carries `client_id` as well as `goal_id` for convenience; it is
derivable through `goals`.

### `clients.csv`

| Column | Type | Description |
|---|---|---|
| `client_id` | string, PK | `C001` … `C040` |
| `client_name` | string | Synthetic, e.g. `Danielle J.` |
| `age` | int | 18–26 |
| `urgency_level` | enum | `low` (~50%), `moderate` (~35%), `high` (~15%) |
| `presenting_concern` | string | e.g. `sleep difficulties`, `exam anxiety` |
| `intake_date` | date `YYYY-MM-DD` | 2026-01-06 … 2026-03-01 |

### `goals.csv`

| Column | Type | Description |
|---|---|---|
| `goal_id` | string, PK | `G0001` … |
| `client_id` | string, FK → clients | |
| `goal_text` | string | The client-defined goal, e.g. `Manage panic attacks` |
| `baseline` | string | Where the client starts, in their terms |
| `target` | string | What success looks like |
| `created_date` | date | 1–5 days after intake |
| `target_date` | date | created + 6, 8, 10 or 12 weeks |
| `status` | enum | `active`, `achieved`, `needs revision`, `pending`. Derived: mean of the last 3 ratings ≥ 8 → `achieved`; ≤ 4 → `needs revision`; otherwise `active`; `pending` if no sessions |

### `sessions.csv`

| Column | Type | Description |
|---|---|---|
| `session_id` | string, PK | `S00001` … |
| `goal_id` | string, FK → goals | |
| `client_id` | string, FK → clients | |
| `session_date` | date | |
| `self_reported_rating` | int | 1–10, **higher is better** (generator emits 2–9) |
| `self_reported_text` | string | The client's own words about the week |
| `barriers` | string | What got in the way; `no barriers this session` if none |
| `agreed_adjustment` | string | What counsellor and client agreed to change |
| `flagged_mismatch_ground_truth` | bool | **Evaluation only.** Marks the ~12% of sessions where a mismatch was deliberately injected. Never read by the detector or shown in the UI; only `experiments/` uses it |

Joins used by the app: `goals.client_id → clients.client_id` and
`sessions.goal_id → goals.goal_id`. There is no schema validation on load; a
missing or renamed column surfaces as a pandas `KeyError`.

## Interfaces

This prototype has **no REST/HTTP API**. Its interfaces are:

### 1. Python interface (the decision layer)

`prototype/mismatch_detector.py` is the only module that app and experiments
call.

```python
from mismatch_detector import MismatchDetector

detector = MismatchDetector(use_ollama=False)
result = detector.check_session(
    {
        "session_id": "S00003",
        "self_reported_rating": 8,
        "self_reported_text": "feeling much better this week",
        "barriers": "relapse into old coping habits under stress",
    },
    rating_history=[4, 9],   # prior ratings for this goal, oldest first
)
```

Returns a `MismatchResult` (real output for the call above):

```json
{
  "session_id": "S00003",
  "is_flagged": true,
  "severity": "low",
  "similarity_score": 1.0,
  "reasons": ["high numeric rating paired with negative barrier language"],
  "explanation": null
}
```

| Field | Meaning |
|---|---|
| `is_flagged` | `True` iff at least one rule fired |
| `severity` | `none` / `low` / `medium` / `high` |
| `similarity_score` | A placeholder `1.0` while the semantic check is off — do not read it as a measurement |
| `reasons` | Every rule that fired, in plain words |
| `explanation` | Ollama's sentence, a fallback note if Ollama is unreachable, or `null` (not flagged / Ollama disabled) |

Input contract and failure behaviour (what is tolerated, what raises) are in
the `check_session` docstring and in
[`docs/testing-and-error-boundaries.md`](docs/testing-and-error-boundaries.md).
In short: blank text/barrier cells are tolerated; a missing key or a
non-numeric rating raises, on purpose.

### 2. Outbound HTTP call (optional, local only)

The only network call in the codebase, made only when `use_ollama=True`:

```
POST http://localhost:11434/api/generate
{"model": "llama3.2:1b", "prompt": "<flag evidence>", "stream": false}
→ reads the "response" field; 5 s timeout
```

Any failure (not running, timeout, HTTP error, malformed JSON) degrades to a
fallback note and stops further attempts for that process. Nothing leaves the
machine.

### 3. User interface

`cd prototype && streamlit run app.py` — pick a client, see their goals
(baseline, target, status), a rating chart, flagged sessions ranked by
severity, and the full session log. A *Show flags* filter narrows the list
by severity, and an Ollama checkbox turns the explanation step on or off.

### 4. Command-line scripts

| Script (run from) | Purpose | Writes |
|---|---|---|
| `generate_synthetic_data.py` (repo root) | Generate the three CSVs | `synthetic_data/*.csv` |
| `baseline/attendance_baseline.py` (in `baseline/`) | Attendance-only comparison | `baseline_report.csv` |
| `experiments/experiment.py` (in `experiments/`) | Detector vs ground truth | `experiment_results_detailed.csv`, `experiment_results_summary.md` |
| `experiments/threshold_sweep.py` (in `experiments/`) | Semantic-threshold sweep (needs sentence-transformers) | prints a table |
| `experiments/proxy_experiment*.py` | Early TF-IDF proxy runs (scikit-learn), kept as history | prints metrics |

## Configuration

All constants are module-level in `prototype/mismatch_detector.py` (tests
patch them; experiments read them).

| Constant | Value | Role |
|---|---|---|
| `HIGH_RATING_THRESHOLD` | 7 | Keyword rule fires only at or above this rating |
| `NEGATIVE_BARRIER_KEYWORDS` | `relapse, reluctance, struggl, disrupted, conflict, overwhelm` | Lowercase substring match; English only |
| `RATING_DROP_THRESHOLD` | 3 | Drop below the rolling average that flags |
| `ROLLING_WINDOW` | 3 | Prior sessions averaged for the baseline |
| `LARGE_DROP_THRESHOLD` | 5 | A drop this big is `high` on its own |
| `USE_SEMANTIC_CHECK` | `False` | Optional similarity rule (see above) |
| `MISMATCH_SIMILARITY_THRESHOLD` | 0.1 | Only used if the semantic check is on |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `localhost:11434` / `llama3.2:1b` | Explanation step |
| `OLLAMA_TIMEOUT_SECONDS` | 5 | Per-call timeout |

## Tests & error boundaries

```bash
pip install pytest requests
pytest tests/ -v          # 54 tests, ~0.2 s, offline
```

54 tests covering each rule, every threshold edge, severity tiers, input
boundaries (blank cells, bad types, missing keys) and every Ollama failure
mode. What each test proves, the full error-boundary table, and — equally
important — what is **not** tested, are in
[`docs/testing-and-error-boundaries.md`](docs/testing-and-error-boundaries.md).

## Known limitations

- **The synthetic labels are incomplete, so precision is understated.** The
  generator injects mismatches into ~12% of sessions (36). In every other
  session the barrier text is drawn at random, independent of the rating and
  wording, so some *unlabelled* sessions contain the same kind of
  contradiction by chance (e.g. rating 8, "feeling much better", barrier
  "relapse into old coping habits" — session `S00003` is one). The detector
  flags them and the experiment scores them as false positives. Of the 85 false
  positives, **54 are keyword-rule flags with exactly this pattern** (rating
  ≥ 7, positive wording, negative barrier language). Counting them as correct
  would give precision ≈ 0.70 (71 / 102); that figure only sizes the effect —
  relabelling after seeing the detector's output would be circular, so the
  headline stays 0.17. The proper fix is to generate barrier text
  conditionally on progress, then re-run `experiments/experiment.py`.
- **No real counsellor validation yet.** `docs/stakeholder-validation.md`
  records one structured walkthrough with a non-clinical peer (n = 1). It
  surfaced a real usability gap, but it is not clinical evidence.
- **English-only keyword rule.** A barrier written only in Tamil or Tanglish
  will not trigger it; the rating-drop rule is language-independent.
- **Ratings are not range-checked** (1–10 is assumed).
- **The UI has no committed automated tests**, only the decision layer does.

## Design principles

1. **The AI never decides — it flags.** Every mismatch routes to a human
   counsellor (`docs/field-workflow-map.md`).
2. **Decisions are inspectable.** Flagging is a testable rule, not an LLM
   judgment. The SLM only explains.
3. **Failures are documented.** The semantic-check result is a real
   negative finding kept in code and in `experiments/experiment_results.md`.

## Repository layout

```
counsellor-outcome-tracker/
├── generate_synthetic_data.py
├── synthetic_data/            # clients.csv, goals.csv, sessions.csv
├── baseline/                  # attendance-only tracker
├── prototype/                 # Streamlit app + mismatch_detector.py
├── tests/                     # 54 pytest tests (rules, boundaries, Ollama failures)
├── experiments/               # baseline comparison + iteration history
└── docs/
    ├── field-workflow-map.md          # 5-stage workflow + human review points
    ├── patient-journeys.md            # two end-to-end walkthroughs
    ├── failure-mode-analysis.md       # how the tool could fail in clinical use
    ├── testing-and-error-boundaries.md
    └── stakeholder-validation.md      # one real peer walkthrough
```

## Status

Review-1 scored 88% (30.8 / 35). All six Review-1 items are addressed:
real-model experiment numbers committed, the rating-drop gap closed,
severity tiers and tests added, missing docs restored, and a real (n = 1)
stakeholder walkthrough recorded. Next: counsellor-level validation and
fixing the synthetic-label limitation above.
