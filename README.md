# Collaborative Outcome Tracker

A Streamlit prototype for a university counselling centre. It replaces
attendance-only outcome tracking with **client-defined goal tracking**, and
flags sessions where a client's rating or wording does not line up with
the rest of their record — for a counsellor to review, never as an
automatic clinical conclusion.

Stack: Python, Streamlit, pandas, Altair, optional sentence-transformers,
optional Ollama (local SLM).

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
triaged. Ollama only writes a one-sentence explanation of an already-made
flag; if Ollama is not running (Streamlit Cloud), explanations are skipped
after the first failed contact.

## Measured comparison vs attendance baseline

293 synthetic sessions, 36 planted mismatches.

| | Attendance baseline | Current prototype |
|---|---|---|
| Precision | n/a | 0.17 |
| Recall | 0 | 0.47 |
| F1 | n/a | 0.25 |

Reproduce: `python3 experiments/experiment.py` (from `experiments/`).

## Quick start

```bash
python3 generate_synthetic_data.py
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

Full local + Ollama notes: `prototype/README.md`.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Repository layout

```
counsellor-outcome-tracker/
├── generate_synthetic_data.py
├── synthetic_data/            # clients.csv, goals.csv, sessions.csv
├── baseline/                  # attendance-only tracker
├── prototype/                 # Streamlit app + mismatch_detector.py
├── tests/                     # pytest suite (severity tiers, semantic skip)
├── experiments/               # baseline comparison + iteration history
└── docs/                      # workflow map, journeys, stakeholder notes
```

## Design principles

1. **The AI never decides — it flags.** Every mismatch routes to a human
   counsellor (`docs/field-workflow-map.md`).
2. **Decisions are inspectable.** Flagging is a testable rule, not an LLM
   judgment. The SLM only explains.
3. **Failures are documented.** The semantic-check result is a real
   negative finding kept in code and in `experiments/experiment_results.md`.

## Status

Review-1 scored 88% (30.8 / 35). Remaining gap that cannot be generated:
**genuine stakeholder input** from a real person (do not treat
`docs/stakeholder-validation.md` as that). Real experiment numbers for the
current detector are committed as above.
