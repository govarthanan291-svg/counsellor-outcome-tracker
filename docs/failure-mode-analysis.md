# Failure-Mode Analysis — AI Mismatch Detection Layer

This document identifies concrete ways the Stage 4 AI layer (semantic
similarity + rating/text divergence check, explained via Ollama) can fail in
real clinical operation, based on evidence from the synthetic data and the
patient journey walkthroughs (`docs/patient-journeys.md`).

---

## Failure mode 1: Silent rating-drop-with-no-barrier-text (CONFIRMED — see C027, S00195)

**What happens:** A client's numeric rating drops sharply (e.g. 7/10 → 2/10)
but their self-reported text still reads positively ("feeling much better
this week"), and the barrier field is empty or "no barriers this session."

**Why the current detector misses it:** The semantic-mismatch check compares
self-reported text against barrier text. With no barrier text to compare
against, there's nothing to contradict — so the check silently passes.

**Real-world consequence:** A client masking distress (a known risk pattern
in clinical settings) is not flagged. The counsellor only catches it if they
manually notice the rating trend themselves — which defeats the purpose of
having an automated flag.

**Mitigation for the next iteration:** Add a second, independent check that
compares the current rating against the client's own rolling average,
regardless of barrier text — flag any drop beyond a threshold (e.g. 3+
points) even when barriers are empty.

---

## Failure mode 2: Cultural / linguistic variation in emotional expression

**What happens:** Some clients express distress indirectly, through
understatement, humor, or culturally-specific phrasing ("I'm managing, I
guess" said with heavy sarcasm) rather than direct negative language.

**Why the current detector misses it:** The sentence-transformers model was
trained on general-purpose text and has no clinical or cultural calibration.
Semantic similarity scores reflect surface meaning, not tone, sarcasm, or
culturally-coded understatement.

**Real-world consequence:** The system could systematically under-flag
certain communication styles, producing unequal quality of care across
different client populations — a fairness/equity failure, not just an
accuracy one.

**Mitigation for the next iteration:** This is not fully solvable with a
similarity-threshold approach alone; flag it explicitly to stakeholders as a
known limitation, and recommend the tool is positioned as a *support* for
counsellor judgment, never a replacement for it — especially in early
deployment.

---

## Failure mode 3: SLM (Ollama) explanation hallucination or unavailability

**What happens:** The Ollama-generated explanation sentence shown to the
counsellor could (a) be unavailable if Ollama isn't running, or (b)
generate a plausible-sounding but inaccurate gloss of the flagged evidence
if the small local model misinterprets the prompt.

**Why this could happen:** Small local models (1B–3B parameters) are more
prone to minor factual slips than larger models, especially under
constrained context.

**Real-world consequence:** A counsellor skimming a busy caseload could
over-trust a misleading AI explanation instead of reading the raw session
data themselves.

**Mitigation already built in:** The flagging decision itself is made by
the deterministic similarity/rating rule, not the SLM — the SLM only
explains an already-made decision. The raw session data (rating, text,
barriers) is always shown alongside the explanation in the UI, so the
counsellor is never asked to trust the explanation alone (see `app.py`).
This is a partial mitigation, not a full fix — the explanation could still
mislead a counsellor who doesn't check the raw data.

---

## Failure mode 4: False positives eroding counsellor trust

**What happens:** The rating/text-vs-barrier rule can flag sessions that
are not actually clinically meaningful — e.g. a client who used mildly
inconsistent wording without any real contradiction in substance.

**Why this could happen:** The similarity threshold (0.35 cosine similarity)
is a blunt, tunable cutoff, not a clinical judgment. Any fixed threshold
will produce some false positives.

**Real-world consequence:** If counsellors repeatedly find flagged sessions
that turn out to be non-issues, they may start ignoring flags altogether
(alert fatigue) — meaning the tool fails silently in a different way: not
by missing real mismatches, but by causing real mismatches to be dismissed
along with the noise.

**Mitigation for the next iteration:** Track false-positive rate against
counsellor feedback over time and let the threshold be tuned per-cohort;
show a confidence indicator rather than a binary flag so counsellors can
triage by severity.

---

## Summary table

| # | Failure mode | Type | Currently mitigated? |
|---|---|---|---|
| 1 | Rating drop with no barrier text | Missed detection (false negative) | No — confirmed gap, fix proposed |
| 2 | Cultural/linguistic variation in expression | Systematic bias / fairness | Partially — flagged as a known limitation |
| 3 | SLM explanation hallucination/unavailability | Misleading output | Partially — raw data always shown alongside |
| 4 | False positives → alert fatigue | Trust erosion | No — proposed for future iteration |

These failure modes are not hypothetical edge cases invented for the
assignment — failure mode 1 was discovered directly while building the
patient-journey demonstration (Step 5), which is exactly the kind of
honest, evidence-based failure analysis this project calls for.
