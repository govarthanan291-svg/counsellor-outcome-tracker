# User / Stakeholder Validation Summary (Simulated)

**Note on methodology:** Real access to a counsellor or client at a
university counselling centre was not available within the project
timeline. This validation is a **simulated stakeholder walkthrough** —
constructed by walking a counsellor persona and a client persona through
the actual working prototype (not a mockup) and reasoning through their
likely reactions based on the tool's real behavior, the patient journeys
already demonstrated, and known constraints in real clinical workflows.
This is disclosed here explicitly rather than presented as real feedback,
in keeping with the project's own emphasis on honest failure/limitation
reporting.

**What real validation would involve, if extended:** A short structured
walkthrough (15–20 min) with an actual counsellor at the centre, showing
the C001 (low urgency) and C027 (high urgency) journeys from
`docs/patient-journeys.md`, followed by a brief structured feedback form
covering trust, workload impact, and clarity of the flagged explanations.

---

## Simulated persona 1: Counsellor

**Walkthrough:** Shown the dashboard for C027 (high urgency), including
the flagged session and the AI-generated explanation, then the missed
rating-drop case identified in the failure-mode analysis.

**Likely reaction (positive):**
- Would likely value that flags come with the *raw evidence* (rating,
  text, barriers) directly alongside the AI explanation, rather than a
  bare verdict — this matches how counsellors already work (checking
  evidence, not accepting conclusions at face value)
- Would likely see the goal-based history as more useful than a session
  count for actual progress conversations with a client

**Likely reaction (concern):**
- Would likely be uncomfortable with the false-positive rate (54 of 293
  sessions flagged in the experiment, only 12 being real matches) if
  deployed as-is — flagging nearly 1 in 5 sessions is not sustainable for
  a caseload during exam-period peak demand
- Would likely ask what happens when Ollama is down mid-session — the
  degraded "[Explanation unavailable]" message needs a clearer, less
  technical wording for a non-technical user

## Simulated persona 2: Client (student)

**Walkthrough:** Reasoned through what it would feel like for a client
whose session was flagged, referencing the C027 journey where the client
was masking distress.

**Likely reaction (positive):**
- Would likely appreciate that the *goal itself* is defined in their own
  words, rather than a clinical label — matches the project's stated aim
  of "meaningful improvement defined with the client," not attendance

**Likely reaction (concern):**
- Would likely feel uneasy knowing an AI system is comparing what they say
  against what they say elsewhere — this needs to be transparently
  disclosed to clients as part of informed consent, not run silently in
  the background

---

## Synthesized recommendation

| Theme | Finding | Action before real deployment |
|---|---|---|
| Trust | Evidence-alongside-flag design is appropriate | Keep as-is |
| Workload | False-positive rate too high for peak-demand periods | Tune threshold, or add a confidence tier instead of binary flag |
| Failure handling | Ollama-unavailable message is too technical | Rewrite for non-technical counsellor-facing language |
| Consent | Clients aren't told their text is being compared automatically | Add explicit disclosure step at intake (Stage 1) |
| Core value | Client-defined goals are seen as a real improvement over attendance-only | No change needed — this is the project's central hypothesis, and it holds up |

This simulated exercise doesn't replace real stakeholder input, but it
surfaces concrete, specific concerns (workload, consent, false-positive
tolerance) that a purely technical evaluation (the precision/recall
numbers in `experiments/experiment_results.md`) would not have caught on
its own — which is itself the argument for why a real validation step
matters before any actual deployment.
