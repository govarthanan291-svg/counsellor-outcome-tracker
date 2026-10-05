# User / Stakeholder Validation Summary

## Real feedback (collected after Review-1)

A structured walkthrough was conducted with a real person (a friend,
non-clinical background) using the actual working prototype — not a mockup.
This replaces the simulated personas used before Review-1, in direct
response to reviewer feedback asking for genuine human input.

**Format:** ~10 minute live demo of the Streamlit app, walking through
client C001 (Danielle J., goal "Manage panic attacks", baseline "2-3 panic
attacks per week", target "reduced to under 1 per month" by 2026-03-19),
running the mismatch check, and reviewing the flagged session together.
Then five open questions, answered in their own words.

**The session that anchored the walkthrough:** Session `S00003`
(2026-02-15) — self-reported rating 8/10, self-reported text "feeling much
better this week," barrier text "relapse into old coping habits." The
prototype flags this as a rating/keyword mismatch (a high rating paired
with clearly negative barrier language). The person independently noticed
and confirmed this contradiction before being told it was the flagged
case.

### Q1 — First impression
> "Idhu oru Mental Health Outcome Tracker maathri theriyudhu. Counselor
> illana therapist, avanga client oda progress ah track panna use
> panranga. Just attendance mattum paakama, client real-ah progress
> aagurangala nu paaka help pannudhu."

(Reads it as a mental-health outcome tracker for counsellors/therapists —
correctly identified the core shift away from attendance.)

### Q2 — Trust
> "Kandipa naan manual-ah check dhaan pannuven. AI flag panradhu automatic
> rating-keyword mismatch vechu dhaan. Enna dhaan code eladhunalu,
> unmaiyana human emotion-ah adhala sariya purinjuka mudiyadhu... So blind
> ah trust panna koodathu."

(Would not trust a flag blindly — wants to verify manually. Matches the
project's own design principle that the AI flags, never decides.)

### Q3 — Workload
> "Aama, nariya cases handle pannum bodhu 1-in-3 flag aana processing
> thodharba konjam overwhelming ah dhaan irukkum. Aana ithu serious cases
> ah prioritize panna nalla use aagum. False flags nariya vunduna dhaan
> kadupayidum."

(Confirms the alert-fatigue risk already documented in
`docs/failure-mode-analysis.md` — but also sees the severity-tiering as a
genuine mitigation for prioritizing serious cases.)

### Q4 — Clarity
> "Aama, starting-la Rating Scale konjam confuse aachu. Rating dropped-nu
> poduranga, aana rating score 8/10 nu iruku (usually 8/10 nalladhunu
> neneipom). Inga high score dhaan bad progress pola. Adhuvum illama
> graph la timeline la clear-ah gaps theriyala."

(A genuinely new finding: a high self-reported rating (8/10) read as
"good" at a glance, even when it's the exact rating that got the session
flagged as a mismatch. The rating/keyword check and the rating-drop check
can both fire on different sessions, and nothing in the UI currently
distinguishes which rule flagged a given session or why "high number +
negative barrier" is the bad case here. Also flagged: no visual gap
marker on the ratings chart for missed or widely-spaced sessions.)

### Q5 — Overall
> "100% better-ah irukku! Attendance dhaan just a number... Ipdhi
> qualitative text and barrier tracking pandradhu dhaan nijamana therapy
> progress."

(Clear endorsement of the core hypothesis: goal- and evidence-based
tracking over attendance-only.)

---

## What this surfaced that the technical experiment didn't

The precision/recall numbers in `experiments/experiment_results.md` say
*how often* the detector is right. This walkthrough surfaced something
different: **why** a flag could be right and still confusing. Q4 is a
concrete, new UI finding — the app doesn't currently tell a counsellor
*which* rule fired (rating-drop vs. keyword) or make clear that, for the
keyword check, a high number is the bad sign, not the good one. That's a
real usability gap this single session caught that the automated test
suite and the quantitative experiment both have no way to catch.

## Action items from this feedback

| Finding | Source | Action |
|---|---|---|
| High rating + negative keyword reads as "good" at a glance | Q4 | Label which rule fired per flag ("rating/keyword mismatch" vs. "rating-drop"), not just the reasons list |
| No visual gap marker for missed/spaced sessions | Q4 | Future work — mark session gaps on the ratings chart |
| 1-in-3 flag rate feels heavy under full caseload | Q3 | Matches documented failure mode 4; severity tiers are a partial mitigation already shipped |
| Manual verification is expected, not optional | Q2 | Confirms the "AI flags, never decides" design principle is read correctly by a first-time user |

---

## Note on the pre-Review-1 simulated validation

Before Review-1, this document used simulated counsellor/client personas,
explicitly disclosed as simulated, because real access wasn't available
within the original timeline. That simulated version is not reproduced
here — this document now reflects the real feedback collected above.
