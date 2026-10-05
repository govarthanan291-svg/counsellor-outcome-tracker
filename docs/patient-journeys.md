# Two Patient Journeys — Workflow Demonstration

This document walks through the full 5-stage workflow (see `docs/field-workflow-map.md`)
for two real synthetic clients of different urgency levels, showing exactly
where human review checkpoints occur and how the AI mismatch-detection layer
behaves — including a case where it currently **fails** to catch a genuine
mismatch (used later in the failure-mode analysis, Step 6).

---

## Journey 1: C001 — Low urgency

**Presenting concern:** Sleep difficulties · **Age:** 19 · **Intake date:** 2026-01-21

### Stage 1 — Intake ✅ human review checkpoint
Risk screening classifies C001 as **low urgency**. Counsellor confirms — no
immediate safety concerns, routine intake proceeds.

### Stage 2 — Goal setting ✅ human review checkpoint
Client and counsellor collaboratively set goal **G0001: "Manage panic
attacks"** — baseline "2–3 panic attacks per week", target "reduced to under
1 per month". Counsellor validates the goal is realistic and client-owned.

### Stage 3 — Sessions (routine logging, no checkpoint)

| Session | Date | Rating | Self-reported | Barriers | Adjustment |
|---|---|---|---|---|---|
| S00001 | 01-30 | 4/10 | about the same as last week | family obligations limited practice time | none needed |
| S00002 | 02-08 | 9/10 | made real progress on this | difficulty applying strategies during exam week | shifted target date +2 weeks |
| S00003 | 02-15 | 8/10 | feeling much better this week | **relapse into old coping habits under stress** | none needed |

### Stage 4 — Progress tracking ✅ human review checkpoint (on flags)
**S00003 is flagged.** Rating (8/10) and text ("feeling much better") say
things are going well — but the barrier field says "relapse into old coping
habits under stress." The rating/text-vs-barrier rule catches this
contradiction and routes it to the counsellor for review, with an AI-generated
explanation sentence (via Ollama) to speed up the read.

### Stage 5 — Outcome review
Counsellor and client review the flagged session together at the next
meeting — clarifies that the "relapse" was a single stressful week, not a
reversal. Goal continues unchanged. Outcome documented with this context,
not just a session count.

---

## Journey 2: C027 — High urgency

**Presenting concern:** Burnout · **Age:** 24 · **Intake date:** 2026-01-29

### Stage 1 — Intake ✅ human review checkpoint
Risk screening classifies C027 as **high urgency** (burnout risk indicators
present). Counsellor confirms elevated priority and schedules sooner
follow-up than a routine case.

### Stage 2 — Goal setting ✅ human review checkpoint
Goal **G0035: "Improve study-life balance"** — baseline "studying till 2am
most nights, no breaks", target "fixed study hours with evening downtime".
Counsellor validates this is achievable given the burnout presentation.

### Stage 3 — Sessions (routine logging, no checkpoint)

| Session | Date | Rating | Self-reported | Barriers | Adjustment |
|---|---|---|---|---|---|
| S00191 | 02-06 | 6/10 | some ups and downs | none | none needed |
| S00192 | 02-12 | 5/10 | about the same as last week | financial stress distracted from goal work | grounding technique added |
| S00193 | 02-20 | 8/10 | made real progress on this | missed sessions due to coursework deadlines | focus reduced to one barrier |
| S00194 | 03-01 | 7/10 | made real progress on this | **reluctance to open up about the real issue** | goal revised to be more realistic |
| S00195 | 03-07 | **2/10** | **feeling much better this week** | none | peer support group involved |

### Stage 4 — Progress tracking ✅ human review checkpoint (on flags)
- **S00194 is correctly flagged** — high rating (7/10) paired with a barrier
  describing reluctance to engage honestly. The rule catches this and routes
  it for review.
- **S00195 is a real mismatch that the current detector MISSES.** The rating
  dropped sharply to 2/10, but the text still says "feeling much better this
  week" — a serious internal contradiction. Because the barrier field is
  empty ("no barriers this session"), the current text-vs-barrier comparison
  has nothing to compare against, so no flag is raised. **This is a genuine
  failure case**, carried forward into the failure-mode analysis (Step 6): the
  detector doesn't yet compare the rating directly against the text's
  sentiment when barriers are absent.

### Stage 5 — Outcome review
Because S00195 was never flagged, this contradiction only surfaces if the
counsellor happens to notice the rating drop themselves during manual
review — which is exactly the kind of gap the tool is meant to close, and
exactly the kind of gap that must be documented honestly rather than hidden.

---

## What this comparison shows

| | C001 (low urgency) | C027 (high urgency) |
|---|---|---|
| Sessions logged | 3 (goal 1) | 5 |
| Flags raised by rule engine | 1 (S00003) | 1 (S00194) |
| Genuine mismatches present | 1 | 2 |
| Mismatches caught | 1/1 | 1/2 |
| Human review checkpoints reached | 2 (intake, goal) + 1 (flag) | 2 (intake, goal) + 1 (flag) |

Both journeys pass through the same three human review checkpoints
regardless of urgency — the workflow doesn't skip review steps for
low-urgency clients. But C027's case exposes a real limitation: **a
sharp rating drop with no accompanying barrier text currently slips through.**
This is exactly the kind of honest, documented failure the project brief
asks for, and it directly motivates one of the three-plus edge cases in the
next step (Step 6: failure-mode analysis).