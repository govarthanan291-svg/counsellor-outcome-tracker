# Field-Workflow Map — Collaborative Outcome Tracker (University Counselling Centre)

## Overview

This document maps the 5-stage workflow for the collaborative outcome tracker, replacing the current attendance-only tracking approach with client-defined, goal-based outcome tracking. Each stage lists its input, process, output, and whether a human review checkpoint is required.

---

## Stage 1: Intake

| Field | Detail |
|---|---|
| **Input** | New client referral / self-referral form; presenting concern; demographic and context info |
| **Process** | Risk screening (safety/crisis indicators checked); intake questionnaire administered; client's initial concerns captured in their own words |
| **Output** | Risk level classification (low / moderate / high); intake record created in system |
| **Human review checkpoint** | ✅ **Yes** — Counsellor/triage staff must review and confirm risk classification before proceeding. No client is routed to goal setting without this sign-off. |

---

## Stage 2: Goal Setting

| Field | Detail |
|---|---|
| **Input** | Intake record; client's presenting concern; counsellor's clinical judgment |
| **Process** | Client and counsellor collaboratively define 1–3 measurable, client-owned goals (not counsellor-imposed); goals are phrased in the client's own language where possible; baseline state recorded for each goal |
| **Output** | Structured goal record (goal text, baseline, target, target date) tied to the client |
| **Human review checkpoint** | ✅ **Yes** — Counsellor validates that goals are realistic, safe, and genuinely client-defined (not just administratively convenient) before they're locked in |

---

## Stage 3: Sessions

| Field | Detail |
|---|---|
| **Input** | Active goal record; scheduled session; counsellor's session notes |
| **Process** | Each session logs: self-reported progress (client's own rating/description), barriers encountered, and any agreed adjustments to the goal or approach |
| **Output** | Session log entries appended to the client's goal history (progress, barriers, adjustments) |
| **Human review checkpoint** | ❌ No — routine logging; counsellor writes notes as part of normal practice, no separate review step required |

---

## Stage 4: Progress Tracking

| Field | Detail |
|---|---|
| **Input** | Accumulated session logs (self-reported progress + barriers + adjustments) for a client's goal |
| **Process** | AI (local SLM + sentence-transformers) compares self-reported progress against the stated goal semantically, flags mismatches (e.g., client says "doing better" but barriers described suggest otherwise) |
| **Output** | Progress status (on-track / mismatch-flagged) with supporting evidence extracted from session logs |
| **Human review checkpoint** | ✅ **Yes** — Any flagged mismatch is routed to the counsellor for review before any action is taken or conclusion is drawn. AI flags, never decides. |

---

## Stage 5: Outcome Review

| Field | Detail |
|---|---|
| **Input** | Goal record, full session history, progress tracking status, any flagged-and-reviewed mismatches |
| **Process** | Counsellor and client jointly review progress against the original goal; goal is either marked achieved, revised, or continued; final outcome documented with evidence |
| **Output** | Final outcome record: baseline, target, measured result, and brief rationale/evidence — replacing the old "attended X sessions" metric |
| **Human review checkpoint** | ❌ No separate checkpoint — this stage *is* the human-led review by design (client + counsellor together) |

---

## Summary: Human Review Checkpoints

| Stage | Checkpoint | Who reviews |
|---|---|---|
| 1. Intake | Risk classification | Counsellor / triage staff |
| 2. Goal setting | Goal validation | Counsellor |
| 4. Progress tracking | Flagged mismatches | Counsellor |

**Design principle:** The AI/automation layer only appears at Stage 4 (progress tracking), and only as a *flagging* mechanism — it never makes clinical judgments or decisions. All decision points remain human-owned, which is central to the failure-mode analysis for this project (an AI misclassification at Stage 4 is caught by design before it can affect a real client outcome).
