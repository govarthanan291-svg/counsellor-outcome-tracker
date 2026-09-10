"""
mismatch_detector.py

Core AI layer for the Collaborative Outcome Tracker (Stage 4: Progress Tracking).

Two models are used together:
  1. sentence-transformers (all-MiniLM-L6-v2) -- computes semantic similarity
     between a client's self-reported progress text and their stated goal /
     barriers, to catch cases where the WORDS sound positive but the
     substance (barriers described) contradicts it.
  2. Ollama (local SLM, e.g. llama3.2:1b or phi3:mini) -- given a flagged
     case, produces a short, human-readable explanation of WHY it was
     flagged, for the counsellor to quickly review. The SLM never decides
     the outcome -- it only explains a decision made by the semantic check.

This keeps the "decision" (flag or don't flag) in a simple, inspectable,
testable numeric rule -- and uses the SLM only for the part language
models are actually good at: turning evidence into a readable sentence.
This separation matters for the failure-mode analysis: even if the SLM
hallucinates or is unavailable, the flagging logic still works correctly
on its own.

CHANGE LOG (post Review-1 feedback):
  - Added a rolling-average rating-drop check (`_rating_drop_mismatch`).
    This closes the confirmed gap from failure mode 1 (see
    docs/failure-mode-analysis.md and docs/patient-journeys.md, client
    C027): a sharp rating drop with no barrier text previously slipped
    through undetected because the semantic check has nothing to compare
    the self-reported text against when barriers are empty. This check is
    independent of barrier text entirely -- it only needs the client's own
    rating history.
  - Added a confidence tier (`severity`: "low" / "medium" / "high") instead
    of a bare binary flag, to address the alert-fatigue risk documented as
    failure mode 4. Severity is based on how many independent checks agree
    and how far the rating-drop exceeds its threshold, so a counsellor can
    triage a busy caseload by severity instead of treating every flag as
    equally urgent.

CHANGE LOG (post real-model evaluation, see experiments/experiment_results.md):
  - The semantic similarity check (`_semantic_mismatch`) was measured
    against the real sentence-transformers model and found to HURT overall
    performance: F1 dropped from 0.25 (rating-drop + keyword checks alone)
    to 0.12-0.16 with the semantic check included, across every threshold
    tested (see experiments/threshold_sweep.py). The reason: general-purpose
    sentence embeddings score two SHORT, topically-different phrases as
    dissimilar even when there's no real emotional contradiction between
    them (e.g. "feeling better this week" vs. "missed a session due to a
    deadline" -- unrelated topics, not a contradiction) -- conflating
    "different subject" with "contradicts what was said" produced far more
    false positives than it caught real mismatches.
  - Given this evidence, the semantic check is now OFF by default
    (`USE_SEMANTIC_CHECK = False`). The code is kept, not deleted, because
    disabling a measured-to-be-harmful check based on real evaluation data
    -- rather than removing the evidence trail -- is itself part of this
    project's documented, evidence-based development process. A properly
    recalibrated semantic approach (e.g. comparing emotional polarity
    rather than raw topical similarity) is noted as future work.

Requirements (install locally, needs internet access once for model download):
    pip install sentence-transformers requests

Ollama setup (optional -- explanations degrade gracefully without it):
    1. Install Ollama: https://ollama.com/download
    2. Run: ollama pull llama3.2:1b
    3. Run: ollama serve   (usually starts automatically after install)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import requests
from sentence_transformers import SentenceTransformer, util

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

# Below this cosine similarity between a client's self-reported text and
# their own barrier description, we consider the two to be in tension --
# e.g. text says "feeling much better" while barriers describe a relapse.
MISMATCH_SIMILARITY_THRESHOLD = 0.1  # best value found via experiments/threshold_sweep.py

# OFF by default -- see the change-log above. Measured against the real
# model, this check hurt overall F1 (0.25 -> 0.12-0.16) rather than helping.
# Left in place, and easy to re-enable, for anyone iterating on a better
# semantic approach.
USE_SEMANTIC_CHECK = False

# A high numeric rating (7+) paired with barrier text that reads negative
# is a second, independent signal checked alongside the text/barrier
# similarity check -- catches cases where the rating and the words disagree.
HIGH_RATING_THRESHOLD = 7
NEGATIVE_BARRIER_KEYWORDS = [
    "relapse", "reluctance", "struggl", "disrupted", "conflict", "overwhelm",
]

# A drop of this many points (or more) from the client's own rolling
# average rating is flagged regardless of what the barrier text says --
# this is what catches a client masking distress in their words while
# their numbers tell a different story.
RATING_DROP_THRESHOLD = 3
ROLLING_WINDOW = 3


@dataclass
class MismatchResult:
    session_id: str
    is_flagged: bool
    severity: str  # "none" | "low" | "medium" | "high"
    similarity_score: float
    reasons: list[str] = field(default_factory=list)
    explanation: Optional[str] = None


class MismatchDetector:
    def __init__(self, use_ollama: bool = True):
        print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.use_ollama = use_ollama

    def _semantic_mismatch(self, self_reported_text: str, barriers: str) -> tuple[bool, float]:
        """Flags when self-reported text and barrier description point in
        different emotional directions (low semantic alignment)."""
        if barriers.strip().lower() in ("no barriers this session", ""):
            return False, 1.0  # nothing to contradict
        emb = self.model.encode([self_reported_text, barriers], convert_to_tensor=True)
        score = float(util.cos_sim(emb[0], emb[1]))
        return score < MISMATCH_SIMILARITY_THRESHOLD, score

    def _rating_text_mismatch(self, rating: int, barriers: str) -> bool:
        """Flags when a high numeric rating is paired with a barrier
        description that contains clearly negative language."""
        if rating < HIGH_RATING_THRESHOLD:
            return False
        barrier_lower = barriers.lower()
        return any(kw in barrier_lower for kw in NEGATIVE_BARRIER_KEYWORDS)

    def _rating_drop_mismatch(self, rating: int, history: list[int]) -> tuple[bool, float]:
        """Flags a sharp drop from the client's own rolling-average rating,
        independent of barrier text. Closes the gap where a client's words
        sound fine but their numbers tell a different story with no
        barrier text to compare against.

        `history` is prior ratings for this goal, oldest first, NOT
        including the current session's rating.
        """
        if len(history) == 0:
            return False, 0.0
        window = history[-ROLLING_WINDOW:]
        rolling_avg = sum(window) / len(window)
        drop = rolling_avg - rating
        return drop >= RATING_DROP_THRESHOLD, round(drop, 2)

    def _generate_explanation(self, session: dict, reasons: list[str]) -> Optional[str]:
        if not self.use_ollama:
            return None
        prompt = (
            "You are assisting a counsellor reviewing a flagged client session. "
            "In one short sentence, explain plainly why this session was flagged "
            "for a possible mismatch between what the client reported and what "
            "actually happened. Do not give clinical advice, just describe the "
            "discrepancy.\n\n"
            f"Self-reported rating (1-10): {session['self_reported_rating']}\n"
            f"Self-reported text: \"{session['self_reported_text']}\"\n"
            f"Barriers noted: \"{session['barriers']}\"\n"
            f"Flag reasons: {', '.join(reasons)}\n"
        )
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            # Ollama not running / model not pulled / network issue -- degrade
            # gracefully. The flag itself is still valid without this text.
            return f"[Explanation unavailable: {e}]"

    def check_session(self, session: dict, rating_history: Optional[list[int]] = None) -> MismatchResult:
        """
        session: dict with session_id, self_reported_rating, self_reported_text, barriers
        rating_history: prior ratings for this same goal, oldest first (optional --
            omitting it disables the rating-drop check, e.g. for a client's first session)
        """
        reasons = []
        rating = int(session["self_reported_rating"])

        sem_flag, score = False, 1.0
        if USE_SEMANTIC_CHECK:
            sem_flag, score = self._semantic_mismatch(
                session["self_reported_text"], session["barriers"]
            )
            if sem_flag:
                reasons.append("self-reported text and barriers semantically diverge")

        if self._rating_text_mismatch(rating, session["barriers"]):
            reasons.append("high numeric rating paired with negative barrier language")

        drop_flag, drop_amount = self._rating_drop_mismatch(rating, rating_history or [])
        if drop_flag:
            reasons.append(f"rating dropped {drop_amount} points below recent average")

        is_flagged = len(reasons) > 0

        # Severity: more independent signals agreeing, or a larger rating
        # drop, means higher confidence this is a real mismatch worth
        # prioritizing -- lets a counsellor triage instead of treating
        # every flag identically (mitigates alert fatigue).
        if not is_flagged:
            severity = "none"
        elif len(reasons) >= 2 or drop_amount >= 5:
            severity = "high"
        elif len(reasons) == 1:
            severity = "medium"
        else:
            severity = "low"

        explanation = self._generate_explanation(session, reasons) if is_flagged else None

        return MismatchResult(
            session_id=session["session_id"],
            is_flagged=is_flagged,
            severity=severity,
            similarity_score=round(score, 3),
            reasons=reasons,
            explanation=explanation,
        )


if __name__ == "__main__":
    # Quick manual smoke test with three example sessions
    detector = MismatchDetector(use_ollama=True)

    example_ok = {
        "session_id": "TEST001",
        "self_reported_rating": 8,
        "self_reported_text": "made real progress on this",
        "barriers": "no barriers this session",
    }
    example_mismatch = {
        "session_id": "TEST002",
        "self_reported_rating": 8,
        "self_reported_text": "feeling much better this week, though honestly still struggling most days",
        "barriers": "relapse into old coping habits under stress",
    }
    example_rating_drop = {
        # This is the previously-missed pattern: positive text, empty
        # barriers, but a sharp drop from the client's recent average.
        "session_id": "TEST003",
        "self_reported_rating": 2,
        "self_reported_text": "feeling much better this week",
        "barriers": "no barriers this session",
    }

    print("--- Clean session (no history needed) ---")
    print(json.dumps(detector.check_session(example_ok).__dict__, indent=2))

    print("\n--- Text/barrier mismatch ---")
    print(json.dumps(detector.check_session(example_mismatch).__dict__, indent=2))

    print("\n--- Rating-drop mismatch (previously missed, now caught) ---")
    print(json.dumps(
        detector.check_session(example_rating_drop, rating_history=[7, 8, 7]).__dict__, indent=2
    ))
