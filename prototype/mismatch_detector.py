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
MISMATCH_SIMILARITY_THRESHOLD = 0.35

# A high numeric rating (7+) paired with barrier text that reads negative
# is a second, independent signal checked alongside the text/barrier
# similarity check -- catches cases where the rating and the words disagree.
HIGH_RATING_THRESHOLD = 7
NEGATIVE_BARRIER_KEYWORDS = [
    "relapse", "reluctance", "struggl", "disrupted", "conflict", "overwhelm",
]


@dataclass
class MismatchResult:
    session_id: str
    is_flagged: bool
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

    def check_session(self, session: dict) -> MismatchResult:
        reasons = []

        sem_flag, score = self._semantic_mismatch(
            session["self_reported_text"], session["barriers"]
        )
        if sem_flag:
            reasons.append("self-reported text and barriers semantically diverge")

        if self._rating_text_mismatch(int(session["self_reported_rating"]), session["barriers"]):
            reasons.append("high numeric rating paired with negative barrier language")

        is_flagged = len(reasons) > 0
        explanation = self._generate_explanation(session, reasons) if is_flagged else None

        return MismatchResult(
            session_id=session["session_id"],
            is_flagged=is_flagged,
            similarity_score=round(score, 3),
            reasons=reasons,
            explanation=explanation,
        )


if __name__ == "__main__":
    # Quick manual smoke test with two example sessions
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

    for ex in (example_ok, example_mismatch):
        result = detector.check_session(ex)
        print(json.dumps(result.__dict__, indent=2))
