"""
mismatch_detector.py

Core AI layer for the Collaborative Outcome Tracker (Stage 4: Progress Tracking).

Decision logic (flag or don't flag) is a small set of inspectable rules:
  1. Rating-drop check -- a sharp drop from the client's own rolling-average
     rating, independent of barrier text.
  2. Keyword check -- a high numeric rating paired with clearly negative
     barrier language.
  3. Semantic similarity check -- OPTIONAL, OFF by default. Measured against
     the real sentence-transformers model and found to hurt precision
     (see experiments/experiment_results.md). Code is kept for future work
     on a polarity-based approach, not topical cosine similarity.

Ollama (local SLM) never decides the outcome. If enabled, it only turns an
already-made flag into a short counsellor-facing sentence. If Ollama is
down (the normal case on Streamlit Cloud), flagging still works and
explanations are skipped after the first failed contact.

CHANGE LOG (post Review-1 feedback):
  - Added `_rating_drop_mismatch` (failure mode 1 / client C027).
  - Added severity tiers (low / medium / high) so a counsellor can triage
    instead of treating every flag as equally urgent (failure mode 4).

CHANGE LOG (post real-model evaluation):
  - Semantic check OFF by default (`USE_SEMANTIC_CHECK = False`).
    F1 was 0.25 with rating-drop + keyword alone, and 0.12-0.16 with the
    semantic check included, at every threshold tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import requests

# Imported lazily in `_load_model` so Streamlit Cloud / tests do not
# download all-MiniLM-L6-v2 unless the optional semantic check is on.
SentenceTransformer = None  # set on first use
util = None

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
# Short enough that a hung local server cannot stall a Cloud session;
# connection-refused (Ollama not installed) fails in milliseconds anyway.
OLLAMA_TIMEOUT_SECONDS = 5

MISMATCH_SIMILARITY_THRESHOLD = 0.1  # best value from experiments/threshold_sweep.py

# OFF by default -- see change-log. Measured to hurt overall F1.
USE_SEMANTIC_CHECK = False

HIGH_RATING_THRESHOLD = 7
NEGATIVE_BARRIER_KEYWORDS = [
    "relapse", "reluctance", "struggl", "disrupted", "conflict", "overwhelm",
]

RATING_DROP_THRESHOLD = 3
# A drop this large, even as a sole signal, is treated as high severity.
LARGE_DROP_THRESHOLD = 5
ROLLING_WINDOW = 3

# Counsellor-facing copy when Ollama is missing or has already failed once.
OLLAMA_UNAVAILABLE_NOTE = (
    "Plain-language explanation skipped — the local SLM is not reachable. "
    "The flag and reasons above are still valid."
)


@dataclass
class MismatchResult:
    session_id: str
    is_flagged: bool
    severity: str  # "none" | "low" | "medium" | "high"
    similarity_score: float
    reasons: list[str] = field(default_factory=list)
    explanation: Optional[str] = None


def _severity(reasons: list[str], drop_amount: float, drop_flagged: bool) -> str:
    """Map independent signals onto a triage tier.

    - none:   no signals
    - low:    keyword mismatch only (noisy substring heuristic)
    - medium: a modest rating-drop [RATING_DROP_THRESHOLD, LARGE_DROP_THRESHOLD)
              as the sole signal
    - high:   two or more signals, or a rating-drop of LARGE_DROP_THRESHOLD+

    The previous `else: low` branch was unreachable: a flagged session always
    has at least one reason, and `len(reasons) == 1` swallowed every
    single-signal case as medium.
    """
    if not reasons:
        return "none"
    if len(reasons) >= 2 or drop_amount >= LARGE_DROP_THRESHOLD:
        return "high"
    if drop_flagged:
        return "medium"
    return "low"


class MismatchDetector:
    def __init__(self, use_ollama: bool = True):
        self._model = None
        self.use_ollama = use_ollama
        # After the first connection/timeout/HTTP failure, skip further
        # Ollama calls for this process (typical Streamlit Cloud case).
        self._ollama_reachable = use_ollama

    def _load_model(self):
        """Load sentence-transformers only if the semantic check is on."""
        global SentenceTransformer, util
        if self._model is not None:
            return self._model
        print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer as ST, util as st_util
        SentenceTransformer = ST
        util = st_util
        self._model = ST("all-MiniLM-L6-v2")
        return self._model

    def _semantic_mismatch(self, self_reported_text: str, barriers: str) -> tuple[bool, float]:
        """Flags when self-reported text and barrier description point in
        different emotional directions (low semantic alignment)."""
        if barriers.strip().lower() in ("no barriers this session", ""):
            return False, 1.0  # nothing to contradict
        model = self._load_model()
        emb = model.encode([self_reported_text, barriers], convert_to_tensor=True)
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
        independent of barrier text.

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
        if not self.use_ollama or not self._ollama_reachable:
            return OLLAMA_UNAVAILABLE_NOTE if self.use_ollama else None
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
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or None
        except (requests.ConnectionError, requests.Timeout):
            # Nothing listening on :11434 (Cloud), or the SLM hung.
            self._ollama_reachable = False
            return OLLAMA_UNAVAILABLE_NOTE
        except Exception:
            self._ollama_reachable = False
            return OLLAMA_UNAVAILABLE_NOTE

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
        severity = _severity(reasons, drop_amount, drop_flag)

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
    detector = MismatchDetector(use_ollama=False)

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
        "session_id": "TEST003",
        "self_reported_rating": 2,
        "self_reported_text": "feeling much better this week",
        "barriers": "no barriers this session",
    }

    print("--- Clean session (no history needed) ---")
    print(json.dumps(detector.check_session(example_ok).__dict__, indent=2))

    print("\n--- Keyword mismatch (low severity: sole keyword signal) ---")
    print(json.dumps(detector.check_session(example_mismatch).__dict__, indent=2))

    print("\n--- Rating-drop mismatch (high: drop >= 5) ---")
    print(json.dumps(
        detector.check_session(example_rating_drop, rating_history=[7, 8, 7]).__dict__, indent=2
    ))
