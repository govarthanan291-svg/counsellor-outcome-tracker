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

PUBLIC INTERFACE (the only thing app.py and experiments/ call):
    MismatchDetector(use_ollama: bool).check_session(session, rating_history)
        -> MismatchResult
    See `check_session` for the exact input contract and which bad inputs
    raise vs. are tolerated. Full test/error-boundary write-up:
    docs/testing-and-error-boundaries.md

CHANGE LOG (post Review-1 feedback):
  - Added `_rating_drop_mismatch` (failure mode 1 / client C027).
  - Added severity tiers (low / medium / high) so a counsellor can triage
    instead of treating every flag as equally urgent (failure mode 4).

CHANGE LOG (post real-model evaluation):
  - Semantic check OFF by default (`USE_SEMANTIC_CHECK = False`).
    F1 was 0.25 with rating-drop + keyword alone, and 0.12-0.16 with the
    semantic check included, at every threshold tested.

CHANGE LOG (post Review-2 feedback, documentation pass):
  - Expanded comments (thresholds, input contract, Ollama behaviour).
  - Fixed a crash found while documenting error boundaries: a blank
    "barriers" cell is read by pandas as float NaN, and `NaN.lower()` raised
    AttributeError whenever rating >= 7. Cells are now normalised by
    `_as_text` before any string method touches them. Covered by
    tests/test_mismatch_detector.py::TestInputBoundaries.
  - Behaviour tightening (side effect of the same change): `self_reported_text`
    is now read up-front, so a missing key raises KeyError consistently.
    Before, it raised only if a flagged session later reached the Ollama
    prompt -- i.e. it depended on the data, which is worse than failing
    every time. app.py always passes full rows, so no UI impact.
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

# ---------------------------------------------------------------------------
# Configuration. These are module-level on purpose: tests patch them, and the
# experiments/ scripts read them, so there is exactly one source of truth.
# ---------------------------------------------------------------------------

# Ollama's local generate endpoint. This is the ONLY network call the module
# makes, and only when use_ollama=True. Nothing leaves the machine.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
# Short enough that a hung local server cannot stall a Cloud session;
# connection-refused (Ollama not installed) fails in milliseconds anyway.
OLLAMA_TIMEOUT_SECONDS = 5

# Semantic check only: cosine similarity below this counts as "diverging".
# 0.1 was the best of 11 thresholds swept in experiments/threshold_sweep.py,
# yet even that lost to the rule-only detector -- hence the flag below.
MISMATCH_SIMILARITY_THRESHOLD = 0.1  # best value from experiments/threshold_sweep.py

# OFF by default -- see change-log. Measured to hurt overall F1.
USE_SEMANTIC_CHECK = False

# Keyword check: a rating at or above this is "positive enough" that negative
# barrier language next to it is a contradiction worth a counsellor's look.
# (Ratings are 1-10, higher = better; the generator emits 2-9.)
HIGH_RATING_THRESHOLD = 7
# Matched as lowercase SUBSTRINGS, so "struggl" catches struggle/struggling/
# struggled. Substring matching is crude (it will also hit "no conflict"),
# which is why a keyword-only flag is tiered LOW, not higher.
NEGATIVE_BARRIER_KEYWORDS = [
    "relapse", "reluctance", "struggl", "disrupted", "conflict", "overwhelm",
]

# Rating-drop check: flag when (average of the last ROLLING_WINDOW prior
# ratings) - (current rating) >= RATING_DROP_THRESHOLD. Comparing a client
# to their OWN recent baseline (not a population norm) is deliberate: a
# steady 3/10 client is not "dropping", a 9 -> 4 client is.
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
    """What `check_session` returns. One per session, always.

    session_id        echoed back from the input so callers can join results
                      to their rows.
    is_flagged        True iff at least one rule fired.
    severity          "none" | "low" | "medium" | "high" (see `_severity`).
    similarity_score  semantic cosine similarity. It is a PLACEHOLDER 1.0
                      whenever the semantic check is off (the default) or
                      has nothing to compare -- do not read it as a measured
                      similarity unless USE_SEMANTIC_CHECK is True.
    reasons           human-readable list of every rule that fired.
    explanation       one counsellor-facing sentence from Ollama, the
                      OLLAMA_UNAVAILABLE_NOTE fallback, or None (not flagged,
                      or use_ollama=False).
    """
    session_id: str
    is_flagged: bool
    severity: str  # "none" | "low" | "medium" | "high"
    similarity_score: float
    reasons: list[str] = field(default_factory=list)
    explanation: Optional[str] = None


def _as_text(value) -> str:
    """Coerce a CSV / DataFrame cell to a plain string.

    Why this exists: pandas reads an empty CSV cell as float NaN, and
    `NaN.lower()` raises AttributeError. A counsellor leaving "barriers"
    blank is normal in real use (our synthetic data never does), so None and
    NaN both become "" -- "no barrier text", not an error.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN is the only float != itself
        return ""
    return str(value)


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
        # The embedding model is NOT loaded here: constructing a detector is
        # cheap and offline-safe, which is what lets the unit tests run with
        # no network and lets Streamlit Cloud start fast.
        self._model = None
        self.use_ollama = use_ollama
        # One-way latch. After the first connection/timeout/HTTP failure we
        # stop calling Ollama for the life of this process (typical Streamlit
        # Cloud case), so a 30-flag page costs one failed call, not 30.
        # It does not self-reset: restart the app after starting Ollama.
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
        different emotional directions (low semantic alignment).

        Returns (flagged, similarity). Only reached when USE_SEMANTIC_CHECK
        is True. Known weakness (why it is off): it measures topical
        similarity, so two unrelated-but-compatible phrases score "dissimilar".
        """
        if barriers.strip().lower() in ("no barriers this session", ""):
            return False, 1.0  # nothing to contradict
        model = self._load_model()
        emb = model.encode([self_reported_text, barriers], convert_to_tensor=True)
        score = float(util.cos_sim(emb[0], emb[1]))
        return score < MISMATCH_SIMILARITY_THRESHOLD, score

    def _rating_text_mismatch(self, rating: int, barriers: str) -> bool:
        """Flags when a high numeric rating is paired with a barrier
        description that contains clearly negative language.

        Note the direction: here the HIGH number is the suspicious part (the
        client says "8/10" but describes a relapse). A low rating never
        triggers this rule -- a low rating next to a bad barrier is
        consistent, not a mismatch.
        """
        if rating < HIGH_RATING_THRESHOLD:
            return False
        barrier_lower = barriers.lower()
        return any(kw in barrier_lower for kw in NEGATIVE_BARRIER_KEYWORDS)

    def _rating_drop_mismatch(self, rating: int, history: list[int]) -> tuple[bool, float]:
        """Flags a sharp drop from the client's own rolling-average rating,
        independent of barrier text.

        `history` is prior ratings for this goal, oldest first, NOT
        including the current session's rating.

        Returns (flagged, drop). `drop` is positive when the rating fell and
        negative when it rose; it is 0.0 with no history (a first session
        cannot "drop").
        """
        if len(history) == 0:
            return False, 0.0
        window = history[-ROLLING_WINDOW:]
        rolling_avg = sum(window) / len(window)
        drop = rolling_avg - rating
        return drop >= RATING_DROP_THRESHOLD, round(drop, 2)

    def _generate_explanation(self, session: dict, reasons: list[str]) -> Optional[str]:
        """One counsellor-facing sentence, or a fallback note, or None.

        State table (use_ollama / _ollama_reachable -> return):
            False / -     -> None                      (explanations disabled)
            True  / False -> OLLAMA_UNAVAILABLE_NOTE   (already failed once)
            True  / True  -> POST to Ollama; on success the model's sentence,
                             on ANY failure the note, and the latch flips.
        The explanation never influences is_flagged or severity: those are
        decided before this is called.
        """
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
            # HTTP error status, malformed JSON, anything else. Same outcome
            # as above on purpose: an explanation is a nice-to-have and must
            # never be able to break flagging.
            self._ollama_reachable = False
            return OLLAMA_UNAVAILABLE_NOTE

    def check_session(self, session: dict, rating_history: Optional[list[int]] = None) -> MismatchResult:
        """
        Evaluate ONE session and return a MismatchResult.

        INPUT CONTRACT
          session: dict with keys
              session_id            any (echoed back)
              self_reported_rating  int, or a string int(...) accepts ("8").
                                    Expected range 1-10 but NOT range-checked.
              self_reported_text    str (blank / None / NaN tolerated -> "")
              barriers              str (blank / None / NaN tolerated -> "")
          rating_history: prior ratings for this same goal, oldest first, not
              including this session. None or [] disables the rating-drop
              check (e.g. a client's first session).

        FAILS LOUDLY, ON PURPOSE (a silently-wrong clinical flag is worse
        than a visible error):
          KeyError    a required key is missing
          ValueError  self_reported_rating is not int-convertible ("eight")
        TOLERATED: blank text/barriers, string-typed numeric ratings,
        upper-case barrier text (matching is case-insensitive).
        NOT VALIDATED (known gap): ratings outside 1-10 are accepted as-is.
        """
        reasons = []
        rating = int(session["self_reported_rating"])
        text = _as_text(session["self_reported_text"])
        barriers = _as_text(session["barriers"])

        # Rule 3 (optional, off by default). `score` stays at the 1.0
        # placeholder unless this block runs.
        sem_flag, score = False, 1.0
        if USE_SEMANTIC_CHECK:
            sem_flag, score = self._semantic_mismatch(text, barriers)
            if sem_flag:
                reasons.append("self-reported text and barriers semantically diverge")

        # Rule 2: high rating + negative barrier language.
        if self._rating_text_mismatch(rating, barriers):
            reasons.append("high numeric rating paired with negative barrier language")

        # Rule 1: sharp drop vs this client's own recent average.
        drop_flag, drop_amount = self._rating_drop_mismatch(rating, rating_history or [])
        if drop_flag:
            reasons.append(f"rating dropped {drop_amount} points below recent average")

        # Decision is final here -- everything below is presentation only.
        is_flagged = len(reasons) > 0
        severity = _severity(reasons, drop_amount, drop_flag)

        # Pass the cleaned strings so a blank cell prints as "" in the
        # prompt, not as the literal word "nan".
        clean = {**session, "self_reported_text": text, "barriers": barriers}
        explanation = self._generate_explanation(clean, reasons) if is_flagged else None

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
