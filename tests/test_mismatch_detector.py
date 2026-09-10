"""
tests/test_mismatch_detector.py

Automated tests for the core AI decision logic in prototype/mismatch_detector.py.

These tests mock the sentence-transformers model itself (no internet /
model download needed to run them -- CI-friendly), so they focus on what
actually matters for correctness: the flagging rules and severity tiers,
which are deterministic and fully our own code.

Run from the project root:
    pip install pytest
    pytest tests/ -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))


@pytest.fixture
def detector():
    """A MismatchDetector with the real sentence-transformers model
    replaced by a mock, so tests run offline and deterministically."""
    with patch("mismatch_detector.SentenceTransformer") as mock_st_cls:
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        from mismatch_detector import MismatchDetector
        det = MismatchDetector(use_ollama=False)
        yield det


def _set_similarity(detector, score: float):
    """Helper: force the mocked model's cosine similarity output."""
    with patch("mismatch_detector.util.cos_sim") as mock_cos_sim:
        mock_cos_sim.return_value = MagicMock(__float__=lambda self: score)
        yield mock_cos_sim


class TestSemanticMismatch:
    def test_no_barriers_means_nothing_to_compare(self, detector):
        flagged, score = detector._semantic_mismatch("feeling great", "no barriers this session")
        assert flagged is False
        assert score == 1.0

    def test_low_similarity_flags(self, detector):
        with patch("mismatch_detector.util.cos_sim", return_value=MagicMock(__float__=lambda s: 0.1)):
            flagged, score = detector._semantic_mismatch("feeling great", "relapse into old habits")
        assert flagged is True
        assert score == pytest.approx(0.1)

    def test_high_similarity_does_not_flag(self, detector):
        with patch("mismatch_detector.util.cos_sim", return_value=MagicMock(__float__=lambda s: 0.9)):
            flagged, score = detector._semantic_mismatch("still struggling", "relapse into old habits")
        assert flagged is False


class TestRatingTextMismatch:
    def test_high_rating_with_negative_keyword_flags(self, detector):
        assert detector._rating_text_mismatch(8, "relapse into old coping habits") is True

    def test_high_rating_with_neutral_barrier_does_not_flag(self, detector):
        assert detector._rating_text_mismatch(8, "no barriers this session") is False

    def test_low_rating_never_flags_this_check(self, detector):
        # Even with a negative keyword, a low rating is expected to be
        # consistent with barriers -- this check only targets the
        # high-rating-but-negative-barrier contradiction.
        assert detector._rating_text_mismatch(3, "relapse into old coping habits") is False

    def test_boundary_rating_exactly_at_threshold(self, detector):
        assert detector._rating_text_mismatch(7, "conflict with roommate") is True


class TestRatingDropMismatch:
    """Covers the fix for the confirmed failure-mode-1 gap: a sharp rating
    drop with no barrier text to compare against."""

    def test_no_history_does_not_flag(self, detector):
        flagged, drop = detector._rating_drop_mismatch(2, [])
        assert flagged is False

    def test_sharp_drop_flags(self, detector):
        # rolling avg of [7, 8, 7] = 7.33, current rating 2 -> drop of 5.33
        flagged, drop = detector._rating_drop_mismatch(2, [7, 8, 7])
        assert flagged is True
        assert drop >= 3

    def test_small_fluctuation_does_not_flag(self, detector):
        # rolling avg of [7, 6, 7] = 6.67, current rating 6 -> drop of 0.67
        flagged, drop = detector._rating_drop_mismatch(6, [7, 6, 7])
        assert flagged is False

    def test_uses_only_last_three_ratings(self, detector):
        # Older history (10, 10) should not pull the average up if the
        # three most recent ratings are already low.
        flagged, drop = detector._rating_drop_mismatch(1, [10, 10, 3, 4, 3])
        # rolling avg of last 3 = [3, 4, 3] = 3.33, drop = 2.33 -> below threshold
        assert flagged is False

    def test_rating_increase_never_flags(self, detector):
        flagged, drop = detector._rating_drop_mismatch(9, [3, 4, 3])
        assert flagged is False


class TestCheckSessionIntegration:
    """End-to-end checks on the combined decision + severity logic."""

    def test_clean_session_not_flagged(self, detector):
        with patch("mismatch_detector.util.cos_sim", return_value=MagicMock(__float__=lambda s: 0.9)):
            session = {
                "session_id": "S1", "self_reported_rating": 8,
                "self_reported_text": "doing well", "barriers": "no barriers this session",
            }
            result = detector.check_session(session)
        assert result.is_flagged is False
        assert result.severity == "none"

    def test_rating_drop_alone_is_flagged_and_medium_severity(self, detector):
        session = {
            "session_id": "S2", "self_reported_rating": 2,
            "self_reported_text": "feeling much better this week",
            "barriers": "no barriers this session",
        }
        result = detector.check_session(session, rating_history=[7, 8, 7])
        assert result.is_flagged is True
        assert "rating dropped" in result.reasons[0]
        assert result.severity in ("medium", "high")

    def test_two_agreeing_signals_are_high_severity(self, detector):
        with patch("mismatch_detector.util.cos_sim", return_value=MagicMock(__float__=lambda s: 0.05)):
            session = {
                "session_id": "S3", "self_reported_rating": 8,
                "self_reported_text": "feeling much better",
                "barriers": "relapse into old coping habits under stress",
            }
            result = detector.check_session(session, rating_history=[8, 8, 8])
        # semantic mismatch + rating/text mismatch both fire here
        assert result.is_flagged is True
        assert result.severity == "high"
        assert len(result.reasons) >= 2

    def test_missing_rating_history_still_works(self, detector):
        """A client's very first session has no prior ratings -- the
        detector should degrade gracefully, not error."""
        with patch("mismatch_detector.util.cos_sim", return_value=MagicMock(__float__=lambda s: 0.9)):
            session = {
                "session_id": "S4", "self_reported_rating": 5,
                "self_reported_text": "first session", "barriers": "no barriers this session",
            }
            result = detector.check_session(session)  # no rating_history passed
        assert result.is_flagged is False
