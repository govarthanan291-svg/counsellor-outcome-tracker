"""
tests/test_mismatch_detector.py

Automated tests for the core AI decision logic in prototype/mismatch_detector.py.

These tests do not download sentence-transformers (the semantic check is
off by default and the model is loaded lazily). They focus on the
flagging rules and severity tiers, which are deterministic.

Run from the project root:
    pip install pytest
    pytest tests/ -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))

import mismatch_detector as md
from mismatch_detector import MismatchDetector


@pytest.fixture
def detector():
    return MismatchDetector(use_ollama=False)


class TestSemanticMismatch:
    def test_no_barriers_means_nothing_to_compare(self, detector):
        flagged, score = detector._semantic_mismatch("feeling great", "no barriers this session")
        assert flagged is False
        assert score == 1.0

    def test_low_similarity_flags(self, detector):
        detector._model = MagicMock()
        detector._model.encode.return_value = [0, 1]
        with patch.object(md, "util") as mock_util:
            mock_util.cos_sim.return_value = 0.05
            flagged, score = detector._semantic_mismatch("feeling great", "relapse into old habits")
        assert flagged is True
        assert score == pytest.approx(0.05)

    def test_high_similarity_does_not_flag(self, detector):
        detector._model = MagicMock()
        detector._model.encode.return_value = [0, 1]
        with patch.object(md, "util") as mock_util:
            mock_util.cos_sim.return_value = 0.9
            flagged, score = detector._semantic_mismatch("still struggling", "relapse into old habits")
        assert flagged is False


class TestRatingTextMismatch:
    def test_high_rating_with_negative_keyword_flags(self, detector):
        assert detector._rating_text_mismatch(8, "relapse into old coping habits") is True

    def test_high_rating_with_neutral_barrier_does_not_flag(self, detector):
        assert detector._rating_text_mismatch(8, "no barriers this session") is False

    def test_low_rating_never_flags_this_check(self, detector):
        assert detector._rating_text_mismatch(3, "relapse into old coping habits") is False

    def test_boundary_rating_exactly_at_threshold(self, detector):
        assert detector._rating_text_mismatch(7, "conflict with roommate") is True

    def test_rating_just_below_keyword_threshold(self, detector):
        assert detector._rating_text_mismatch(6, "conflict with roommate") is False


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

    def test_drop_exactly_at_threshold(self, detector):
        flagged, drop = detector._rating_drop_mismatch(3, [6, 6, 6])
        assert flagged is True
        assert drop == pytest.approx(3.0)

    def test_drop_just_below_threshold(self, detector):
        # avg 6.0, rating 4 -> drop 2.0
        flagged, drop = detector._rating_drop_mismatch(4, [6, 6, 6])
        assert flagged is False
        assert drop == pytest.approx(2.0)

    def test_small_fluctuation_does_not_flag(self, detector):
        flagged, drop = detector._rating_drop_mismatch(6, [7, 6, 7])
        assert flagged is False

    def test_uses_only_last_three_ratings(self, detector):
        flagged, drop = detector._rating_drop_mismatch(1, [10, 10, 3, 4, 3])
        # rolling avg of last 3 = [3, 4, 3] = 3.33, drop = 2.33 -> below threshold
        assert flagged is False

    def test_rating_increase_never_flags(self, detector):
        flagged, drop = detector._rating_drop_mismatch(9, [3, 4, 3])
        assert flagged is False

    def test_single_prior_rating_still_computes_average(self, detector):
        flagged, drop = detector._rating_drop_mismatch(2, [8])
        assert flagged is True
        assert drop == pytest.approx(6.0)


class TestSeverityTiers:
    """Explicit low / medium / high boundaries. See `_severity` in
    mismatch_detector.py for the mapping."""

    def test_unflagged_is_none(self, detector):
        session = {
            "session_id": "SEV0",
            "self_reported_rating": 8,
            "self_reported_text": "doing well",
            "barriers": "no barriers this session",
        }
        result = detector.check_session(session)
        assert result.is_flagged is False
        assert result.severity == "none"

    def test_keyword_only_is_low(self, detector):
        session = {
            "session_id": "SEV_LOW",
            "self_reported_rating": 8,
            "self_reported_text": "feeling much better",
            "barriers": "relapse into old coping habits under stress",
        }
        result = detector.check_session(session)  # no history → no drop signal
        assert result.is_flagged is True
        assert result.severity == "low"
        assert len(result.reasons) == 1

    def test_modest_drop_only_is_medium(self, detector):
        # avg 6.0, rating 3 → drop 3.0, exactly RATING_DROP_THRESHOLD, below LARGE_DROP
        session = {
            "session_id": "SEV_MED",
            "self_reported_rating": 3,
            "self_reported_text": "feeling much better this week",
            "barriers": "no barriers this session",
        }
        result = detector.check_session(session, rating_history=[6, 6, 6])
        assert result.is_flagged is True
        assert result.severity == "medium"
        assert len(result.reasons) == 1

    def test_drop_just_below_large_threshold_stays_medium(self, detector):
        # avg 8.0, rating 4 → drop 4.0 (in [3, 5))
        session = {
            "session_id": "SEV_MED2",
            "self_reported_rating": 4,
            "self_reported_text": "ok this week",
            "barriers": "no barriers this session",
        }
        result = detector.check_session(session, rating_history=[8, 8, 8])
        assert result.severity == "medium"

    def test_drop_exactly_five_is_high(self, detector):
        # avg 8.0, rating 3 → drop 5.0
        session = {
            "session_id": "SEV_HIGH_DROP",
            "self_reported_rating": 3,
            "self_reported_text": "feeling much better this week",
            "barriers": "no barriers this session",
        }
        result = detector.check_session(session, rating_history=[8, 8, 8])
        assert result.is_flagged is True
        assert result.severity == "high"
        assert len(result.reasons) == 1

    def test_two_agreeing_signals_are_high(self, detector):
        # rating 7 fires the keyword check AND is a 3-point drop from 10s
        session = {
            "session_id": "SEV_HIGH_BOTH",
            "self_reported_rating": 7,
            "self_reported_text": "feeling much better",
            "barriers": "relapse into old coping habits under stress",
        }
        result = detector.check_session(session, rating_history=[10, 10, 10])
        assert result.is_flagged is True
        assert result.severity == "high"
        assert len(result.reasons) >= 2


class TestSemanticCheckDisabled:
    def test_semantic_method_is_not_called_when_flag_is_false(self, detector):
        assert md.USE_SEMANTIC_CHECK is False
        session = {
            "session_id": "SEM_OFF",
            "self_reported_rating": 8,
            "self_reported_text": "feeling great",
            "barriers": "relapse into old habits",
        }
        with patch.object(detector, "_semantic_mismatch") as mock_sem:
            detector.check_session(session)
        mock_sem.assert_not_called()

    def test_model_is_not_loaded_on_init_or_check(self, detector):
        session = {
            "session_id": "SEM_LAZY",
            "self_reported_rating": 5,
            "self_reported_text": "first session",
            "barriers": "no barriers this session",
        }
        with patch.object(detector, "_load_model") as mock_load:
            detector.check_session(session)
        mock_load.assert_not_called()
        assert detector._model is None

    def test_semantic_method_is_called_when_flag_is_forced_on(self, detector):
        session = {
            "session_id": "SEM_ON",
            "self_reported_rating": 5,
            "self_reported_text": "feeling great",
            "barriers": "roommate conflict",
        }
        with patch.object(md, "USE_SEMANTIC_CHECK", True):
            with patch.object(
                detector, "_semantic_mismatch", return_value=(False, 0.42)
            ) as mock_sem:
                result = detector.check_session(session)
        mock_sem.assert_called_once_with("feeling great", "roommate conflict")
        assert result.similarity_score == pytest.approx(0.42)


class TestCheckSessionIntegration:
    def test_clean_session_not_flagged(self, detector):
        session = {
            "session_id": "S1", "self_reported_rating": 8,
            "self_reported_text": "doing well", "barriers": "no barriers this session",
        }
        result = detector.check_session(session)
        assert result.is_flagged is False
        assert result.severity == "none"

    def test_large_rating_drop_alone_is_high(self, detector):
        session = {
            "session_id": "S2", "self_reported_rating": 2,
            "self_reported_text": "feeling much better this week",
            "barriers": "no barriers this session",
        }
        result = detector.check_session(session, rating_history=[7, 8, 7])
        assert result.is_flagged is True
        assert "rating dropped" in result.reasons[0]
        assert result.severity == "high"

    def test_missing_rating_history_still_works(self, detector):
        session = {
            "session_id": "S4", "self_reported_rating": 5,
            "self_reported_text": "first session", "barriers": "no barriers this session",
        }
        result = detector.check_session(session)
        assert result.is_flagged is False

    def test_ollama_off_does_not_call_network(self, detector):
        session = {
            "session_id": "S5",
            "self_reported_rating": 8,
            "self_reported_text": "fine",
            "barriers": "relapse",
        }
        with patch.object(md.requests, "post") as mock_post:
            result = detector.check_session(session)
        mock_post.assert_not_called()
        assert result.explanation is None

    def test_ollama_failure_is_cached_and_not_retried(self):
        det = MismatchDetector(use_ollama=True)
        session = {
            "session_id": "S6",
            "self_reported_rating": 8,
            "self_reported_text": "fine",
            "barriers": "relapse",
        }
        with patch.object(md.requests, "post", side_effect=md.requests.ConnectionError("down")) as mock_post:
            first = det.check_session(session)
            second = det.check_session({**session, "session_id": "S7"})
        assert mock_post.call_count == 1
        assert first.explanation == md.OLLAMA_UNAVAILABLE_NOTE
        assert second.explanation == md.OLLAMA_UNAVAILABLE_NOTE
        assert det._ollama_reachable is False
