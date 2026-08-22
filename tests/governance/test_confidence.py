"""Tests for confidence scoring"""

import pytest

from pricepilot.governance.confidence import ConfidenceScorer
from pricepilot.governance.state import ConfidenceLevel


@pytest.fixture
def scorer():
    """Create confidence scorer"""
    return ConfidenceScorer(
        high_threshold=0.90,
        medium_threshold=0.70,
    )


def test_high_confidence(scorer):
    """Test high confidence scenario"""
    score = scorer.calculate(
        forecast_interval_width=10,  # Narrow interval
        forecasted_demand=100,
        elasticity_std=0.1,  # Low uncertainty
        elasticity_mean=-2.0,
        anomaly_detected=False,
    )

    assert score.score >= 0.90
    assert score.level == ConfidenceLevel.HIGH


def test_low_confidence(scorer):
    """Test low confidence scenario"""
    score = scorer.calculate(
        forecast_interval_width=80,  # Very wide interval
        forecasted_demand=100,
        elasticity_std=1.5,  # High uncertainty
        elasticity_mean=-2.0,
        anomaly_detected=True,  # Anomaly detected
    )

    assert score.score < 0.70
    assert score.level == ConfidenceLevel.LOW


def test_medium_confidence(scorer):
    """Test medium confidence scenario"""
    score = scorer.calculate(
        forecast_interval_width=25,  # Moderate interval
        forecasted_demand=100,
        elasticity_std=0.3,  # Moderate uncertainty
        elasticity_mean=-2.0,
        anomaly_detected=False,
    )

    assert 0.70 <= score.score < 0.90
    assert score.level == ConfidenceLevel.MEDIUM


def test_anomaly_penalty(scorer):
    """Test that anomaly reduces confidence"""
    base_score = scorer.calculate(
        forecast_interval_width=10,
        forecasted_demand=100,
        elasticity_std=0.1,
        elasticity_mean=-2.0,
        anomaly_detected=False,
    )

    anomaly_score = scorer.calculate(
        forecast_interval_width=10,
        forecasted_demand=100,
        elasticity_std=0.1,
        elasticity_mean=-2.0,
        anomaly_detected=True,
    )

    assert anomaly_score.score < base_score.score


def test_score_details(scorer):
    """Test that score details are provided"""
    score = scorer.calculate(
        forecast_interval_width=20,
        forecasted_demand=100,
        elasticity_std=0.2,
        elasticity_mean=-2.0,
        anomaly_detected=False,
    )

    assert "forecast_score" in score.details
    assert "elasticity_score" in score.details
    assert "anomaly_penalty" in score.details
    assert "relative_interval" in score.details
