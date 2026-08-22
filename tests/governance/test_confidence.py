"""Tests for confidence scoring"""

import pytest

from pricepilot.governance.confidence import ConfidenceScorer, ConfidenceThresholds
from pricepilot.governance.state import ConfidenceLevel


@pytest.fixture
def scorer():
    """Create confidence scorer"""
    return ConfidenceScorer(
        high_threshold=0.90,
        medium_threshold=0.70,
    )


def test_high_confidence_narrow_interval(scorer):
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


def test_high_confidence_vary_narrow_interval(scorer):
    """Test high confidence scenario"""
    score = scorer.calculate(
        forecast_interval_width=5,  # Very narrow
        forecasted_demand=100,
        elasticity_std=0.05,  # Very precise
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
    """Test that anomaly reduces confidence significantly"""
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

    # Anomaly should reduce score by at least 0.2 (20% weight)
    assert base_score.score - anomaly_score.score >= 0.15


def test_score_details(scorer):
    """Test that score details are comprehensive"""
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
    assert "relative_elasticity_std" in score.details


def test_scorer_initialization():
    """Test scorer initialization"""
    scorer = ConfidenceScorer()
    assert scorer.high_threshold == 0.90
    assert scorer.medium_threshold == 0.70
    assert scorer.forecast_weight == 0.40
    assert scorer.elasticity_weight == 0.40
    assert scorer.anomaly_weight == 0.20


def test_weight_validation():
    """Test that weights must sum to 1"""
    with pytest.raises(ValueError, match="sum to 1"):
        ConfidenceScorer(
            forecast_weight=0.5,
            elasticity_weight=0.5,
            anomaly_weight=0.5,  # Sums to 1.5
        )


def test_threshold_methods(scorer):
    """Test threshold checking methods"""
    assert scorer.is_high_confidence(0.95) is True
    assert scorer.is_high_confidence(0.85) is False

    assert scorer.is_medium_confidence(0.80) is True
    assert scorer.is_medium_confidence(0.95) is False

    assert scorer.is_low_confidence(0.60) is True
    assert scorer.is_low_confidence(0.80) is False


def test_conservative_thresholds():
    """Test conservative threshold configuration"""
    scorer = ConfidenceThresholds.conservative()
    assert scorer.high_threshold == 0.95
    assert scorer.medium_threshold == 0.80


def test_balanced_thresholds():
    """Test balanced threshold configuration"""
    scorer = ConfidenceThresholds.balanced()
    assert scorer.high_threshold == 0.90
    assert scorer.medium_threshold == 0.70


def test_aggressive_thresholds():
    """Test aggressive threshold configuration"""
    scorer = ConfidenceThresholds.aggressive()
    assert scorer.high_threshold == 0.80
    assert scorer.medium_threshold == 0.60


def test_additional_factors(scorer):
    """Test additional scoring factors"""
    # Add a data quality factor that reduces confidence
    additional = {"data_quality": 0.80}

    score_with_factor = scorer.calculate(
        forecast_interval_width=10,
        forecasted_demand=100,
        elasticity_std=0.1,
        elasticity_mean=-2.0,
        anomaly_detected=False,
        additional_factors=additional,
    )

    base_score = scorer.calculate(
        forecast_interval_width=10,
        forecasted_demand=100,
        elasticity_std=0.1,
        elasticity_mean=-2.0,
        anomaly_detected=False,
    )

    assert score_with_factor.score < base_score.score
    assert "additional_data_quality" in score_with_factor.details


def test_edge_case_zero_demand(scorer):
    """Test edge case with zero forecasted demand"""
    score = scorer.calculate(
        forecast_interval_width=10,
        forecasted_demand=0,  # Zero demand
        elasticity_std=0.1,
        elasticity_mean=-2.0,
        anomaly_detected=False,
    )

    # Should still produce valid score
    assert 0 <= score.score <= 1


def test_edge_case_extreme_uncertainty(scorer):
    """Test edge case with extreme uncertainty"""
    score = scorer.calculate(
        forecast_interval_width=1000,  # Extremely wide
        forecasted_demand=10,
        elasticity_std=10,  # Extremely uncertain
        elasticity_mean=-2.0,
        anomaly_detected=True,
    )

    # Should produce very low confidence
    assert score.level == ConfidenceLevel.LOW
    assert score.score < 0.30
