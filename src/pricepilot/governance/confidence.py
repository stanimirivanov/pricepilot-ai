"""Confidence scoring for pricing decisions"""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pricepilot.governance.state import ConfidenceLevel


@dataclass
class ConfidenceScore:
    """Container for confidence score"""

    score: float  # 0 to 1
    level: ConfidenceLevel
    details: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "score": self.score,
            "level": self.level.value,
            "details": self.details,
        }


class ConfidenceScorer:
    """Calculate confidence scores for pricing decisions"""

    def __init__(
        self,
        high_threshold: float = 0.90,
        medium_threshold: float = 0.70,
    ):
        """
        Initialize confidence scorer

        Args:
            high_threshold: Score >= this = HIGH confidence
            medium_threshold: Score >= this = MEDIUM confidence
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def calculate(
        self,
        forecast_interval_width: float,
        forecasted_demand: float,
        elasticity_std: float,
        elasticity_mean: float,
        anomaly_detected: bool = False,
    ) -> ConfidenceScore:
        """
        Calculate confidence score based on multiple factors

        Args:
            forecast_interval_width: Width of forecast prediction interval
            forecasted_demand: Forecasted demand value
            elasticity_std: Standard deviation of elasticity estimate
            elasticity_mean: Mean of elasticity estimate
            anomaly_detected: Whether anomaly was detected

        Returns:
            ConfidenceScore with level and details
        """
        details = {}

        # Factor 1: Forecast uncertainty
        relative_interval = forecast_interval_width / max(abs(forecasted_demand), 1)
        if relative_interval < 0.15:
            forecast_score = 1.0
        elif relative_interval < 0.30:
            forecast_score = 0.75
        elif relative_interval < 0.50:
            forecast_score = 0.50
        else:
            forecast_score = 0.25

        details["forecast_score"] = forecast_score
        details["relative_interval"] = relative_interval

        # Factor 2: Elasticity uncertainty
        relative_elasticity_std = (
            abs(elasticity_std / elasticity_mean) if elasticity_mean != 0 else 1.0
        )
        if relative_elasticity_std < 0.10:
            elasticity_score = 1.0
        elif relative_elasticity_std < 0.25:
            elasticity_score = 0.75
        elif relative_elasticity_std < 0.50:
            elasticity_score = 0.50
        else:
            elasticity_score = 0.25

        details["elasticity_score"] = elasticity_score
        details["relative_elasticity_std"] = relative_elasticity_std

        # Factor 3: Anomaly penalty
        if anomaly_detected:
            anomaly_penalty = 0.5  # Reduce confidence by 50%
        else:
            anomaly_penalty = 1.0

        details["anomaly_penalty"] = anomaly_penalty

        # Combine scores (weighted average)
        weights = {
            "forecast": 0.40,
            "elasticity": 0.40,
            "anomaly": 0.20,
        }

        combined_score = (
            weights["forecast"] * forecast_score
            + weights["elasticity"] * elasticity_score
            + weights["anomaly"] * anomaly_penalty
        )

        # Normalize to 0-1
        combined_score = min(max(combined_score, 0.0), 1.0)

        # Determine level
        if combined_score >= self.high_threshold:
            level = ConfidenceLevel.HIGH
        elif combined_score >= self.medium_threshold:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        logger.info(
            f"Confidence: score={combined_score:.3f}, level={level.value}, "
            f"forecast={forecast_score:.2f}, elasticity={elasticity_score:.2f}, "
            f"anomaly_penalty={anomaly_penalty:.2f}"
        )

        return ConfidenceScore(
            score=combined_score,
            level=level,
            details=details,
        )
