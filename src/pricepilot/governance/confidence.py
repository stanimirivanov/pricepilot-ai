"""Confidence scoring for governance decisions"""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pricepilot.governance.state import ConfidenceLevel


@dataclass
class ConfidenceScore:
    """Container for confidence score results"""

    score: float  # 0.0 to 1.0
    level: ConfidenceLevel
    details: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "score": self.score,
            "level": self.level.value,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"ConfidenceScore(score={self.score:.3f}, level={self.level.value})"


class ConfidenceScorer:
    """Calculate confidence scores for pricing decisions"""

    def __init__(
        self,
        high_threshold: float = 0.90,
        medium_threshold: float = 0.70,
        forecast_weight: float = 0.40,
        elasticity_weight: float = 0.40,
        anomaly_weight: float = 0.20,
    ):
        """
        Initialize confidence scorer

        Args:
            high_threshold: Score >= this = HIGH confidence
            medium_threshold: Score >= this = MEDIUM confidence
            forecast_weight: Weight for forecast uncertainty factor
            elasticity_weight: Weight for elasticity uncertainty factor
            anomaly_weight: Weight for anomaly penalty factor
        """
        # Validate weights
        total_weight = forecast_weight + elasticity_weight + anomaly_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Validate thresholds
        if not (0 < medium_threshold < high_threshold < 1):
            raise ValueError(
                f"Invalid thresholds: must satisfy 0 < medium({medium_threshold}) "
                f"< high({high_threshold}) < 1"
            )

        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.forecast_weight = forecast_weight
        self.elasticity_weight = elasticity_weight
        self.anomaly_weight = anomaly_weight

        logger.debug(
            f"ConfidenceScorer initialized: high={high_threshold}, "
            f"medium={medium_threshold}, weights=({forecast_weight}, "
            f"{elasticity_weight}, {anomaly_weight})"
        )

    def _score_forecast_uncertainty(
        self,
        forecast_interval_width: float,
        forecasted_demand: float,
    ) -> tuple[float, dict[str, float]]:
        """
        Score forecast uncertainty based on interval width

        Args:
            forecast_interval_width: Width of prediction interval
            forecasted_demand: Forecasted demand value

        Returns:
            Tuple of (score, details)
        """
        # Relative interval width
        relative_interval = forecast_interval_width / max(abs(forecasted_demand), 1.0)

        # Score based on relative interval
        if relative_interval < 0.10:
            score = 1.0  # Very narrow interval
        elif relative_interval < 0.20:
            score = 0.85  # Narrow
        elif relative_interval < 0.30:
            score = 0.70  # Moderate
        elif relative_interval < 0.40:
            score = 0.55  # Wide
        elif relative_interval < 0.50:
            score = 0.40  # Very wide
        else:
            score = 0.25  # Extremely wide

        details = {
            "relative_interval": relative_interval,
            "forecast_score": score,
        }

        return score, details

    def _score_elasticity_uncertainty(
        self,
        elasticity_std: float,
        elasticity_mean: float,
    ) -> tuple[float, dict[str, float]]:
        """
        Score elasticity uncertainty based on posterior distribution

        Args:
            elasticity_std: Standard deviation of elasticity estimate
            elasticity_mean: Mean of elasticity estimate

        Returns:
            Tuple of (score, details)
        """
        # Relative standard deviation (coefficient of variation)
        if elasticity_mean != 0:
            relative_std = abs(elasticity_std / elasticity_mean)
        else:
            relative_std = 1.0  # Maximum uncertainty

        # Score based on relative std
        if relative_std < 0.05:
            score = 1.0  # Very precise estimate
        elif relative_std < 0.10:
            score = 0.90  # Precise
        elif relative_std < 0.20:
            score = 0.75  # Moderate
        elif relative_std < 0.30:
            score = 0.60  # Uncertain
        elif relative_std < 0.50:
            score = 0.40  # Very uncertain
        else:
            score = 0.20  # Extremely uncertain

        details = {
            "relative_elasticity_std": relative_std,
            "elasticity_score": score,
        }

        return score, details

    def _score_anomaly_penalty(
        self,
        anomaly_detected: bool,
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate anomaly penalty factor

        Args:
            anomaly_detected: Whether anomaly was detected

        Returns:
            Tuple of (penalty, details)
        """
        if anomaly_detected:
            penalty = 0.0  # Complete penalty - very low confidence
        else:
            penalty = 1.0  # No penalty

        details = {
            "anomaly_penalty": penalty,
        }

        return penalty, details

    def calculate(
        self,
        forecast_interval_width: float,
        forecasted_demand: float,
        elasticity_std: float,
        elasticity_mean: float,
        anomaly_detected: bool = False,
        additional_factors: dict[str, float] | None = None,
    ) -> ConfidenceScore:
        """
        Calculate confidence score based on multiple factors

        Args:
            forecast_interval_width: Width of forecast prediction interval
            forecasted_demand: Forecasted demand value
            elasticity_std: Standard deviation of elasticity estimate
            elasticity_mean: Mean of elasticity estimate
            anomaly_detected: Whether anomaly was detected
            additional_factors: Optional additional scoring factors

        Returns:
            ConfidenceScore with level and details
        """
        all_details = {}

        # Factor 1: Forecast uncertainty
        forecast_score, forecast_details = self._score_forecast_uncertainty(
            forecast_interval_width,
            forecasted_demand,
        )
        all_details.update(forecast_details)

        # Factor 2: Elasticity uncertainty
        elasticity_score, elasticity_details = self._score_elasticity_uncertainty(
            elasticity_std,
            elasticity_mean,
        )
        all_details.update(elasticity_details)

        # Factor 3: Anomaly penalty
        anomaly_penalty, anomaly_details = self._score_anomaly_penalty(
            anomaly_detected,
        )
        all_details.update(anomaly_details)

        # Combine weighted scores
        combined_score = (
            self.forecast_weight * forecast_score
            + self.elasticity_weight * elasticity_score
            + self.anomaly_weight * anomaly_penalty
        )

        # Apply additional factors if provided
        if additional_factors:
            for factor_name, factor_score in additional_factors.items():
                all_details[f"additional_{factor_name}"] = factor_score
                # Additional factors can modify the combined score
                combined_score *= factor_score

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
            f"Confidence: score={combined_score:.3f}, level={level.value} "
            f"(forecast={forecast_score:.2f}, elasticity={elasticity_score:.2f}, "
            f"anomaly={anomaly_penalty:.2f})"
        )

        return ConfidenceScore(
            score=combined_score,
            level=level,
            details=all_details,
        )

    def is_high_confidence(self, score: float) -> bool:
        """Check if score represents high confidence"""
        return score >= self.high_threshold

    def is_medium_confidence(self, score: float) -> bool:
        """Check if score represents medium confidence"""
        return self.medium_threshold <= score < self.high_threshold

    def is_low_confidence(self, score: float) -> bool:
        """Check if score represents low confidence"""
        return score < self.medium_threshold


class ConfidenceThresholds:
    """Pre-defined confidence threshold configurations"""

    @staticmethod
    def conservative() -> ConfidenceScorer:
        """
        Conservative thresholds (require high confidence for approval)

        Use when:
        - High risk tolerance is low
        - Regulatory compliance required
        - Limited historical data
        """
        return ConfidenceScorer(
            high_threshold=0.95,
            medium_threshold=0.80,
        )

    @staticmethod
    def balanced() -> ConfidenceScorer:
        """
        Balanced thresholds (default)

        Use when:
        - Standard business conditions
        - Moderate risk tolerance
        - Established pricing models
        """
        return ConfidenceScorer(
            high_threshold=0.90,
            medium_threshold=0.70,
        )

    @staticmethod
    def aggressive() -> ConfidenceScorer:
        """
        Aggressive thresholds (approve with lower confidence)

        Use when:
        - High risk tolerance
        - Experimental phase
        - Quick iteration desired
        """
        return ConfidenceScorer(
            high_threshold=0.80,
            medium_threshold=0.60,
        )

    @staticmethod
    def custom(
        high_threshold: float,
        medium_threshold: float,
        forecast_weight: float = 0.40,
        elasticity_weight: float = 0.40,
        anomaly_weight: float = 0.20,
    ) -> ConfidenceScorer:
        """
        Custom threshold configuration

        Args:
            high_threshold: Score >= this = HIGH confidence
            medium_threshold: Score >= this = MEDIUM confidence
            forecast_weight: Weight for forecast uncertainty
            elasticity_weight: Weight for elasticity uncertainty
            anomaly_weight: Weight for anomaly penalty

        Returns:
            Configured ConfidenceScorer
        """
        return ConfidenceScorer(
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
            forecast_weight=forecast_weight,
            elasticity_weight=elasticity_weight,
            anomaly_weight=anomaly_weight,
        )
