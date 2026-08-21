"""Anomaly detection for demand patterns using PyOD"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyResult:
    """Container for anomaly detection results"""

    dates: pd.DatetimeIndex
    scores: np.ndarray
    labels: np.ndarray  # 1 = anomaly, 0 = normal
    threshold: float
    n_anomalies: int
    total_points: int

    def anomaly_percentage(self) -> float:
        """Percentage of anomalies detected"""
        return (self.n_anomalies / self.total_points) * 100 if self.total_points > 0 else 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame(
            {
                "date": self.dates,
                "anomaly_score": self.scores,
                "is_anomaly": self.labels.astype(bool),
            }
        )

    def get_anomaly_dates(self) -> pd.DatetimeIndex:
        """Get dates where anomalies were detected"""
        return self.dates[self.labels == 1]


class DemandAnomalyDetector:
    """Anomaly detection for demand time series"""

    def __init__(
        self,
        detector_type: str = "isolation_forest",
        contamination: float = 0.05,  # Expected fraction of anomalies
        random_state: int = 42,
        window_size: int = 7,  # Rolling window for feature creation
    ):
        """
        Initialize anomaly detector

        Args:
            detector_type: Type of detector ("isolation_forest", "knn", "lof")
            contamination: Expected fraction of anomalies (0.05 = 5%)
            random_state: Random seed for reproducibility
            window_size: Rolling window size for feature engineering
        """
        self.detector_type = detector_type
        self.contamination = contamination
        self.random_state = random_state
        self.window_size = window_size

        self.detector = self._create_detector(detector_type, contamination)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.threshold: float | None = None

    def _create_detector(self, detector_type: str, contamination: float):
        """Create PyOD detector instance"""
        if detector_type == "isolation_forest":
            return IForest(
                contamination=contamination,
                random_state=self.random_state,
                n_estimators=100,
            )
        elif detector_type == "knn":
            return KNN(
                contamination=contamination,
                n_neighbors=5,
            )
        elif detector_type == "lof":
            return LOF(
                contamination=contamination,
                n_neighbors=20,
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def _create_features(self, demand: np.ndarray) -> np.ndarray:
        """
        Create features for anomaly detection

        Features include:
        - Current demand value
        - Rolling mean
        - Rolling std
        - Difference from rolling mean
        - Rate of change

        Args:
            demand: Array of demand values

        Returns:
            Feature matrix
        """
        n = len(demand)
        features = []

        # Current value
        features.append(demand)

        # Rolling statistics
        for window in [3, 7, 14]:
            if n >= window:
                rolling_mean = pd.Series(demand).rolling(window=window, min_periods=1).mean().values
                rolling_std = pd.Series(demand).rolling(window=window, min_periods=1).std().values
                features.append(rolling_mean)
                features.append(rolling_std)

                # Difference from rolling mean
                diff_from_mean = demand - rolling_mean
                features.append(diff_from_mean)

        # Rate of change
        if n >= 2:
            rate_of_change = np.diff(demand, prepend=demand[0])
            features.append(rate_of_change)

        # Day of week (if available)
        if n >= 7:
            day_of_week = np.arange(n) % 7
            features.append(day_of_week.astype(float))

        return np.column_stack(features)

    def fit(self, demand: np.ndarray) -> "DemandAnomalyDetector":
        """
        Fit anomaly detector to demand data

        Args:
            demand: Array of demand values

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting {self.detector_type} anomaly detector...")

        # Create features
        features = self._create_features(demand)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit detector
        self.detector.fit(features_scaled)

        # Calculate threshold
        scores = self.detector.decision_function(features_scaled)
        self.threshold = np.percentile(scores, (1 - self.contamination) * 100)

        self.is_fitted = True
        logger.info(f"Anomaly detector fitted. Threshold: {self.threshold:.3f}")

        return self

    def detect(
        self,
        demand: np.ndarray,
        dates: pd.DatetimeIndex | None = None,
    ) -> AnomalyResult:
        """
        Detect anomalies in demand data

        Args:
            demand: Array of demand values
            dates: Optional dates for each demand value

        Returns:
            AnomalyResult with scores and labels
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        if dates is None:
            dates = pd.DatetimeIndex([f"Day {i + 1}" for i in range(len(demand))])

        # Create features
        features = self._create_features(demand)
        features_scaled = self.scaler.transform(features)

        # Get anomaly scores
        scores = self.detector.decision_function(features_scaled)

        # Label anomalies
        labels = self.detector.predict(features_scaled)

        n_anomalies = int(np.sum(labels))

        result = AnomalyResult(
            dates=pd.DatetimeIndex(dates),
            scores=scores,
            labels=labels,
            threshold=self.threshold if self.threshold is not None else 0.0,
            n_anomalies=n_anomalies,
            total_points=len(demand),
        )

        logger.info(f"Detected {n_anomalies} anomalies ({result.anomaly_percentage():.1f}%)")

        return result

    def fit_detect(
        self,
        demand: np.ndarray,
        dates: pd.DatetimeIndex | None = None,
    ) -> AnomalyResult:
        """Fit and detect in one step"""
        self.fit(demand)
        return self.detect(demand, dates)

    def is_anomaly(self, demand_value: float, history: np.ndarray) -> bool:
        """
        Check if a single demand value is anomalous given history

        Args:
            demand_value: Single demand value to check
            history: Historical demand values

        Returns:
            True if anomalous, False otherwise
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        # Combine history with new value
        combined = np.append(history, demand_value)

        # Detect anomalies in combined data
        result = self.detect(combined)

        # Check if the last point is anomalous
        return bool(result.labels[-1] == 1)

    def plot_anomalies(
        self,
        demand: np.ndarray,
        dates: pd.DatetimeIndex | None = None,
        save_path: str | None = None,
    ) -> None:
        """Plot demand with anomalies highlighted"""
        import matplotlib.pyplot as plt

        if dates is None:
            dates = pd.DatetimeIndex([f"Day {i + 1}" for i in range(len(demand))])

        result = self.detect(demand, dates)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Demand with anomalies
        axes[0].plot(dates, demand, label="Demand", color="blue", linewidth=1.5)

        # Highlight anomalies
        anomaly_dates = result.get_anomaly_dates()
        anomaly_values = demand[result.labels == 1]

        axes[0].scatter(
            anomaly_dates,
            anomaly_values,
            color="red",
            s=100,
            label=f"Anomalies ({result.n_anomalies})",
            zorder=5,
        )

        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Demand")
        axes[0].set_title("Demand with Anomaly Detection")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Anomaly scores
        axes[1].plot(dates, result.scores, label="Anomaly Score", color="purple", linewidth=1.5)
        axes[1].axhline(
            result.threshold,
            color="red",
            linestyle="--",
            label=f"Threshold ({result.threshold:.2f})",
        )
        axes[1].fill_between(
            dates,
            result.threshold,
            result.scores,
            where=result.scores > result.threshold,
            alpha=0.3,
            color="red",
            label="Anomaly Region",
        )
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Anomaly Score")
        axes[1].set_title("Anomaly Scores Over Time")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved anomaly plot to {save_path}")
        else:
            plt.show()


class AnomalyAwarePricingModel:
    """Integrates anomaly detection with forecast pricing"""

    def __init__(
        self,
        pricing_model,
        anomaly_detector: DemandAnomalyDetector,
    ):
        """
        Initialize anomaly-aware pricing

        Args:
            pricing_model: ForecastPricingModel instance
            anomaly_detector: DemandAnomalyDetector instance
        """
        self.pricing_model = pricing_model
        self.anomaly_detector = anomaly_detector

    def price_with_anomaly_check(
        self,
        current_price: float,
        historical_demand: np.ndarray,
        forecasted_demand: float,
    ) -> dict[str, Any]:
        """
        Calculate price while checking for anomalies

        Args:
            current_price: Current price
            historical_demand: Historical demand values
            forecasted_demand: Forecasted demand for tomorrow

        Returns:
            Dictionary with pricing decision and anomaly status
        """
        # Check if forecasted demand is anomalous
        is_anomaly = self.anomaly_detector.is_anomaly(
            forecasted_demand,
            historical_demand,
        )

        # Get pricing result
        pricing_result = self.pricing_model.price_for_tomorrow(current_price)

        result = {
            "pricing_result": pricing_result,
            "is_anomaly": is_anomaly,
            "anomaly_status": "ANOMALY" if is_anomaly else "NORMAL",
            "forecasted_demand": forecasted_demand,
        }

        if is_anomaly:
            logger.warning(f"Anomaly detected in forecast: demand={forecasted_demand:.0f}")

        return result
