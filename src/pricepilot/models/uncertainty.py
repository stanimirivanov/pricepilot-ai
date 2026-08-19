"""Uncertainty quantification using MAPIE (conformal prediction)"""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from mapie.regression import SplitConformalRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


@dataclass
class PredictionInterval:
    """Container for prediction intervals"""

    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    alpha: float  # Significance level (e.g., 0.1 for 90% interval)

    def width(self) -> np.ndarray:
        """Calculate interval width"""
        return self.upper - self.lower

    def coverage(self, actual: np.ndarray) -> float:
        """Calculate empirical coverage"""
        return float(np.mean((actual >= self.lower) & (actual <= self.upper)))


class UncertaintyQuantifier:
    """Uncertainty quantification using MAPIE split conformal prediction"""

    def __init__(
        self,
        alpha: float = 0.1,  # 90% prediction interval
        n_estimators: int = 100,
        random_state: int = 42,
        conformalize_size: float = 0.2,
    ):
        """Initialize uncertainty quantifier

        Args:
            alpha: Significance level (0.1 = 90% prediction interval)
            n_estimators: Number of trees in random forest
            random_state: Random seed for reproducibility
            conformalize_size: Fraction of data for calibration
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        self.conformalize_size = conformalize_size
        self.random_state = random_state

        self.base_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

        # prefit=False allows fit() to train the estimator
        self.mapie_model = SplitConformalRegressor(
            estimator=self.base_model,
            confidence_level=self.confidence_level,
            prefit=False,  # Critical: set to False
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "UncertaintyQuantifier":
        """
        Fit conformal prediction model

        Args:
            x: Feature matrix
            y: Target values

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting uncertainty quantifier (confidence={self.confidence_level})...")

        # Split data into train and calibration sets
        x_train, x_cal, y_train, y_cal = train_test_split(
            x,
            y,
            test_size=self.conformalize_size,
            random_state=self.random_state,
        )

        # Step 1: Fit base estimator on training data
        self.mapie_model.fit(x_train, y_train)

        # Step 2: Conformalize on calibration data
        self.mapie_model.conformalize(x_cal, y_cal)

        self.is_fitted = True
        logger.info("Uncertainty quantifier fitted and conformalized")
        return self

    def predict(
        self,
        x: np.ndarray,
    ) -> PredictionInterval:
        """
        Generate prediction intervals

        Args:
            X: Feature matrix for prediction

        Returns:
            PredictionInterval with lower, upper, and mean predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # predict_interval returns (predictions, intervals)
        y_pred, y_pred_interval = self.mapie_model.predict_interval(x)

        # y_pred_interval shape: (n_samples, 2) for single confidence level
        # or (n_samples, 2, n_confidence_levels) for multiple levels

        if y_pred_interval.ndim == 3:
            # Take first confidence level
            y_pred_interval = y_pred_interval[:, :, 0]

        return PredictionInterval(
            lower=y_pred_interval[:, 0],
            upper=y_pred_interval[:, 1],
            mean=y_pred,
            alpha=self.alpha,
        )

    def calculate_coverage(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Calculate empirical coverage on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Empirical coverage (should be close to 1 - alpha)
        """
        predictions = self.predict(x_test)
        coverage = predictions.coverage(y_test)

        logger.info(f"Empirical coverage: {coverage:.3f} (expected: {self.confidence_level:.3f})")
        return coverage

    def get_interval_width_stats(
        self,
        x: np.ndarray,
    ) -> dict:
        """
        Calculate interval width statistics

        Args:
            X: Feature matrix

        Returns:
            Dictionary with width statistics
        """
        predictions = self.predict(x)
        widths = predictions.width()

        return {
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "min_width": float(np.min(widths)),
            "max_width": float(np.max(widths)),
            "std_width": float(np.std(widths)),
        }


def create_demand_features(
    prices: np.ndarray,
    weather: np.ndarray | None = None,
    day_of_week: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create feature matrix for demand prediction

    Args:
        prices: Array of prices
        weather: Optional weather features
        day_of_week: Optional day of week features

    Returns:
        Feature matrix
    """
    features = [prices.reshape(-1, 1)]

    if weather is not None:
        features.append(weather.reshape(-1, 1))

    if day_of_week is not None:
        n_samples = len(day_of_week)
        day_features = np.zeros((n_samples, 7))
        for i, day in enumerate(day_of_week):
            day_features[i, int(day)] = 1
        features.append(day_features)

    return np.hstack(features)
