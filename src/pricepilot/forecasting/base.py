"""Base classes for time series forecasting"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Container for forecast results"""

    dates: pd.DatetimeIndex
    mean: np.ndarray
    lower: np.ndarray  # Lower prediction interval
    upper: np.ndarray  # Upper prediction interval
    model_name: str
    horizon: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame(
            {
                "date": self.dates,
                "forecast": self.mean,
                "lower_bound": self.lower,
                "upper_bound": self.upper,
                "model": self.model_name,
            }
        )

    def summary(self) -> dict:
        """Return summary statistics"""
        return {
            "model": self.model_name,
            "horizon": self.horizon,
            "mean_forecast": float(np.mean(self.mean)),
            "std_forecast": float(np.std(self.mean)),
            "mean_interval_width": float(np.mean(self.upper - self.lower)),
        }


class BaseForecaster(ABC):
    """Abstract base class for forecasters"""

    def __init__(self, horizon: int = 7):
        self.horizon = horizon
        self.is_fitted = False
        self.model_name = self.__class__.__name__

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseForecaster":
        """Fit forecaster to historical data"""
        pass

    @abstractmethod
    def predict(self, steps: int | None = None) -> ForecastResult:
        """Generate forecast for future periods"""
        pass

    def fit_predict(self, data: pd.DataFrame, steps: int | None = None) -> ForecastResult:
        """Fit and predict in one step"""
        self.fit(data)
        return self.predict(steps)
