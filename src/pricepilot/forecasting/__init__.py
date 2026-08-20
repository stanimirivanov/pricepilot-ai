"""Forecasting module for demand prediction"""

from pricepilot.forecasting.base import BaseForecaster, ForecastResult
from pricepilot.forecasting.config import ForecastingConfig
from pricepilot.forecasting.statsforecaster import StatsForecastForecaster

__all__ = [
    "BaseForecaster",
    "ForecastResult",
    "ForecastingConfig",
    "StatsForecastForecaster",
]
