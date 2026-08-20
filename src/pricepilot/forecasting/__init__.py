# src/pricepilot/forecasting/__init__.py
"""Forecasting module for demand prediction"""

from pricepilot.forecasting.base import BaseForecaster, ForecastResult
from pricepilot.forecasting.config import ForecastingConfig

__all__ = ["BaseForecaster", "ForecastResult", "ForecastingConfig"]
