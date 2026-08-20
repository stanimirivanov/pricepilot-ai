"""Configuration for forecasting models"""

from dataclasses import dataclass


@dataclass
class ForecastingConfig:
    """Configuration for time series forecasting"""

    horizon: int = 7  # Days to forecast
    seasonality: int = 7  # Weekly seasonality
    confidence_level: float = 0.9  # For prediction intervals
    min_train_size: int = 30  # Minimum days required for training

    # Model-specific parameters
    arima_order: tuple = (2, 0, 2)  # ARIMA(p,d,q)
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True

    def validate(self) -> None:
        """Validate configuration"""
        if self.horizon < 1:
            raise ValueError("Horizon must be at least 1 day")
        if self.seasonality < 1:
            raise ValueError("Seasonality must be at least 1 day")
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.min_train_size < 10:
            raise ValueError("Minimum training size must be at least 10 days")
