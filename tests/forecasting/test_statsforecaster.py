"""Tests for StatsForecastForecaster"""

import numpy as np
import pandas as pd
import pytest

from pricepilot.forecasting.config import ForecastingConfig
from pricepilot.forecasting.statsforecaster import StatsForecastForecaster


@pytest.fixture
def daily_demand_series() -> pd.DataFrame:
    """Create daily demand time series with weekly pattern"""
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    day_of_week = dates.dayofweek
    weekend_effect = np.where(day_of_week >= 5, 20, 0)
    seasonal = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.default_rng(42).normal(0, 5, len(dates))

    return pd.DataFrame(
        {
            "ds": dates,
            "y": 100 + weekend_effect + seasonal + noise,
            "unique_id": "test_demand",
        }
    )


@pytest.fixture
def short_series() -> pd.DataFrame:
    """Create short time series for quick tests"""
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    noise = np.random.default_rng(42).normal(0, 3, len(dates))

    return pd.DataFrame(
        {
            "ds": dates,
            "y": 100 + noise,
            "unique_id": "short_test",
        }
    )


def test_forecaster_initialization():
    """Test basic initialization"""
    config = ForecastingConfig(horizon=7, seasonality=7)
    forecaster = StatsForecastForecaster(config=config)

    assert forecaster.horizon == 7
    assert forecaster.config.seasonality == 7
    assert len(forecaster.models) > 0


def test_forecaster_fit(daily_demand_series):
    """Test fitting the forecaster"""
    config = ForecastingConfig(horizon=7, seasonality=7, min_train_size=30)
    forecaster = StatsForecastForecaster(config=config)

    forecaster.fit(daily_demand_series)

    assert forecaster.is_fitted
    assert forecaster.last_date == daily_demand_series["ds"].iloc[-1]


def test_forecaster_predict(daily_demand_series):
    """Test generating forecasts"""
    config = ForecastingConfig(horizon=7, seasonality=7)
    forecaster = StatsForecastForecaster(config=config)
    forecaster.fit(daily_demand_series)

    forecast = forecaster.predict(steps=7)

    assert len(forecast.mean) == 7
    assert len(forecast.lower) == 7
    assert len(forecast.upper) == 7
    assert forecast.horizon == 7
    assert forecast.model_name.startswith("StatsForecast")


def test_forecast_bounds(daily_demand_series):
    """Test that forecast bounds are reasonable"""
    config = ForecastingConfig(horizon=7, seasonality=7)
    forecaster = StatsForecastForecaster(config=config)
    forecaster.fit(daily_demand_series)

    forecast = forecaster.predict(steps=7)

    # Lower bounds should be less than mean
    assert np.all(forecast.lower <= forecast.mean)

    # Upper bounds should be greater than mean
    assert np.all(forecast.upper >= forecast.mean)

    # Bounds should be reasonable (not too wide or narrow)
    mean_width = np.mean(forecast.upper - forecast.lower)
    assert mean_width > 0
    assert mean_width < 100  # Not absurdly wide


def test_forecast_accuracy(daily_demand_series):
    """Test forecast accuracy on held-out data"""
    # Split data
    train_size = int(len(daily_demand_series) * 0.8)
    train_data = daily_demand_series.iloc[:train_size]
    test_data = daily_demand_series.iloc[train_size : train_size + 7]

    # Fit and evaluate
    config = ForecastingConfig(horizon=7, seasonality=7)
    forecaster = StatsForecastForecaster(config=config)
    forecaster.fit(train_data)

    metrics = forecaster.evaluate_accuracy(test_data)

    # Check metrics exist
    assert "mae" in metrics
    assert "rmse" in metrics

    # RMSE should be reasonable (not too high)
    assert metrics["rmse"] < 50

    # MAE should be less than RMSE
    assert metrics["mae"] <= metrics["rmse"]


def test_insufficient_data():
    """Test error with insufficient data"""
    # Create very short series
    short_data = pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=5, freq="D"),
            "y": [100, 101, 102, 103, 104],
            "unique_id": "short",
        }
    )

    config = ForecastingConfig(min_train_size=30)
    forecaster = StatsForecastForecaster(config=config)

    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster.fit(short_data)


def test_missing_columns():
    """Test error with missing columns"""
    bad_data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=30, freq="D"),
            "value": np.random.randn(30),
        }
    )

    forecaster = StatsForecastForecaster()

    with pytest.raises(ValueError, match="must contain"):
        forecaster.fit(bad_data)


def test_predict_before_fit():
    """Test error when predicting before fitting"""
    forecaster = StatsForecastForecaster()

    with pytest.raises(ValueError, match="not fitted"):
        forecaster.predict()


def test_multiple_models(short_series):
    """Test with multiple models"""
    config = ForecastingConfig(horizon=7, seasonality=7, min_train_size=30)
    forecaster = StatsForecastForecaster(
        config=config,
        models=["auto_arima", "seasonal_naive"],
    )

    forecaster.fit(short_series)
    forecast = forecaster.predict(steps=7)

    assert len(forecast.mean) == 7
    assert len(forecaster.models) == 2
