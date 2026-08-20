"""Tests for forecasting base classes"""

import numpy as np
import pandas as pd
import pytest

from pricepilot.forecasting.base import BaseForecaster, ForecastResult


def test_forecast_result_creation():
    """Test ForecastResult dataclass"""
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    result = ForecastResult(
        dates=dates,
        mean=np.array([100, 101, 102, 103, 104, 105, 106]),
        lower=np.array([95, 96, 97, 98, 99, 100, 101]),
        upper=np.array([105, 106, 107, 108, 109, 110, 111]),
        model_name="TestModel",
        horizon=7,
    )

    assert len(result.mean) == 7
    assert result.model_name == "TestModel"
    assert result.horizon == 7

    df = result.to_dataframe()
    assert list(df.columns) == ["date", "forecast", "lower_bound", "upper_bound", "model"]
    assert len(df) == 7


def test_forecast_result_summary():
    """Test forecast summary statistics"""
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    result = ForecastResult(
        dates=dates,
        mean=np.array([100, 101, 102, 103, 104, 105, 106]),
        lower=np.array([95, 96, 97, 98, 99, 100, 101]),
        upper=np.array([105, 106, 107, 108, 109, 110, 111]),
        model_name="TestModel",
        horizon=7,
    )

    summary = result.summary()
    assert summary["model"] == "TestModel"
    assert summary["horizon"] == 7
    assert summary["mean_forecast"] == pytest.approx(103.0)
    assert summary["mean_interval_width"] == pytest.approx(10.0)


def test_base_forecaster_is_abstract():
    """Test that BaseForecaster requires abstract method implementation"""
    with pytest.raises(TypeError, match="abstract"):
        # Attempting to instantiate should fail because methods are abstract
        BaseForecaster()  # type: ignore[abstract]


def test_base_forecaster_can_be_subclassed():
    """Test that BaseForecaster can be properly subclassed"""

    class ValidForecaster(BaseForecaster):
        """Concrete implementation for testing"""

        def fit(self, data: pd.DataFrame) -> "ValidForecaster":
            self.is_fitted = True
            return self

        def predict(self, steps: int | None = None) -> ForecastResult:
            if steps is None:
                steps = self.horizon
            dates = pd.date_range("2024-01-01", periods=steps, freq="D")
            return ForecastResult(
                dates=dates,
                mean=np.zeros(steps),
                lower=np.zeros(steps),
                upper=np.zeros(steps),
                model_name="ValidForecaster",
                horizon=steps,
            )

    # Should work without errors
    forecaster = ValidForecaster(horizon=7)
    assert forecaster.horizon == 7
    assert not forecaster.is_fitted

    # Test fit
    test_data = pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=30, freq="D"),
            "y": np.random.randn(30),
        }
    )
    result = forecaster.fit(test_data)
    assert result.is_fitted

    # Test predict
    forecast = forecaster.predict()
    assert len(forecast.mean) == 7
    assert forecast.model_name == "ValidForecaster"


def test_base_forecaster_horizon_validation():
    """Test horizon parameter validation"""
    # Valid horizon
    forecaster = type(
        "TestForecaster",
        (BaseForecaster,),
        {
            "fit": lambda self, data: self,
            "predict": lambda self, steps=None: None,
        },
    )(horizon=14)

    assert forecaster.horizon == 14


def test_fit_predict_method():
    """Test fit_predict convenience method"""

    class TestForecaster(BaseForecaster):
        def fit(self, data: pd.DataFrame) -> "TestForecaster":
            self.is_fitted = True
            return self

        def predict(self, steps: int | None = None) -> ForecastResult:
            if steps is None:
                steps = self.horizon
            dates = pd.date_range("2024-01-01", periods=steps, freq="D")
            return ForecastResult(
                dates=dates,
                mean=np.ones(steps),
                lower=np.zeros(steps),
                upper=np.ones(steps) * 2,
                model_name="TestForecaster",
                horizon=steps,
            )

    forecaster = TestForecaster(horizon=5)
    test_data = pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=30, freq="D"),
            "y": np.random.randn(30),
        }
    )

    result = forecaster.fit_predict(test_data)
    assert result.horizon == 5
    assert len(result.mean) == 5
    assert forecaster.is_fitted
