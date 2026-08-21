"""Tests for forecast-integrated pricing model"""

import numpy as np
import pandas as pd
import pytest
from pricepilot.models.forecast_pricing import ForecastPricingModel, ForecastPricingResult

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.forecasting.config import ForecastingConfig
from pricepilot.forecasting.statsforecaster import StatsForecastForecaster
from pricepilot.models.elasticity import PriceElasticityModel
from pricepilot.optimization.price_optimizer import OptimizationConfig


@pytest.fixture
def synthetic_data():
    """Generate synthetic data"""
    config = DataGeneratorConfig(
        start_date="2022-01-01",
        end_date="2023-12-31",
        seed=42,
    )
    generator = CarWashDataGenerator(config)
    return generator.generate()


@pytest.fixture
def elasticity_model(synthetic_data):
    """Fit elasticity model"""
    model = PriceElasticityModel(
        samples=500,
        tune=200,
        chains=2,
    )
    model.fit(
        prices=synthetic_data["price"].values,
        demand=synthetic_data["quantity_sold"].values,
        weather_features=synthetic_data["is_sunny"].values,
        progressbar=False,
    )
    return model


@pytest.fixture
def forecaster(synthetic_data):
    """Fit forecaster"""
    ts_data = pd.DataFrame(
        {
            "ds": synthetic_data["date"],
            "y": synthetic_data["quantity_sold"],
            "unique_id": "car_wash",
        }
    )

    config = ForecastingConfig(horizon=7, seasonality=7, min_train_size=30)
    forecaster = StatsForecastForecaster(config=config)
    forecaster.fit(ts_data)
    return forecaster


@pytest.fixture
def pricing_model(elasticity_model, forecaster):
    """Create forecast pricing model"""
    optimizer_config = OptimizationConfig(
        min_price=5.0,
        max_price=50.0,
        break_even_price=8.0,
        max_price_change_pct=0.50,
    )
    return ForecastPricingModel(
        elasticity_model=elasticity_model,
        forecaster=forecaster,
        optimizer_config=optimizer_config,
    )


def test_model_initialization(pricing_model):
    """Test model initialization"""
    assert pricing_model.elasticity_mean < 0  # Negative elasticity
    assert pricing_model.elasticity_std > 0
    assert pricing_model.elasticity_hdi[0] < pricing_model.elasticity_hdi[1]


def test_price_for_tomorrow(pricing_model):
    """Test pricing for tomorrow"""
    result = pricing_model.price_for_tomorrow(current_price=15.0)

    assert isinstance(result, ForecastPricingResult)
    assert result.forecasted_demand > 0
    assert result.optimal_price > 0
    assert result.expected_revenue > 0
    assert result.confidence in ["high", "medium", "low"]
    assert result.demand_lower <= result.forecasted_demand <= result.demand_upper


def test_price_for_next_week(pricing_model):
    """Test pricing for entire week"""
    results = pricing_model.price_for_next_week(current_price=15.0)

    assert len(results) == 7
    assert all(isinstance(r, ForecastPricingResult) for r in results)

    # Prices should be within bounds
    for r in results:
        assert 5.0 <= r.optimal_price <= 50.0


def test_demand_function_incorporates_forecast(pricing_model):
    """Test that demand function uses forecast"""
    forecasted_demand = 120.0
    demand_function = pricing_model._create_demand_function(forecasted_demand)

    # At base price, demand should equal forecast
    assert abs(demand_function(15.0) - forecasted_demand) < 0.01

    # Higher price should reduce demand
    assert demand_function(20.0) < demand_function(15.0)

    # Lower price should increase demand
    assert demand_function(10.0) > demand_function(15.0)


def test_higher_forecast_leads_to_higher_price(pricing_model):
    """Test that higher demand forecast results in higher optimal price"""
    # Create mock forecast with high demand
    high_forecast = type(
        "Forecast",
        (),
        {
            "mean": np.array([150.0, 150.0, 150.0]),
            "lower": np.array([140.0, 140.0, 140.0]),
            "upper": np.array([160.0, 160.0, 160.0]),
            "dates": pd.date_range("2024-01-01", periods=3, freq="D"),
            "model_name": "Test",
            "horizon": 3,
        },
    )()

    # Create mock forecast with low demand
    low_forecast = type(
        "Forecast",
        (),
        {
            "mean": np.array([50.0, 50.0, 50.0]),
            "lower": np.array([40.0, 40.0, 40.0]),
            "upper": np.array([60.0, 60.0, 60.0]),
            "dates": pd.date_range("2024-01-01", periods=3, freq="D"),
            "model_name": "Test",
            "horizon": 3,
        },
    )()

    high_result = pricing_model.price_for_tomorrow(
        current_price=15.0,
        forecast=high_forecast,
        steps_ahead=1,
    )
    low_result = pricing_model.price_for_tomorrow(
        current_price=15.0,
        forecast=low_forecast,
        steps_ahead=1,
    )

    assert high_result.optimal_price > low_result.optimal_price


def test_confidence_levels(pricing_model):
    """Test that confidence is assigned correctly"""
    # High confidence (narrow interval)
    narrow_forecast = type(
        "Forecast",
        (),
        {
            "mean": np.array([100.0]),
            "lower": np.array([95.0]),
            "upper": np.array([105.0]),
            "dates": pd.date_range("2024-01-01", periods=1, freq="D"),
            "model_name": "Test",
            "horizon": 1,
        },
    )()

    # Low confidence (wide interval)
    wide_forecast = type(
        "Forecast",
        (),
        {
            "mean": np.array([100.0]),
            "lower": np.array([50.0]),
            "upper": np.array([150.0]),
            "dates": pd.date_range("2024-01-01", periods=1, freq="D"),
            "model_name": "Test",
            "horizon": 1,
        },
    )()

    narrow_result = pricing_model.price_for_tomorrow(
        current_price=15.0,
        forecast=narrow_forecast,
        steps_ahead=1,
    )
    wide_result = pricing_model.price_for_tomorrow(
        current_price=15.0,
        forecast=wide_forecast,
        steps_ahead=1,
    )

    assert narrow_result.confidence == "high"
    assert wide_result.confidence == "low"


def test_requires_fitted_models():
    """Test that model requires fitted components"""

    # Create unfitted models
    unfitted_elasticity = PriceElasticityModel()
    unfitted_forecaster = StatsForecastForecaster()

    with pytest.raises(ValueError, match="fitted"):
        ForecastPricingModel(
            elasticity_model=unfitted_elasticity,
            forecaster=unfitted_forecaster,
        )
