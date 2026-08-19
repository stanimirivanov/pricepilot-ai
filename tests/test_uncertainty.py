# tests/test_uncertainty.py
import numpy as np
import pytest

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.models.uncertainty import UncertaintyQuantifier, create_demand_features


@pytest.fixture
def synthetic_data():
    """Generate synthetic data"""
    config = DataGeneratorConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        seed=42,
    )
    generator = CarWashDataGenerator(config)
    return generator.generate()


@pytest.fixture
def fitted_quantifier(synthetic_data):
    """Fit uncertainty quantifier"""
    # Create features
    X = create_demand_features(
        prices=synthetic_data["price"].values,
        weather=synthetic_data["is_sunny"].values.astype(float),
    )
    y = synthetic_data["quantity_sold"].values

    # Split data
    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Fit model
    quantifier = UncertaintyQuantifier(alpha=0.1, n_estimators=50)
    quantifier.fit(X_train, y_train)

    return quantifier, X_test, y_test


def test_prediction_intervals(fitted_quantifier):
    """Test that prediction intervals are generated"""
    quantifier, X_test, _ = fitted_quantifier

    predictions = quantifier.predict(X_test)

    assert predictions.lower.shape == predictions.mean.shape
    assert predictions.upper.shape == predictions.mean.shape
    assert np.all(predictions.lower <= predictions.mean)
    assert np.all(predictions.mean <= predictions.upper)


def test_coverage(fitted_quantifier):
    """Test empirical coverage"""
    quantifier, X_test, y_test = fitted_quantifier

    coverage = quantifier.calculate_coverage(X_test, y_test)

    # Coverage should be close to 0.9 (90% interval)
    assert 0.8 <= coverage <= 0.95


def test_interval_width_stats(fitted_quantifier):
    """Test interval width statistics"""
    quantifier, X_test, _ = fitted_quantifier

    stats = quantifier.get_interval_width_stats(X_test)

    assert "mean_width" in stats
    assert stats["mean_width"] > 0
    assert stats["min_width"] <= stats["mean_width"] <= stats["max_width"]


def test_create_demand_features(synthetic_data):
    """Test feature creation"""
    prices = synthetic_data["price"].values
    weather = synthetic_data["is_sunny"].values.astype(float)

    # Simple case
    X_simple = create_demand_features(prices)
    assert X_simple.shape == (len(prices), 1)

    # With weather
    X_weather = create_demand_features(prices, weather)
    assert X_weather.shape == (len(prices), 2)

    # With day of week
    day_of_week = synthetic_data["day_of_week"].values
    X_full = create_demand_features(prices, weather, day_of_week)
    assert X_full.shape == (len(prices), 9)  # 1 + 1 + 7
