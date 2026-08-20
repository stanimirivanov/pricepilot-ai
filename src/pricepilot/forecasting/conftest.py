"""Test fixtures for forecasting tests"""

import numpy as np
import pandas as pd
import pytest

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig


@pytest.fixture
def historical_data() -> pd.DataFrame:
    """Generate historical data for forecasting tests"""
    config = DataGeneratorConfig(
        start_date="2022-01-01",
        end_date="2023-12-31",
        seed=42,
    )
    generator = CarWashDataGenerator(config)
    return generator.generate()


@pytest.fixture
def daily_demand_series(historical_data: pd.DataFrame) -> pd.DataFrame:
    """Create daily demand time series"""
    return pd.DataFrame(
        {
            "ds": historical_data["date"],
            "y": historical_data["quantity_sold"],
            "unique_id": "car_wash_demand",
        }
    )


@pytest.fixture
def weekly_pattern_data() -> pd.DataFrame:
    """Create data with strong weekly pattern"""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    # Strong weekly pattern: higher on weekends
    day_of_week = dates.dayofweek
    weekend_effect = np.where(day_of_week >= 5, 20, 0)
    base_demand = 100 + weekend_effect
    noise = np.random.default_rng(42).normal(0, 5, len(dates))

    return pd.DataFrame(
        {
            "ds": dates,
            "y": base_demand + noise,
            "unique_id": "test_demand",
        }
    )
