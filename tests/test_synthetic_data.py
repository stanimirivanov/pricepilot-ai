import pandas as pd
import pytest

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig


@pytest.fixture
def generator():
    config = DataGeneratorConfig(start_date="2023-01-01", end_date="2023-12-31", seed=42)
    return CarWashDataGenerator(config)


def test_data_generation(generator):
    """Test basic data generation"""
    df = generator.generate()

    # Check data shape
    assert len(df) == 365  # One year of daily data
    assert all(
        col in df.columns
        for col in [
            "date",
            "price",
            "quantity_sold",
            "revenue",
            "is_raining",
            "is_sunny",
            "temperature",
        ]
    )


def test_price_effect(generator):
    """Test that higher prices lead to lower demand"""
    df = generator.generate()

    # Calculate correlation
    correlation = df["price"].corr(df["quantity_sold"])
    assert correlation < 0  # Negative correlation expected


def test_weather_effect(generator):
    """Test that sunny days have higher demand"""
    df = generator.generate()

    sunny_demand = df[df["is_sunny"]]["quantity_sold"].mean()
    rainy_demand = df[df["is_raining"]]["quantity_sold"].mean()

    assert sunny_demand > rainy_demand


def test_weekend_effect(generator):
    """Test that weekends have higher demand"""
    df = generator.generate()

    weekend_demand = df[df["is_weekend"]]["quantity_sold"].mean()
    weekday_demand = df[~df["is_weekend"]]["quantity_sold"].mean()

    assert weekend_demand > weekday_demand


def test_reproducibility():
    """Test that same seed produces same data"""
    config = DataGeneratorConfig(seed=42)
    gen1 = CarWashDataGenerator(config)
    gen2 = CarWashDataGenerator(config)

    df1 = gen1.generate()
    df2 = gen2.generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_non_negative_demand(generator):
    """Test that demand is never negative"""
    df = generator.generate()
    assert (df["quantity_sold"] >= 0).all()


def test_save_and_load(generator, tmp_path):
    """Test saving and loading data"""
    filepath = tmp_path / "test_data.csv"
    df = generator.generate_and_save(str(filepath))

    # Load and verify
    loaded_df = pd.read_csv(filepath)
    pd.testing.assert_frame_equal(df, loaded_df)
