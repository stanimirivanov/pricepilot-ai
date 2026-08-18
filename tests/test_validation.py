import pytest

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.data.validation import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def valid_data():
    config = DataGeneratorConfig(seed=42)
    generator = CarWashDataGenerator(config)
    return generator.generate()


def test_valid_data_passes(validator, valid_data):
    """Test that valid data passes validation"""
    result = validator.validate(valid_data)
    assert result.is_valid
    assert len(result.errors) == 0


def test_missing_columns_detected(validator, valid_data):
    """Test that missing columns are detected"""
    df = valid_data.drop("price", axis=1)
    result = validator.validate(df)
    assert not result.is_valid
    assert any("price" in error for error in result.errors)


def test_negative_prices_detected(validator, valid_data):
    """Test that negative prices are detected"""
    df = valid_data.copy()
    df.loc[0, "price"] = -10
    result = validator.validate(df)
    assert not result.is_valid
    assert any("positive" in error for error in result.errors)


def test_validation_metrics(validator, valid_data):
    """Test that metrics are calculated"""
    result = validator.validate(valid_data)
    assert "n_records" in result.metrics
    assert result.metrics["n_records"] == len(valid_data)
    assert "missing_values_pct" in result.metrics
    assert result.metrics["missing_values_pct"] == 0
