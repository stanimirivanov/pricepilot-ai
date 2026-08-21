"""Tests for the complete pricing pipeline"""

from pathlib import Path

import pytest

from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PipelineResult, PricingPipeline


@pytest.fixture
def pipeline_config(tmp_path):
    """Create pipeline config with temp data path"""
    return PipelineConfig(
        data_path=str(tmp_path / "test_data.csv"),
        elasticity_samples=200,  # Reduced for testing
        elasticity_tune=100,
        elasticity_chains=1,
        forecast_horizon=3,
        anomaly_contamination=0.05,
        track_with_mlflow=False,
    )


@pytest.fixture
def fitted_pipeline(pipeline_config):
    """Create and fit pipeline"""
    pipeline = PricingPipeline(
        config=pipeline_config,
        enable_mlflow=False,  # Disable MLflow for tests
    )
    pipeline.load_or_generate_data()
    pipeline.fit_models()
    return pipeline


def test_pipeline_config_validation():
    """Test configuration validation"""
    # Valid config
    config = PipelineConfig()
    config.validate()

    # Invalid config
    with pytest.raises(ValueError):
        bad_config = PipelineConfig(min_price=10, max_price=5)
        bad_config.validate()


def test_pipeline_initialization(pipeline_config):
    """Test pipeline initialization"""
    pipeline = PricingPipeline(config=pipeline_config, enable_mlflow=False)
    assert pipeline.config == pipeline_config
    assert pipeline.data is None
    assert pipeline.elasticity_model is None


def test_data_loading(pipeline_config, tmp_path):
    """Test data loading/generation"""
    pipeline = PricingPipeline(config=pipeline_config, enable_mlflow=False)
    data = pipeline.load_or_generate_data()

    assert data is not None
    assert len(data) > 100
    assert "price" in data.columns
    assert "quantity_sold" in data.columns
    assert Path(pipeline_config.data_path).exists()


def test_model_fitting(fitted_pipeline):
    """Test model fitting"""
    assert fitted_pipeline.elasticity_model is not None
    assert fitted_pipeline.forecaster is not None
    assert fitted_pipeline.anomaly_detector is not None
    assert fitted_pipeline.pricing_model is not None
    assert fitted_pipeline.anomaly_pricing is not None


def test_get_tomorrow_price(fitted_pipeline):
    """Test getting tomorrow's price"""
    result = fitted_pipeline.get_tomorrow_price()

    assert isinstance(result, PipelineResult)
    assert result.forecasted_demand > 0
    assert result.optimal_price > 0
    assert result.expected_revenue > 0
    assert result.confidence in ["high", "medium", "low"]
    assert result.anomaly_status in ["ANOMALY", "NORMAL"]
    assert result.execution_time > 0


def test_get_week_prices(fitted_pipeline):
    """Test getting week prices"""
    results = fitted_pipeline.get_week_prices()

    assert len(results) == 7
    assert all(isinstance(r, PipelineResult) for r in results)

    # Prices should be within bounds
    for r in results:
        assert (
            fitted_pipeline.config.min_price <= r.optimal_price <= fitted_pipeline.config.max_price
        )


def test_pipeline_result_to_dict(fitted_pipeline):
    """Test PipelineResult conversion to dictionary"""
    result = fitted_pipeline.get_tomorrow_price()
    result_dict = result.to_dict()

    assert "timestamp" in result_dict
    assert "current_price" in result_dict
    assert "forecasted_demand" in result_dict
    assert "optimal_price" in result_dict
    assert "anomaly_status" in result_dict


def test_pipeline_context_manager(pipeline_config):
    """Test pipeline as context manager"""
    with PricingPipeline(config=pipeline_config, enable_mlflow=False) as pipeline:
        pipeline.load_or_generate_data()
        pipeline.fit_models()
        result = pipeline.get_tomorrow_price()
        assert result is not None


def test_custom_current_price(fitted_pipeline):
    """Test with custom current price"""
    custom_price = 20.0
    result = fitted_pipeline.get_tomorrow_price(current_price=custom_price)

    assert result.current_price == custom_price

    # Price change should respect constraints
    max_change = fitted_pipeline.config.max_price_change_pct
    assert abs(result.price_change_pct) <= max_change + 0.01
