import matplotlib.pyplot as plt
import pytest

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.models.elasticity import PriceElasticityModel


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known elasticity"""
    config = DataGeneratorConfig(
        start_date="2023-01-01",
        end_date="2023-06-30",
        price_elasticity=-2.0,
        seed=42,
    )
    generator = CarWashDataGenerator(config)
    return generator.generate()


@pytest.fixture
def fitted_model(synthetic_data):
    """Fit model on synthetic data"""
    model = PriceElasticityModel(
        samples=500,  # Reduced for testing
        tune=200,
        chains=2,
    )
    results = model.fit(
        prices=synthetic_data["price"].values,
        demand=synthetic_data["quantity_sold"].values,
        weather_features=synthetic_data["is_sunny"].values,
        progressbar=False,
    )
    return model, results


def test_model_fitting(fitted_model):
    """Test that model fits without errors"""
    model, results = fitted_model
    assert results is not None
    assert hasattr(results, "posterior_mean")


def test_elasticity_recovery(fitted_model, synthetic_data):
    """Test that model recovers true elasticity"""
    model, results = fitted_model
    true_elasticity = -2.0  # Known from data generation

    # Posterior mean should be close to true value
    assert abs(results.posterior_mean - true_elasticity) < 1.0

    # Credible interval should contain true value
    ci_lower, ci_upper = results.hdi_95
    assert ci_lower < true_elasticity < ci_upper


def test_convergence_diagnostics(fitted_model):
    """Test MCMC convergence"""
    model, results = fitted_model
    # R-hat should be close to 1
    assert results.rhat < 1.1

    # Effective sample size should be reasonable
    assert results.effective_sample_size > 100


def test_weather_effect(synthetic_data):
    """Test that weather feature improves model"""
    # Fit with weather
    model_with_weather = PriceElasticityModel(samples=300, tune=100, chains=2)
    results_with = model_with_weather.fit(
        prices=synthetic_data["price"].values,
        demand=synthetic_data["quantity_sold"].values,
        weather_features=synthetic_data["is_sunny"].values,
        progressbar=False,
    )

    # Fit without weather
    model_without_weather = PriceElasticityModel(samples=300, tune=100, chains=2)
    results_without = model_without_weather.fit(
        prices=synthetic_data["price"].values,
        demand=synthetic_data["quantity_sold"].values,
        progressbar=False,
    )

    # Model with weather should have lower posterior std (more certainty)
    assert results_with.posterior_std < results_without.posterior_std


def test_plot_posterior_runs(fitted_model, tmp_path):
    """Test that posterior plotting doesn't crash"""
    model, results = fitted_model

    # Test with save_path
    save_path = tmp_path / "posterior.png"
    model.plot_posterior(save_path=str(save_path))
    assert save_path.exists()

    # Close all plots
    plt.close("all")


def test_plot_trace_runs(fitted_model, tmp_path):
    """Test that trace plotting doesn't crash"""
    model, results = fitted_model

    # Test with save_path
    save_path = tmp_path / "trace.png"
    model.plot_trace(save_path=str(save_path))
    assert save_path.exists()

    # Close all plots
    plt.close("all")
