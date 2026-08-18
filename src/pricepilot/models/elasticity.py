"""Bayesian price elasticity estimation using PyMC"""

from dataclasses import dataclass
from typing import Any, cast

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from loguru import logger
from xarray import DataArray, Dataset


@dataclass
class ElasticityResults:
    """Container for elasticity estimation results"""

    posterior_mean: float
    posterior_std: float
    hdi_95: tuple[float, float]
    rhat: float
    effective_sample_size: float
    trace: Any  # InferenceData object

    def summary(self) -> dict[str, float]:
        """Return summary statistics"""
        return {
            "elasticity_mean": self.posterior_mean,
            "elasticity_std": self.posterior_std,
            "hdi_lower": self.hdi_95[0],
            "hdi_upper": self.hdi_95[1],
            "rhat": self.rhat,
            "ess": self.effective_sample_size,
        }


class PriceElasticityModel:
    """Bayesian price elasticity model using PyMC"""

    def __init__(
        self,
        prior_mean: float = -2.0,
        prior_std: float = 1.0,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int = 42,
    ):
        """Initialize Bayesian elasticity model"""
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.samples = samples
        self.tune = tune
        self.chains = chains
        self.random_seed = random_seed

        self.model: pm.Model | None = None
        self.trace: az.InferenceData | None = None
        self.results: ElasticityResults | None = None

    def _get_posterior_group(self) -> Dataset:
        """Get posterior group from trace as Dataset"""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use getattr to access posterior group
        posterior = getattr(self.trace, "posterior", None)
        if posterior is None:
            raise ValueError("No posterior group found in trace")

        return cast(Dataset, posterior)

    def _extract_variable(self, var_name: str) -> DataArray:
        """Extract variable from posterior as DataArray"""
        posterior = self._get_posterior_group()

        if var_name not in posterior:
            raise ValueError(f"Variable '{var_name}' not found in posterior")

        return cast(DataArray, posterior[var_name])

    def build_model(
        self,
        prices: np.ndarray,
        demand: np.ndarray,
        weather_features: np.ndarray | None = None,
    ) -> pm.Model:
        """Build PyMC model for elasticity estimation"""
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal("alpha", mu=100, sigma=20)
            beta_price = pm.Normal(
                "beta_price",
                mu=self.prior_mean,
                sigma=self.prior_std,
            )

            # Weather effect (if provided)
            if weather_features is not None:
                beta_weather = pm.Normal("beta_weather", mu=10, sigma=5)
                mu = alpha + beta_price * prices + beta_weather * weather_features
            else:
                mu = alpha + beta_price * prices

            # Observation noise
            sigma = pm.HalfNormal("sigma", sigma=10)

            # Likelihood
            pm.Normal("demand", mu=mu, sigma=sigma, observed=demand)

        self.model = model
        return model

    def fit(
        self,
        prices: np.ndarray,
        demand: np.ndarray,
        weather_features: np.ndarray | None = None,
        progressbar: bool = True,
    ) -> ElasticityResults:
        """Fit Bayesian model to data"""
        logger.info("Building Bayesian elasticity model...")
        model = self.build_model(prices, demand, weather_features)

        logger.info(f"Sampling posterior ({self.samples} samples, {self.chains} chains)...")
        with model:
            self.trace = pm.sample(
                draws=self.samples,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_seed,
                progressbar=progressbar,
                return_inferencedata=True,
            )

        # Extract posterior samples
        beta_price_da = self._extract_variable("beta_price")
        beta_price_posterior = beta_price_da.values.flatten()

        # Calculate HDI
        hdi = az.hdi(beta_price_da, hdi_prob=0.95)

        # Calculate convergence diagnostics
        rhat_da = cast(DataArray, az.rhat(beta_price_da))
        ess_da = cast(DataArray, az.ess(beta_price_da))

        rhat_value = float(rhat_da.values)
        ess_value = float(ess_da.values)

        self.results = ElasticityResults(
            posterior_mean=float(np.mean(beta_price_posterior)),
            posterior_std=float(np.std(beta_price_posterior)),
            hdi_95=(float(hdi[0]), float(hdi[1])),
            rhat=rhat_value,
            effective_sample_size=ess_value,
            trace=self.trace,
        )

        logger.info(f"Elasticity posterior mean: {self.results.posterior_mean:.3f}")
        logger.info(f"95% HDI: [{self.results.hdi_95[0]:.3f}, {self.results.hdi_95[1]:.3f}]")
        logger.info(f"R-hat: {self.results.rhat:.3f} (should be < 1.1)")

        return self.results

    def plot_posterior(self, save_path: str | None = None) -> None:
        """Plot posterior distribution of elasticity"""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create figure with 2 subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Posterior distribution
        az.plot_posterior(
            self.trace,
            var_names=["beta_price"],
            ax=axes[0],
        )
        axes[0].set_title("Price Elasticity Posterior Distribution")

        # Forest plot (shows posterior intervals)
        az.plot_forest(
            self.trace,
            var_names=["beta_price", "alpha", "sigma"],
            ax=axes[1],
        )
        axes[1].set_title("Parameter Estimates with 95% HDI")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved posterior plot to {save_path}")
        else:
            plt.show()

    def plot_trace(self, save_path: str | None = None) -> None:
        """Plot MCMC trace for convergence diagnostics"""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Trace plot creates its own figure
        axes = az.plot_trace(
            self.trace,
            var_names=["beta_price", "alpha"],
        )

        # Get the figure from the axes
        fig = axes[0][0].figure
        fig.suptitle("MCMC Trace and Posterior Distributions", fontsize=14, y=1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved trace plot to {save_path}")
        else:
            plt.show()

    def get_elasticity_credible_interval(self, interval: float = 0.95) -> tuple[float, float]:
        """Get credible interval for elasticity"""
        beta_price_da = self._extract_variable("beta_price")
        hdi = az.hdi(beta_price_da, hdi_prob=interval)

        return float(hdi[0]), float(hdi[1])
