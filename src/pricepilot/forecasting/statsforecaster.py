"""StatsForecast-based time series forecasting"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

from pricepilot.forecasting.base import BaseForecaster, ForecastResult
from pricepilot.forecasting.config import ForecastingConfig


class StatsForecastForecaster(BaseForecaster):
    """Time series forecaster using StatsForecast library"""

    def __init__(
        self,
        config: ForecastingConfig | None = None,
        models: list[str] | None = None,
    ):
        """
        Initialize StatsForecast forecaster

        Args:
            config: Forecasting configuration
            models: List of model names to use ("auto_arima", "auto_ets", "seasonal_naive")
        """
        super().__init__(horizon=config.horizon if config else 7)
        self.config = config or ForecastingConfig()
        self.horizon = self.config.horizon

        # Define models to use
        if models is None:
            models = ["auto_arima", "seasonal_naive"]

        self.model_names = models
        self.models = self._create_models(models)
        self.forecaster: StatsForecast | None = None
        self.fitted_models: dict[str, Any] | None = None
        self.last_date: pd.Timestamp | None = None

    def _create_models(self, model_names: list[str]) -> list[Any]:
        """Create StatsForecast model instances"""
        models = []

        for name in model_names:
            if name == "auto_arima":
                models.append(
                    AutoARIMA(
                        season_length=self.config.seasonality,
                        approximation=True,
                        allowmean=True,
                    )
                )
            elif name == "auto_ets":
                models.append(
                    AutoETS(
                        season_length=self.config.seasonality,
                        model="ZZZ",
                    )
                )
            elif name == "seasonal_naive":
                models.append(SeasonalNaive(season_length=self.config.seasonality))
            else:
                logger.warning(f"Unknown model: {name}, skipping")

        if not models:
            raise ValueError("No valid models specified")

        return models

    def fit(self, data: pd.DataFrame) -> "StatsForecastForecaster":
        """
        Fit forecaster to historical data

        Args:
            data: DataFrame with columns:
                - ds: Date column
                - y: Target value (demand)
                - unique_id: Identifier (optional, defaults to "series_1")

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting StatsForecast with models: {self.model_names}")

        # Validate data
        if "ds" not in data.columns or "y" not in data.columns:
            raise ValueError("Data must contain 'ds' and 'y' columns")

        # Ensure unique_id column exists
        if "unique_id" not in data.columns:
            data = data.copy()
            data["unique_id"] = "series_1"

        # Ensure data is sorted
        data = data.sort_values("ds").reset_index(drop=True)

        # Check minimum training size
        if len(data) < self.config.min_train_size:
            raise ValueError(f"Insufficient data: {len(data)} < {self.config.min_train_size} days")

        # Store last date for future predictions
        self.last_date = pd.to_datetime(data["ds"].iloc[-1])

        # Fit models
        try:
            self.forecaster = StatsForecast(
                df=data,
                models=self.models,
                freq="D",  # Daily frequency
                n_jobs=-1,
                verbose=False,
            )
            logger.info("StatsForecast fitted successfully")
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Failed to fit StatsForecast: {e}")
            raise

        return self

    def predict(
        self,
        steps: int | None = None,
        level: list[int] | None = None,
    ) -> ForecastResult:
        """
        Generate forecast for future periods

        Args:
            steps: Number of periods to forecast (defaults to horizon)
            level: Confidence levels for prediction intervals (defaults to [90])

        Returns:
            ForecastResult with predictions and intervals
        """
        if not self.is_fitted or self.forecaster is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if steps is None:
            steps = self.horizon

        if level is None:
            level = [int(self.config.confidence_level * 100)]

        logger.info(f"Generating {steps}-day forecast")

        # Generate forecast
        forecast_df = self.forecaster.forecast(
            h=steps,
            level=level,
        )

        # Extract predictions for the primary model
        primary_model = self.model_names[0]
        forecast_col = primary_model
        lower_col = f"{primary_model}-lo-{level[0]}"
        upper_col = f"{primary_model}-hi-{level[0]}"

        if forecast_col not in forecast_df.columns:
            # Use first available model
            forecast_col = [c for c in forecast_df.columns if c in self.model_names][0]
            lower_col = f"{forecast_col}-lo-{level[0]}"
            upper_col = f"{forecast_col}-hi-{level[0]}"

        # Create date range for forecast
        forecast_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )

        # Extract values
        mean = forecast_df[forecast_col].values
        lower = forecast_df[lower_col].values if lower_col in forecast_df.columns else mean * 0.8
        upper = forecast_df[upper_col].values if upper_col in forecast_df.columns else mean * 1.2

        result = ForecastResult(
            dates=forecast_dates,
            mean=mean,
            lower=lower,
            upper=upper,
            model_name=f"StatsForecast_{primary_model}",
            horizon=steps,
        )

        logger.info(
            f"Forecast generated: mean={np.mean(mean):.1f}, range=[{np.min(lower):.1f}, {np.max(upper):.1f}]"
        )
        return result

    def evaluate_accuracy(
        self,
        test_data: pd.DataFrame,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate forecast accuracy on test data

        Args:
            test_data: DataFrame with actual values
            metrics: List of metrics to calculate

        Returns:
            Dictionary with metric values
        """
        if metrics is None:
            metrics = ["mae", "rmse", "mape", "smape"]

        # Generate forecast for test period
        forecast = self.predict(steps=len(test_data))
        actual = test_data["y"].values

        results = {}

        for metric in metrics:
            if metric == "mae":
                results["mae"] = float(np.mean(np.abs(actual - forecast.mean)))
            elif metric == "rmse":
                results["rmse"] = float(np.sqrt(np.mean((actual - forecast.mean) ** 2)))
            elif metric == "mape":
                # Avoid division by zero
                mask = actual != 0
                if mask.any():
                    results["mape"] = float(
                        np.mean(np.abs((actual[mask] - forecast.mean[mask]) / actual[mask])) * 100
                    )
                else:
                    results["mape"] = np.nan
            elif metric == "smape":
                denominator = (np.abs(actual) + np.abs(forecast.mean)) / 2
                mask = denominator != 0
                if mask.any():
                    results["smape"] = float(
                        np.mean(2 * np.abs(forecast.mean[mask] - actual[mask]) / denominator[mask])
                        * 100
                    )
                else:
                    results["smape"] = np.nan

        logger.info(f"Forecast accuracy: {results}")
        return results

    def plot_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast: ForecastResult,
        save_path: str | None = None,
    ) -> None:
        """Plot forecast against historical data"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        ax.plot(
            pd.to_datetime(historical_data["ds"]),
            historical_data["y"],
            label="Historical",
            color="blue",
            linewidth=1.5,
        )

        # Plot forecast
        ax.plot(
            forecast.dates,
            forecast.mean,
            label="Forecast",
            color="red",
            linewidth=2,
        )

        # Plot prediction intervals
        ax.fill_between(
            forecast.dates,
            forecast.lower,
            forecast.upper,
            alpha=0.2,
            color="red",
            label=f"{self.config.confidence_level * 100:.0f}% Prediction Interval",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        ax.set_title(f"Demand Forecast - {forecast.model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved forecast plot to {save_path}")
        else:
            plt.show()
