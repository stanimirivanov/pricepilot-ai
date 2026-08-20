"""StatsForecast-based time series forecasting"""

from typing import Any, cast

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
        """Initialize StatsForecast forecaster"""
        super().__init__(horizon=config.horizon if config else 7)
        self.config = config or ForecastingConfig()
        self.horizon = self.config.horizon

        if models is None:
            models = ["auto_arima", "seasonal_naive"]

        self.model_names = models
        self.models = self._create_models(models)

        # Map lowercase names to StatsForecast column names
        self.model_column_map = {
            "auto_arima": "AutoARIMA",
            "auto_ets": "AutoETS",
            "seasonal_naive": "SeasonalNaive",
        }

        self.forecaster: StatsForecast | None = None
        self.training_data: pd.DataFrame | None = None
        self.is_fitted = False
        self.last_date: pd.Timestamp | None = None

    def _create_models(self, model_names: list[str]) -> list[Any]:
        """Create StatsForecast model instances"""
        models = []

        for name in model_names:
            if name == "auto_arima":
                models.append(AutoARIMA(season_length=self.config.seasonality))
            elif name == "auto_ets":
                models.append(AutoETS(season_length=self.config.seasonality))
            elif name == "seasonal_naive":
                models.append(SeasonalNaive(season_length=self.config.seasonality))
            else:
                logger.warning(f"Unknown model: {name}, skipping")

        if not models:
            raise ValueError("No valid models specified")

        return models

    def _get_forecast_column(self, forecast_df: pd.DataFrame) -> str:
        """Find the forecast column name in the output DataFrame"""
        # Map our model names to StatsForecast column names
        for model_name in self.model_names:
            column_name = self.model_column_map.get(model_name, model_name)
            if column_name in forecast_df.columns:
                return column_name

        # Fallback: find any column that's not ds, unique_id, or interval columns
        excluded = {"ds", "unique_id", "y"}
        for col in forecast_df.columns:
            if col not in excluded and not col.endswith(("-lo", "-hi")):
                if "-lo-" not in col and "-hi-" not in col:
                    return str(col)

        raise ValueError(f"No forecast columns found. Available: {forecast_df.columns.tolist()}")

    def fit(self, data: pd.DataFrame) -> "StatsForecastForecaster":
        """Fit forecaster to historical data"""
        logger.info(f"Fitting StatsForecast with models: {self.model_names}")

        if "ds" not in data.columns or "y" not in data.columns:
            raise ValueError("Data must contain 'ds' and 'y' columns")

        if "unique_id" not in data.columns:
            data = data.copy()
            data["unique_id"] = "series_1"

        data = data.sort_values("ds").reset_index(drop=True)

        if len(data) < self.config.min_train_size:
            raise ValueError(f"Insufficient data: {len(data)} < {self.config.min_train_size} days")

        self.training_data = data.copy()
        self.last_date = pd.to_datetime(data["ds"].iloc[-1])

        try:
            self.forecaster = StatsForecast(
                models=self.models,
                freq="D",
                n_jobs=-1,
                verbose=False,
            )
            self.forecaster.fit(data)
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
        """Generate forecast for future periods"""
        if not self.is_fitted or self.forecaster is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.training_data is None or self.last_date is None:
            raise ValueError("Training data not available. Call fit() first.")

        if steps is None:
            steps = self.horizon

        if level is None:
            level = [int(self.config.confidence_level * 100)]

        logger.info(f"Generating {steps}-day forecast")

        # Generate forecast - cast to pandas DataFrame
        forecast_raw = self.forecaster.forecast(
            h=steps,
            df=self.training_data,
            level=level,
        )
        forecast_df = cast(pd.DataFrame, forecast_raw)

        # Find the forecast column
        forecast_col = self._get_forecast_column(forecast_df)

        # Find interval columns
        lo_col = f"{forecast_col}-lo-{level[0]}"
        hi_col = f"{forecast_col}-hi-{level[0]}"

        # Create date range for forecast
        forecast_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )

        # Extract values as numpy arrays with explicit type casting
        mean = np.asarray(forecast_df[forecast_col].values, dtype=np.float64)

        if lo_col in forecast_df.columns and hi_col in forecast_df.columns:
            lower = np.asarray(forecast_df[lo_col].values, dtype=np.float64)
            upper = np.asarray(forecast_df[hi_col].values, dtype=np.float64)
        else:
            std_estimate = float(np.std(mean)) * 0.2
            lower = mean - 1.645 * std_estimate
            upper = mean + 1.645 * std_estimate
            logger.warning("Prediction intervals not found, using heuristic")

        result = ForecastResult(
            dates=forecast_dates,
            mean=mean,
            lower=lower,
            upper=upper,
            model_name=f"StatsForecast_{forecast_col}",
            horizon=steps,
        )

        logger.info(
            f"Forecast generated: mean={float(np.mean(mean)):.1f}, "
            f"range=[{float(np.min(lower)):.1f}, {float(np.max(upper)):.1f}]"
        )
        return result

    def evaluate_accuracy(
        self,
        test_data: pd.DataFrame,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate forecast accuracy on test data"""
        if metrics is None:
            metrics = ["mae", "rmse", "mape", "smape"]

        forecast = self.predict(steps=len(test_data))
        actual = np.asarray(test_data["y"].values, dtype=np.float64)

        results = {}

        for metric in metrics:
            if metric == "mae":
                results["mae"] = float(np.mean(np.abs(actual - forecast.mean)))
            elif metric == "rmse":
                results["rmse"] = float(np.sqrt(np.mean((actual - forecast.mean) ** 2)))
            elif metric == "mape":
                mask = actual != 0
                if np.any(mask):
                    results["mape"] = float(
                        np.mean(np.abs((actual[mask] - forecast.mean[mask]) / actual[mask])) * 100
                    )
                else:
                    results["mape"] = float("nan")
            elif metric == "smape":
                denominator = (np.abs(actual) + np.abs(forecast.mean)) / 2
                mask = denominator != 0
                if np.any(mask):
                    results["smape"] = float(
                        np.mean(2 * np.abs(forecast.mean[mask] - actual[mask]) / denominator[mask])
                        * 100
                    )
                else:
                    results["smape"] = float("nan")

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

        ax.plot(
            pd.to_datetime(historical_data["ds"]),
            historical_data["y"],
            label="Historical",
            color="blue",
            linewidth=1.5,
        )

        ax.plot(
            forecast.dates,
            forecast.mean,
            label="Forecast",
            color="red",
            linewidth=2,
        )

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
