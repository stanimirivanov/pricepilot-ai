"""Forecast-integrated pricing model combining demand forecasts with elasticity"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from pricepilot.forecasting.base import ForecastResult
from pricepilot.forecasting.statsforecaster import StatsForecastForecaster
from pricepilot.models.elasticity import PriceElasticityModel
from pricepilot.optimization.price_optimizer import OptimizationConfig, PriceOptimizer


@dataclass
class ForecastPricingResult:
    """Container for forecast-based pricing decisions"""

    date: pd.Timestamp
    forecasted_demand: float
    demand_lower: float
    demand_upper: float
    optimal_price: float
    expected_revenue: float
    confidence: str  # "high", "medium", "low"
    price_change_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "forecasted_demand": self.forecasted_demand,
            "demand_lower": self.demand_lower,
            "demand_upper": self.demand_upper,
            "optimal_price": self.optimal_price,
            "expected_revenue": self.expected_revenue,
            "confidence": self.confidence,
            "price_change_pct": self.price_change_pct,
        }


class ForecastPricingModel:
    """Integrates demand forecasting with price elasticity for proactive pricing"""

    def __init__(
        self,
        elasticity_model: PriceElasticityModel,
        forecaster: StatsForecastForecaster,
        optimizer_config: OptimizationConfig | None = None,
    ):
        """
        Initialize forecast-pricing integration

        Args:
            elasticity_model: Fitted elasticity model
            forecaster: Fitted forecasting model
            optimizer_config: Configuration for price optimization
        """
        self.elasticity_model = elasticity_model
        self.forecaster = forecaster
        self.optimizer_config = optimizer_config or OptimizationConfig()

        # Store elasticity estimates
        if elasticity_model.results is None:
            raise ValueError("Elasticity model must be fitted first")

        self.elasticity_mean = elasticity_model.results.posterior_mean
        self.elasticity_std = elasticity_model.results.posterior_std
        self.elasticity_hdi = elasticity_model.results.hdi_95

        logger.info(f"Initialized ForecastPricingModel with elasticity={self.elasticity_mean:.3f}")

    def _create_demand_function(
        self,
        forecasted_demand: float,
        base_price: float = 15.0,
    ):
        """
        Create demand function that incorporates forecast

        The forecast shifts the intercept (base demand) while maintaining
        the estimated price elasticity.

        Args:
            forecasted_demand: Expected demand at base price
            base_price: Reference price for the forecast

        Returns:
            Callable demand function
        """
        elasticity = self.elasticity_mean

        def demand_function(price: float) -> float:
            """Demand at given price, adjusted for forecast"""
            # Linear demand: Q = forecasted_demand + elasticity * (price - base_price)
            # At price = base_price, demand = forecasted_demand
            # For each $1 above base_price, demand changes by elasticity amount
            demand = forecasted_demand + elasticity * (price - base_price)
            return max(demand, 0)  # Ensure non-negative

        return demand_function

    def price_for_tomorrow(
        self,
        current_price: float,
        forecast: ForecastResult | None = None,
        steps_ahead: int = 1,
    ) -> ForecastPricingResult:
        """
        Calculate optimal price for tomorrow based on demand forecast

        Args:
            current_price: Current price
            forecast: Forecast result (if None, generates new forecast)
            steps_ahead: Which forecast step to use (1 = tomorrow)

        Returns:
            ForecastPricingResult with pricing decision
        """
        # Generate forecast if not provided
        if forecast is None:
            forecast = self.forecaster.predict(steps=steps_ahead)

        if steps_ahead > len(forecast.mean):
            raise ValueError(
                f"steps_ahead={steps_ahead} exceeds forecast horizon={len(forecast.mean)}"
            )

        # Extract forecasted demand for the target day
        forecasted_demand = float(forecast.mean[steps_ahead - 1])
        demand_lower = float(forecast.lower[steps_ahead - 1])
        demand_upper = float(forecast.upper[steps_ahead - 1])
        forecast_date = forecast.dates[steps_ahead - 1]

        # Create demand function with forecast adjustment
        demand_function = self._create_demand_function(forecasted_demand)

        # Create optimizer
        optimizer = PriceOptimizer(
            demand_function=demand_function,
            config=self.optimizer_config,
        )

        # Optimize price
        optimization_result = optimizer.optimize(current_price)

        # Determine confidence based on forecast interval width
        interval_width = demand_upper - demand_lower
        relative_width = interval_width / max(forecasted_demand, 1)

        if relative_width < 0.15:
            confidence = "high"
        elif relative_width < 0.30:
            confidence = "medium"
        else:
            confidence = "low"

        result = ForecastPricingResult(
            date=pd.Timestamp(forecast_date),
            forecasted_demand=forecasted_demand,
            demand_lower=demand_lower,
            demand_upper=demand_upper,
            optimal_price=optimization_result.optimal_price,
            expected_revenue=optimization_result.expected_revenue,
            confidence=confidence,
            price_change_pct=optimization_result.price_change_pct,
        )

        logger.info(
            f"Tomorrow ({result.date.strftime('%Y-%m-%d')}): "
            f"forecast_demand={forecasted_demand:.0f}, "
            f"optimal_price=${result.optimal_price:.2f}, "
            f"confidence={confidence}"
        )

        return result

    def price_for_next_week(
        self,
        current_price: float,
        forecast: ForecastResult | None = None,
    ) -> list[ForecastPricingResult]:
        """
        Calculate optimal prices for the next 7 days

        Args:
            current_price: Current price
            forecast: Forecast result (if None, generates 7-day forecast)

        Returns:
            List of ForecastPricingResult for each day
        """
        if forecast is None:
            forecast = self.forecaster.predict(steps=7)

        results = []

        for day in range(1, len(forecast.mean) + 1):
            # Update reference price based on previous day's optimal
            if results:
                current_price = results[-1].optimal_price

            result = self.price_for_tomorrow(
                current_price=current_price,
                forecast=forecast,
                steps_ahead=day,
            )
            results.append(result)

        return results

    def evaluate_pricing_strategy(
        self,
        historical_data: pd.DataFrame,
        test_days: int = 30,
    ) -> dict[str, float]:
        """
        Evaluate the forecast-based pricing strategy against historical data

        Args:
            historical_data: DataFrame with ds, y, and price columns
            test_days: Number of days to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Split data
        train_data = historical_data.iloc[:-test_days]
        test_data = historical_data.iloc[-test_days:]

        # Fit forecaster on training data
        self.forecaster.fit(train_data[["ds", "y", "unique_id"]])

        # Calculate prices for each test day
        total_revenue = 0
        total_demand = 0
        current_price = train_data["price"].iloc[-1] if "price" in train_data.columns else 15.0

        for idx, row in test_data.iterrows():
            # Generate forecast for this day
            forecast = self.forecaster.predict(steps=1)

            # Get optimal price
            pricing_result = self.price_for_tomorrow(
                current_price=current_price,
                forecast=forecast,
                steps_ahead=1,
            )

            # Simulate demand at this price (using actual elasticity)
            actual_demand = row["y"]

            # Calculate revenue
            revenue = pricing_result.optimal_price * actual_demand
            total_revenue += revenue
            total_demand += actual_demand

            # Update price for next day
            current_price = pricing_result.optimal_price

        # Calculate metrics
        avg_daily_revenue = total_revenue / test_days
        avg_daily_demand = total_demand / test_days
        avg_price = total_revenue / total_demand if total_demand > 0 else 0

        return {
            "test_days": test_days,
            "avg_daily_revenue": float(avg_daily_revenue),
            "avg_daily_demand": float(avg_daily_demand),
            "avg_price": float(avg_price),
        }

    def plot_pricing_forecast(
        self,
        current_price: float,
        forecast: ForecastResult | None = None,
        save_path: str | None = None,
    ) -> None:
        """Plot forecasted demand and optimal prices"""
        import matplotlib.pyplot as plt

        if forecast is None:
            forecast = self.forecaster.predict(steps=7)

        # Calculate prices for each day
        results = self.price_for_next_week(current_price, forecast)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Demand forecast
        axes[0].plot(
            forecast.dates,
            forecast.mean,
            label="Forecasted Demand",
            color="blue",
            linewidth=2,
        )
        axes[0].fill_between(
            forecast.dates,
            forecast.lower,
            forecast.upper,
            alpha=0.2,
            color="blue",
            label="90% Prediction Interval",
        )
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Demand (units)")
        axes[0].set_title("Demand Forecast for Next 7 Days")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Optimal prices
        prices = [r.optimal_price for r in results]
        dates = [r.date for r in results]

        axes[1].plot(
            dates,
            prices,
            marker="o",
            label="Optimal Price",
            color="red",
            linewidth=2,
        )
        axes[1].axhline(
            current_price,
            color="green",
            linestyle="--",
            label=f"Current Price (${current_price:.2f})",
        )
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Price ($)")
        axes[1].set_title("Optimal Prices Based on Forecast")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved pricing forecast plot to {save_path}")
        else:
            plt.show()
