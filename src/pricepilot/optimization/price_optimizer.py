"""Constrained price optimization using SciPy"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from loguru import logger
from scipy.optimize import Bounds, minimize


@dataclass
class OptimizationConfig:
    """Configuration for price optimization"""

    min_price: float = 5.0
    max_price: float = 50.0
    max_price_change_pct: float = 0.20
    break_even_price: float = 8.0
    competitor_price: float | None = None
    competitor_weight: float = 0.3


@dataclass
class OptimizationResult:
    """Container for optimization results"""

    optimal_price: float
    expected_demand: float
    expected_revenue: float
    success: bool
    message: str
    iterations: int
    current_price: float
    price_change_pct: float


class PriceOptimizer:
    """Constrained price optimization engine"""

    def __init__(
        self,
        demand_function: Callable[[float], float],
        config: OptimizationConfig,
    ):
        """
        Initialize optimizer

        Args:
            demand_function: Function that maps price to expected demand
            config: Optimization configuration
        """
        self.demand_function = demand_function
        self.config = config

    def objective_function(self, price: float | np.ndarray) -> float:
        """
        Objective function to minimize (negative revenue)

        Args:
            price: Price point(s)

        Returns:
            Negative expected revenue (for minimization)
        """
        # Convert to scalar float
        price_scalar = float(price[0]) if isinstance(price, np.ndarray) else float(price)

        expected_demand = float(self.demand_function(price_scalar))
        expected_revenue = price_scalar * expected_demand

        # Add competitor penalty if competitor price is set
        if self.config.competitor_price is not None:
            competitor_penalty = self.config.competitor_weight * max(
                0, price_scalar - self.config.competitor_price
            )
            expected_revenue -= competitor_penalty * expected_demand

        return -expected_revenue

    def optimize(
        self,
        current_price: float,
        initial_guess: float | None = None,
    ) -> OptimizationResult:
        """
        Find optimal price subject to constraints

        Args:
            current_price: Current price before optimization
            initial_guess: Starting point for optimization

        Returns:
            OptimizationResult with optimal price and expected metrics
        """
        logger.info(f"Optimizing price (current: ${current_price:.2f})")

        # Set bounds for price change
        min_change = current_price * (1 - self.config.max_price_change_pct)
        max_change = current_price * (1 + self.config.max_price_change_pct)

        # Ensure bounds respect absolute min/max prices
        lower_bound = max(self.config.min_price, min_change, self.config.break_even_price)
        upper_bound = min(self.config.max_price, max_change)

        # Bounds expects scalar floats, not lists
        bounds = Bounds(lb=lower_bound, ub=upper_bound)

        # Initial guess
        x0 = np.array([current_price]) if initial_guess is None else np.array([initial_guess])

        # Optimize
        result = minimize(
            self.objective_function,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100},
        )

        optimal_price = float(result.x[0])
        expected_demand = float(self.demand_function(optimal_price))
        expected_revenue = optimal_price * expected_demand
        price_change_pct = (optimal_price - current_price) / current_price

        optimization_result = OptimizationResult(
            optimal_price=optimal_price,
            expected_demand=expected_demand,
            expected_revenue=expected_revenue,
            success=result.success,
            message=str(result.message),
            iterations=int(result.nit),
            current_price=current_price,
            price_change_pct=price_change_pct,
        )

        logger.info(f"Optimal price: ${optimal_price:.2f} (change: {price_change_pct * 100:.1f}%)")
        logger.info(f"Expected demand: {expected_demand:.1f} units")
        logger.info(f"Expected revenue: ${expected_revenue:.2f}")

        return optimization_result

    def analyze_price_range(
        self,
        current_price: float,
        price_range: tuple[float, float] = (5.0, 50.0),
        n_points: int = 100,
    ) -> dict[str, np.ndarray | float]:
        """
        Analyze revenue across price range

        Args:
            current_price: Current price
            price_range: Min and max prices to analyze
            n_points: Number of points in range

        Returns:
            Dictionary with prices, demand, and revenue arrays
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        demand = np.array([float(self.demand_function(float(p))) for p in prices])
        revenue = prices * demand

        # Mark current and optimal prices
        optimal_result = self.optimize(current_price)

        return {
            "prices": prices,
            "demand": demand,
            "revenue": revenue,
            "current_price": current_price,
            "optimal_price": optimal_result.optimal_price,
        }

    def plot_revenue_curve(
        self,
        current_price: float,
        save_path: str | None = None,
    ) -> None:
        """Plot revenue vs price curve"""
        import matplotlib.pyplot as plt

        analysis = self.analyze_price_range(current_price)

        # Extract values with proper types
        prices = cast(np.ndarray, analysis["prices"])
        demand = cast(np.ndarray, analysis["demand"])
        revenue = cast(np.ndarray, analysis["revenue"])
        current_price_val = cast(float, analysis["current_price"])
        optimal_price_val = cast(float, analysis["optimal_price"])

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Revenue curve
        ax1.plot(
            prices,
            revenue,
            color="blue",
            label="Expected Revenue",
            linewidth=2,
        )
        ax1.set_xlabel("Price ($)")
        ax1.set_ylabel("Revenue ($)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Demand curve on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(
            prices,
            demand,
            color="red",
            label="Expected Demand",
            linestyle="--",
            linewidth=2,
        )
        ax2.set_ylabel("Demand (units)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Mark current and optimal prices
        ax1.axvline(
            current_price_val,
            color="green",
            linestyle=":",
            label=f"Current: ${current_price_val:.2f}",
        )
        ax1.axvline(
            optimal_price_val,
            color="orange",
            linestyle="--",
            label=f"Optimal: ${optimal_price_val:.2f}",
        )

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title("Price Optimization Analysis")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved revenue curve to {save_path}")
        else:
            plt.show()
