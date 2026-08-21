"""Run the complete pricing pipeline"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

import click
from loguru import logger

from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PricingPipeline


@click.command()
@click.option("--data-path", default="data/raw/car_wash_transactions.csv")
@click.option("--regenerate", is_flag=True, help="Regenerate synthetic data")
@click.option("--current-price", default=None, type=float, help="Current price")
@click.option("--enable-mlflow/--disable-mlflow", default=True, help="Enable MLflow tracking")
@click.option("--output", default=None, help="Output JSON file for results")
def run_pipeline(
    data_path: str,
    regenerate: bool,
    current_price: float | None,
    enable_mlflow: bool,
    output: str | None,
):
    """Run complete dynamic pricing pipeline"""

    config = PipelineConfig(
        data_path=data_path,
        track_with_mlflow=enable_mlflow,
    )

    with PricingPipeline(config=config, enable_mlflow=enable_mlflow) as pipeline:
        # Load data
        pipeline.load_or_generate_data(regenerate=regenerate)

        # Fit models
        pipeline.fit_models()

        # Get tomorrow's price
        result = pipeline.get_tomorrow_price(current_price=current_price)

        # Print result
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE RESULT - TOMORROW'S PRICING DECISION")
        logger.info("=" * 60)
        logger.info(f"Current Price:     ${result.current_price:.2f}")
        logger.info(f"Forecasted Demand: {result.forecasted_demand:.0f} units")
        logger.info(
            f"Demand Interval:   [{result.demand_interval[0]:.0f}, {result.demand_interval[1]:.0f}]"
        )
        logger.info(f"Optimal Price:     ${result.optimal_price:.2f}")
        logger.info(f"Expected Revenue:  ${result.expected_revenue:.2f}")
        logger.info(f"Confidence:        {result.confidence.upper()}")
        logger.info(f"Anomaly Status:    {result.anomaly_status}")
        logger.info(f"Price Change:      {result.price_change_pct * 100:+.1f}%")
        logger.info(f"Execution Time:    {result.execution_time:.2f}s")
        logger.info("=" * 60)

        # Save to JSON if requested
        if output:
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Results saved to {output}")

        return result


if __name__ == "__main__":
    run_pipeline()
