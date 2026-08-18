import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
from loguru import logger

from pricepilot.config.settings import Settings
from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.utils.logging_config import setup_logging


@click.command()
@click.option(
    "--output-path",
    default=None,
    help="Output CSV path (defaults to settings.data_raw_dir/car_wash_transactions.csv)",
)
@click.option("--start-date", default=None, help="Start date (defaults to 2022-01-01)")
@click.option("--end-date", default=None, help="End date (defaults to 2023-12-31)")
@click.option("--seed", default=None, help="Random seed (defaults to 42)", type=int)
@click.option("--base-price", default=None, help="Base price (defaults to $15.00)", type=float)
@click.option(
    "--price-elasticity", default=None, help="Price elasticity (defaults to -2.0)", type=float
)
def generate(
    output_path: str | None,
    start_date: str | None,
    end_date: str | None,
    seed: int | None,
    base_price: float | None,
    price_elasticity: float | None,
):
    """Generate synthetic car wash data"""
    # Load settings
    settings = Settings()

    # Setup logging
    setup_logging(settings)

    # Use settings for defaults
    if output_path is None:
        output_path = str(settings.data_raw_dir / "car_wash_transactions.csv")

    logger.info("Generating synthetic car wash data...")

    # Build config with CLI overrides
    config_kwargs = {}
    if start_date:
        config_kwargs["start_date"] = start_date
    if end_date:
        config_kwargs["end_date"] = end_date
    if seed is not None:
        config_kwargs["seed"] = seed
    if base_price is not None:
        config_kwargs["base_price"] = base_price
    if price_elasticity is not None:
        config_kwargs["price_elasticity"] = price_elasticity

    config = DataGeneratorConfig(**config_kwargs)

    logger.info(f"Configuration: {config}")

    generator = CarWashDataGenerator(config)
    df = generator.generate_and_save(output_path)

    logger.info(f"Generated {len(df)} records")
    logger.info(f"Saved to {output_path}")

    # Log summary statistics
    logger.info("Data Summary:")
    logger.info(f"Average price: ${df['price'].mean():.2f}")
    logger.info(f"Average daily demand: {df['quantity_sold'].mean():.0f} cars")
    logger.info(f"Average daily revenue: ${df['revenue'].mean():.2f}")
    logger.info(f"Rainy days: {df['is_raining'].sum()} ({df['is_raining'].mean() * 100:.1f}%)")

    return df


if __name__ == "__main__":
    generate()
