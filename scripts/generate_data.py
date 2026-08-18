import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
from loguru import logger

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig


@click.command()
@click.option(
    "--output-path",
    default="data/raw/car_wash_transactions.csv",
    help="Output file path (CSV or parquet)",
)
@click.option("--start-date", default="2022-01-01", help="Start date")
@click.option("--end-date", default="2023-12-31", help="End date")
@click.option("--seed", default=42, help="Random seed for reproducibility", type=int)
@click.option(
    "--format", type=click.Choice(["csv", "parquet"]), default="csv", help="Output format"
)
def generate(
    output_path: str,
    start_date: str,
    end_date: str,
    seed: int,
    format: str,
):
    """Generate synthetic car wash data"""
    logger.info("Generating synthetic car wash data...")

    # Adjust output path based on format if not explicitly specified
    if format == "parquet" and output_path.endswith(".csv"):
        output_path = output_path.replace(".csv", ".parquet")
    elif format == "csv" and output_path.endswith(".parquet"):
        output_path = output_path.replace(".parquet", ".csv")

    config = DataGeneratorConfig(
        start_date=start_date,
        end_date=end_date,
        seed=seed,
    )

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
