import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from loguru import logger

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig

app = typer.Typer()


@app.command()
def generate(
    output_path: str = typer.Option("data/raw/car_wash_transactions.csv", help="Output CSV path"),
    start_date: str = typer.Option("2022-01-01", help="Start date"),
    end_date: str = typer.Option("2023-12-31", help="End date"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """Generate synthetic car wash data"""
    logger.info("Generating synthetic car wash data...")

    config = DataGeneratorConfig(start_date=start_date, end_date=end_date, seed=seed)

    generator = CarWashDataGenerator(config)
    df = generator.generate_and_save(output_path)

    logger.info(f"Generated {len(df)} records")
    logger.info(f"Saved to {output_path}")

    # Log summary statistics
    logger.info("\nData Summary:")
    logger.info(f"Average price: ${df['price'].mean():.2f}")
    logger.info(f"Average daily demand: {df['quantity_sold'].mean():.0f} cars")
    logger.info(f"Average daily revenue: ${df['revenue'].mean():.2f}")
    logger.info(f"Rainy days: {df['is_raining'].sum()} ({df['is_raining'].mean() * 100:.1f}%)")

    return df


if __name__ == "__main__":
    app()
