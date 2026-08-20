"""Run demand forecasting example"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import pandas as pd
from loguru import logger

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.forecasting.config import ForecastingConfig
from pricepilot.forecasting.statsforecaster import StatsForecastForecaster


@click.command()
@click.option("--horizon", default=7, help="Forecast horizon (days)")
@click.option("--model", default="auto_arima", help="Model type")
def run_forecast(horizon: int, model: str):
    """Run demand forecasting example"""

    # Generate synthetic data
    logger.info("Generating synthetic data...")
    config = DataGeneratorConfig(
        start_date="2022-01-01",
        end_date="2023-12-31",
        seed=42,
    )
    generator = CarWashDataGenerator(config)
    data = generator.generate()

    # Prepare time series
    ts_data = pd.DataFrame(
        {
            "ds": data["date"],
            "y": data["quantity_sold"],
            "unique_id": "car_wash_demand",
        }
    )

    # Split data
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data.iloc[:train_size]
    test_data = ts_data.iloc[train_size : train_size + horizon]

    # Create forecaster
    forecast_config = ForecastingConfig(
        horizon=horizon,
        seasonality=7,
        confidence_level=0.9,
    )
    forecaster = StatsForecastForecaster(
        config=forecast_config,
        models=[model],
    )

    # Fit
    logger.info(f"Fitting {model} model...")
    forecaster.fit(train_data)

    # Predict
    logger.info(f"Generating {horizon}-day forecast...")
    forecast = forecaster.predict(steps=horizon)

    # Evaluate
    if len(test_data) == horizon:
        metrics = forecaster.evaluate_accuracy(test_data)
        logger.info(f"Forecast accuracy: {metrics}")

    # Display results
    forecast_df = forecast.to_dataframe()
    logger.info("\nForecast Results:")
    for _, row in forecast_df.iterrows():
        logger.info(
            f"  {row['date'].strftime('%Y-%m-%d')}: "
            f"{row['forecast']:.0f} "
            f"[{row['lower_bound']:.0f} - {row['upper_bound']:.0f}]"
        )

    # Plot
    forecaster.plot_forecast(
        train_data.iloc[-30:],  # Last 30 days of training
        forecast,
        save_path="forecast_plot.png",
    )
    logger.info("Forecast plot saved to forecast_plot.png")

    return forecast


if __name__ == "__main__":
    run_forecast()
