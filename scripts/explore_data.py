# scripts/explore_data.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

from pricepilot.config.settings import Settings
from pricepilot.utils.mlflow_tracking import MLflowTracker

app = typer.Typer()


@app.command()
def explore(
    data_path: str = typer.Option("data/raw/car_wash_transactions.csv"),
    output_dir: str = typer.Option("data/processed/visualizations"),
):
    """Explore and log data statistics to MLflow"""

    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Initialize MLflow
    settings = Settings()
    tracker = MLflowTracker(settings)

    with tracker.start_run("data_exploration"):
        # Log basic statistics
        tracker.log_params(
            {
                "data_path": data_path,
                "n_records": len(df),
                "date_range": f"{df['date'].min()} to {df['date'].max()}",
            }
        )

        tracker.log_metrics(
            {
                "avg_daily_demand": df["quantity_sold"].mean(),
                "avg_price": df["price"].mean(),
                "avg_revenue": df["revenue"].mean(),
                "rainy_day_pct": df["is_raining"].mean() * 100,
                "weekend_demand_ratio": (
                    df[df["is_weekend"]]["quantity_sold"].mean()
                    / df[~df["is_weekend"]]["quantity_sold"].mean()
                ),
            }
        )

        # Create visualizations
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Time series plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(df["date"], df["price"])
        axes[0].set_title("Price Over Time")
        axes[0].set_ylabel("Price ($)")

        axes[1].plot(df["date"], df["quantity_sold"])
        axes[1].set_title("Daily Demand")
        axes[1].set_ylabel("Cars Washed")

        axes[2].plot(df["date"], df["revenue"])
        axes[2].set_title("Daily Revenue")
        axes[2].set_ylabel("Revenue ($)")

        plt.tight_layout()
        fig.savefig(f"{output_dir}/time_series.png")
        mlflow.log_artifact(f"{output_dir}/time_series.png")
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_cols = [
            "price",
            "quantity_sold",
            "temperature",
            "is_raining",
            "is_sunny",
            "is_weekend",
        ]
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Feature Correlations")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlations.png")
        mlflow.log_artifact(f"{output_dir}/correlations.png")
        plt.close()

        logger.info("Data exploration complete. Check MLflow UI for results.")


if __name__ == "__main__":
    app()
