"""Pipeline configuration"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for the complete pricing pipeline"""

    # Data settings
    data_path: str = "data/raw/car_wash_transactions.csv"

    # Model settings
    elasticity_samples: int = 1000
    elasticity_tune: int = 500
    elasticity_chains: int = 2

    # Forecasting settings
    forecast_horizon: int = 7
    forecast_seasonality: int = 7
    forecast_confidence: float = 0.9

    # Anomaly detection settings
    anomaly_contamination: float = 0.05
    anomaly_window: int = 7

    # Pricing settings
    min_price: float = 5.0
    max_price: float = 50.0
    break_even_price: float = 8.0
    max_price_change_pct: float = 0.30

    # MLflow settings
    track_with_mlflow: bool = True
    experiment_name: str = "pricepilot_pipeline"

    def validate(self) -> None:
        """Validate configuration"""
        if self.forecast_horizon < 1:
            raise ValueError("Forecast horizon must be at least 1")
        if self.min_price >= self.max_price:
            raise ValueError("Min price must be less than max price")
        if not (0 < self.anomaly_contamination < 0.5):
            raise ValueError("Anomaly contamination must be between 0 and 0.5")
