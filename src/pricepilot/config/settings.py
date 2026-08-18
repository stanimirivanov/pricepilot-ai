from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model hyperparameters"""

    elasticity_prior_mean: float = -2.0
    elasticity_prior_std: float = 1.0
    forecast_horizon: int = 7
    min_price: float = 5.0
    max_price: float = 50.0
    max_daily_price_change_pct: float = 0.20


class BusinessRules(BaseModel):
    """Business constraints"""

    break_even_price: float = 8.0
    competitor_weight: float = 0.3
    weather_sensitivity: float = 10.0
    weekend_multiplier: float = 1.2


class Settings(BaseSettings):
    """Main application settings"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    # Paths
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    model_checkpoint_dir: Path = Path("models/checkpoints")

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "pricepilot"

    # Components
    model: ModelConfig = Field(default_factory=ModelConfig)
    business_rules: BusinessRules = Field(default_factory=BusinessRules)

    @classmethod
    def load_yaml(cls, yaml_path: Path) -> "Settings":
        """Load settings from YAML file"""
        with open(yaml_path) as f:
            yaml_config = yaml.safe_load(f)
        return cls(**yaml_config)
