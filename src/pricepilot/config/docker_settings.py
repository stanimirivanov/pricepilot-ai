"""Docker-specific configuration"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DockerSettings(BaseSettings):
    """Docker environment settings"""

    model_config = SettingsConfigDict(
        env_file=".env.docker",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    environment: str = "docker"
    log_level: str = "INFO"

    # Paths (Docker-specific)
    data_raw_dir: Path = Path("/app/data/raw")
    data_processed_dir: Path = Path("/app/data/processed")
    data_feedback_dir: Path = Path("/app/data/feedback")
    model_checkpoint_dir: Path = Path("/app/models/checkpoints")
    log_dir: Path = Path("/app/logs")

    # MLflow
    mlflow_tracking_uri: str = "sqlite:////app/mlflow/mlflow.db"
    mlflow_experiment_name: str = "pricepilot"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
