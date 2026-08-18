import os
import tempfile
from contextlib import contextmanager
from typing import Any

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient

from pricepilot.config.settings import Settings


class MLflowTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(self, settings: Settings):
        self.settings = settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        self.client = MlflowClient()

    @contextmanager
    def start_run(self, run_name: str, tags: dict[str, str] | None = None):
        """Context manager for MLflow runs"""
        with mlflow.start_run(run_name=run_name) as run:
            # Set default tags
            mlflow.set_tag("environment", self.settings.environment)
            if tags:
                mlflow.set_tags(tags)

            logger.info(f"MLflow run started: {run_name} (ID: {run.info.run_id})")
            try:
                yield run
                logger.info(f"MLflow run completed: {run_name}")
            except Exception as e:
                logger.error(f"MLflow run failed: {run_name} - {str(e)}")
                mlflow.set_tag("status", "failed")
                raise

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics"""
        mlflow.log_metrics(metrics)

    def log_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Log dataframe as artifact"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, name)
            os.unlink(f.name)

    def log_model(self, model: Any, name: str) -> None:
        """Log model artifact"""
        mlflow.sklearn.log_model(model, name)

    def get_experiment_id(self) -> str:
        """Get current experiment ID"""
        experiment = mlflow.get_experiment_by_name(self.settings.mlflow_experiment_name)
        return experiment.experiment_id if experiment else None

    def list_runs(self, max_results: int = 10) -> pd.DataFrame:
        """List recent runs"""
        runs = mlflow.search_runs(
            experiment_ids=[self.get_experiment_id()],
            max_results=max_results,
            order_by=["start_time DESC"],
        )
        return runs
