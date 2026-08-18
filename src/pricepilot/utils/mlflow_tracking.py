import io
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import mlflow
import pandas as pd
from loguru import logger

from pricepilot.config.settings import Settings


class MLflowTracker:
    """MLflow experiment tracking wrapper with auto-creation"""

    def __init__(self, settings: Settings):
        self.settings = settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        # Auto-create experiment on initialization
        self._ensure_experiment_exists()

    def _ensure_experiment_exists(self) -> None:
        """Ensure the experiment exists, create if necessary"""
        experiment = mlflow.get_experiment_by_name(self.settings.mlflow_experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.settings.mlflow_experiment_name)
            logger.info(f"Created new MLflow experiment: {self.settings.mlflow_experiment_name}")
        else:
            logger.debug(
                f"Using existing MLflow experiment: {self.settings.mlflow_experiment_name}"
            )

    @contextmanager
    def start_run(
        self, run_name: str, tags: dict[str, str] | None = None
    ) -> Generator[mlflow.ActiveRun, None, None]:
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
        """Log dataframe as artifact using StringIO"""
        # Ensure name has .csv extension for log_text
        if not name.endswith(".csv"):
            name = f"{name}.csv"

        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        # Log as text artifact
        mlflow.log_text(csv_buffer.getvalue(), name)
        logger.debug(f"Logged dataframe '{name}' as CSV artifact")

    def log_model(self, model: Any, name: str) -> None:
        """Log model artifact"""
        from mlflow.sklearn import log_model as sklearn_log_model

        sklearn_log_model(model, name)

    def get_experiment_id(self) -> str:
        """Get current experiment ID (guaranteed to exist)"""
        experiment = mlflow.get_experiment_by_name(self.settings.mlflow_experiment_name)
        if experiment is None:
            # This should never happen after initialization, but just in case
            raise RuntimeError(f"Experiment '{self.settings.mlflow_experiment_name}' not found")
        return experiment.experiment_id

    def list_runs(self, max_results: int = 10) -> pd.DataFrame:
        """List recent runs as DataFrame"""
        experiment_id = self.get_experiment_id()
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"],
            output_format="pandas",  # Explicitly request pandas DataFrame
        )

        # Use cast to tell type checker this is definitely a DataFrame
        return cast(pd.DataFrame, runs_df)
