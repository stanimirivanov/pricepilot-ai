"""Check MLflow and protobuf compatibility"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger


def check_mlflow():
    """Check MLflow installation"""
    try:
        import mlflow

        logger.info(f"MLflow version: {mlflow.__version__}")

        import google.protobuf

        logger.info(f"Protobuf version: {google.protobuf.__version__}")

        # Test MLflow import
        from mlflow.tracking import MlflowClient

        logger.info("MLflow imports successfully")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Try running: uv add 'protobuf<5.0.0'")
        sys.exit(1)


if __name__ == "__main__":
    check_mlflow()
