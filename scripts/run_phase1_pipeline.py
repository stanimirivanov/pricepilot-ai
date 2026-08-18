"""
Phase 1 Pipeline: Data Generation and Validation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
from loguru import logger

from pricepilot.config.settings import Settings
from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.data.validation import DataValidator
from pricepilot.utils.mlflow_tracking import MLflowTracker


@click.command()
@click.option("--seed", default=42, help="Random seed", type=int)
@click.option(
    "--output-path",
    default="data/raw/car_wash_transactions.csv",
    help="Output CSV path",
)
def run_pipeline(seed: int, output_path: str):
    """Run complete Phase 1 pipeline"""

    settings = Settings()
    tracker = MLflowTracker(settings)

    with tracker.start_run("phase1_pipeline"):
        logger.info("Starting Phase 1 Pipeline")

        # Step 1: Generate data
        logger.info("Step 1: Generating synthetic data...")
        config = DataGeneratorConfig(seed=seed)
        generator = CarWashDataGenerator(config)
        df = generator.generate_and_save(output_path)

        # Log generation parameters
        tracker.log_params(
            {
                "seed": seed,
                "n_records": len(df),
                "generation_config": str(config),  # Convert to string for MLflow
            }
        )

        # Step 2: Validate data
        logger.info("Step 2: Validating data...")
        validator = DataValidator()
        validation_result = validator.validate(df)

        # Log validation results
        tracker.log_metrics(validation_result.metrics)
        tracker.log_params(
            {
                "validation_errors": len(validation_result.errors),
                "validation_warnings": len(validation_result.warnings),
            }
        )

        if not validation_result.is_valid:
            logger.error("Pipeline failed validation")
            tracker.log_params({"pipeline_status": "failed"})
            raise ValueError("Data validation failed")

        # Step 3: Log data artifact
        logger.info("Step 3: Logging data artifact...")
        tracker.log_dataframe(df, "raw_data")

        tracker.log_params({"pipeline_status": "success"})
        logger.info("Phase 1 Pipeline completed successfully")

        return df


if __name__ == "__main__":
    run_pipeline()
