"""End-to-end dynamic pricing pipeline"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from pricepilot.data.synthetic_data import CarWashDataGenerator, DataGeneratorConfig
from pricepilot.data.validation import DataValidator
from pricepilot.forecasting.config import ForecastingConfig
from pricepilot.forecasting.statsforecaster import StatsForecastForecaster
from pricepilot.models.anomaly_detector import AnomalyAwarePricingModel, DemandAnomalyDetector
from pricepilot.models.elasticity import PriceElasticityModel
from pricepilot.models.forecast_pricing import ForecastPricingModel
from pricepilot.optimization.price_optimizer import OptimizationConfig
from pricepilot.pipeline.config import PipelineConfig
from pricepilot.utils.mlflow_tracking import MLflowTracker


@dataclass
class PipelineResult:
    """Container for pipeline execution results"""

    timestamp: pd.Timestamp
    current_price: float
    forecasted_demand: float
    demand_interval: tuple
    optimal_price: float
    expected_revenue: float
    confidence: str
    anomaly_detected: bool
    anomaly_status: str
    price_change_pct: float
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "forecasted_demand": self.forecasted_demand,
            "demand_lower": self.demand_interval[0],
            "demand_upper": self.demand_interval[1],
            "optimal_price": self.optimal_price,
            "expected_revenue": self.expected_revenue,
            "confidence": self.confidence,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_status": self.anomaly_status,
            "price_change_pct": self.price_change_pct,
            "execution_time": self.execution_time,
        }


class PricingPipeline:
    """Complete dynamic pricing pipeline orchestrator"""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        enable_mlflow: bool = True,
    ):
        """
        Initialize pricing pipeline

        Args:
            config: Pipeline configuration
            enable_mlflow: Whether to track with MLflow
        """
        self.config = config or PipelineConfig()
        self.config.validate()
        self.enable_mlflow = enable_mlflow

        # Components (fitted lazily)
        self.data: pd.DataFrame | None = None
        self.elasticity_model: PriceElasticityModel | None = None
        self.forecaster: StatsForecastForecaster | None = None
        self.anomaly_detector: DemandAnomalyDetector | None = None
        self.pricing_model: ForecastPricingModel | None = None
        self.anomaly_pricing: AnomalyAwarePricingModel | None = None

        # MLflow tracker
        self.mlflow_tracker: MLflowTracker | None = None
        self.mlflow_run: Any | None = None

        logger.info("PricingPipeline initialized")

    @property
    def data_df(self) -> pd.DataFrame:
        """Get data as DataFrame, raising if not loaded"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_or_generate_data() first.")
        return self.data

    @property
    def forecaster_fitted(self) -> StatsForecastForecaster:
        """Get fitted forecaster, raising if not fitted"""
        if self.forecaster is None:
            raise ValueError("Forecaster not fitted. Call fit_models() first.")
        return self.forecaster

    @property
    def anomaly_pricing_fitted(self) -> AnomalyAwarePricingModel:
        """Get fitted anomaly pricing model, raising if not fitted"""
        if self.anomaly_pricing is None:
            raise ValueError("Models not fitted. Call fit_models() first.")
        return self.anomaly_pricing

    def load_or_generate_data(self, regenerate: bool = False) -> pd.DataFrame:
        """
        Load existing data or generate synthetic data

        Args:
            regenerate: Force regeneration of synthetic data

        Returns:
            DataFrame with historical data
        """
        from pathlib import Path

        data_path = Path(self.config.data_path)

        if data_path.exists() and not regenerate:
            logger.info(f"Loading data from {data_path}")
            self.data = pd.read_csv(data_path, parse_dates=["date"])
        else:
            logger.info("Generating synthetic data...")
            generator_config = DataGeneratorConfig(seed=42)
            generator = CarWashDataGenerator(generator_config)
            self.data = generator.generate_and_save(str(data_path))

        # Validate data
        validator = DataValidator()
        validation_result = validator.validate(self.data)

        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.errors}")

        logger.info(f"Loaded {len(self.data)} days of data")
        return self.data

    def fit_models(self) -> "PricingPipeline":
        """Fit all models in the pipeline"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_or_generate_data() first.")

        data = self.data  # Local variable for type narrowing

        # Start MLflow run
        if self.enable_mlflow:
            from pricepilot.config.settings import Settings

            settings = Settings()
            settings.mlflow_experiment_name = self.config.experiment_name
            self.mlflow_tracker = MLflowTracker(settings)
            self.mlflow_run = self.mlflow_tracker.start_run("pipeline_execution")
            self.mlflow_run.__enter__()

        # Step 1: Fit elasticity model
        logger.info("Step 1: Fitting elasticity model...")
        self.elasticity_model = PriceElasticityModel(
            samples=self.config.elasticity_samples,
            tune=self.config.elasticity_tune,
            chains=self.config.elasticity_chains,
        )
        self.elasticity_model.fit(
            prices=data["price"].values,
            demand=data["quantity_sold"].values,
            weather_features=data["is_sunny"].values,
            progressbar=False,
        )

        if self.enable_mlflow and self.mlflow_tracker and self.elasticity_model.results:
            self.mlflow_tracker.log_params(
                {
                    "elasticity_mean": self.elasticity_model.results.posterior_mean,
                    "elasticity_std": self.elasticity_model.results.posterior_std,
                }
            )

        # Step 2: Fit forecaster
        logger.info("Step 2: Fitting forecaster...")
        ts_data = pd.DataFrame(
            {
                "ds": data["date"],
                "y": data["quantity_sold"],
                "unique_id": "car_wash",
            }
        )
        forecast_config = ForecastingConfig(
            horizon=self.config.forecast_horizon,
            seasonality=self.config.forecast_seasonality,
            confidence_level=self.config.forecast_confidence,
        )
        self.forecaster = StatsForecastForecaster(config=forecast_config)
        self.forecaster.fit(ts_data)

        # Step 3: Fit anomaly detector
        logger.info("Step 3: Fitting anomaly detector...")
        self.anomaly_detector = DemandAnomalyDetector(
            contamination=self.config.anomaly_contamination,
        )
        self.anomaly_detector.fit(data["quantity_sold"].values)

        # Step 4: Create pricing model
        logger.info("Step 4: Creating pricing model...")
        optimizer_config = OptimizationConfig(
            min_price=self.config.min_price,
            max_price=self.config.max_price,
            break_even_price=self.config.break_even_price,
            max_price_change_pct=self.config.max_price_change_pct,
        )
        self.pricing_model = ForecastPricingModel(
            elasticity_model=self.elasticity_model,
            forecaster=self.forecaster,
            optimizer_config=optimizer_config,
        )

        # Step 5: Create anomaly-aware pricing model
        logger.info("Step 5: Creating anomaly-aware pricing model...")
        self.anomaly_pricing = AnomalyAwarePricingModel(
            pricing_model=self.pricing_model,
            anomaly_detector=self.anomaly_detector,
        )

        logger.info("All models fitted successfully")
        return self

    def get_tomorrow_price(
        self,
        current_price: float | None = None,
    ) -> PipelineResult:
        """
        Get optimal price for tomorrow

        Args:
            current_price: Current price (defaults to last historical price)

        Returns:
            PipelineResult with pricing decision
        """
        import time

        # Use properties for type safety
        anomaly_pricing = self.anomaly_pricing_fitted
        forecaster = self.forecaster_fitted
        data = self.data_df

        if current_price is None:
            current_price = float(data["price"].iloc[-1])

        start_time = time.time()

        # Generate forecast
        forecast = forecaster.predict(steps=1)
        forecasted_demand = float(forecast.mean[0])
        demand_lower = float(forecast.lower[0])
        demand_upper = float(forecast.upper[0])

        # Check for anomaly
        historical_demand = data["quantity_sold"].values[-30:]

        anomaly_result = anomaly_pricing.price_with_anomaly_check(
            current_price=current_price,
            historical_demand=historical_demand,
            forecasted_demand=forecasted_demand,
        )

        execution_time = time.time() - start_time

        result = PipelineResult(
            timestamp=pd.Timestamp.now(),
            current_price=current_price,
            forecasted_demand=forecasted_demand,
            demand_interval=(demand_lower, demand_upper),
            optimal_price=anomaly_result["pricing_result"].optimal_price,
            expected_revenue=anomaly_result["pricing_result"].expected_revenue,
            confidence=anomaly_result["pricing_result"].confidence,
            anomaly_detected=anomaly_result["is_anomaly"],
            anomaly_status=anomaly_result["anomaly_status"],
            price_change_pct=anomaly_result["pricing_result"].price_change_pct,
            execution_time=execution_time,
        )

        # Log to MLflow
        if self.enable_mlflow and self.mlflow_tracker:
            self.mlflow_tracker.log_metrics(
                {
                    "forecasted_demand": result.forecasted_demand,
                    "optimal_price": result.optimal_price,
                    "expected_revenue": result.expected_revenue,
                    "execution_time": result.execution_time,
                }
            )
            self.mlflow_tracker.log_params(
                {
                    "anomaly_status": result.anomaly_status,
                    "confidence": result.confidence,
                }
            )

        logger.info(f"Pipeline result: {result.to_dict()}")
        return result

    def get_week_prices(
        self,
        current_price: float | None = None,
    ) -> list[PipelineResult]:
        """Get optimal prices for the next 7 days"""
        data = self.data_df
        if current_price is None:
            current_price = float(data["price"].iloc[-1])

        results = []
        for day in range(7):
            result = self.get_tomorrow_price(current_price)
            results.append(result)
            current_price = result.optimal_price

        return results

    def close(self) -> None:
        """Close MLflow run if open"""
        if self.enable_mlflow and self.mlflow_run:
            self.mlflow_run.__exit__(None, None, None)
            logger.info("MLflow run closed")

    def __enter__(self) -> "PricingPipeline":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.close()
