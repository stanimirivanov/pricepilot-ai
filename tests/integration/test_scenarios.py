"""End-to-end scenario tests"""

import json

import pandas as pd
import pytest

from pricepilot.governance.confidence import ConfidenceScorer
from pricepilot.governance.pricing_workflow import PricingGovernanceWorkflow
from pricepilot.governance.state import GovernanceState
from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PricingPipeline


class TestPricingScenarios:
    """Test complete pricing scenarios"""

    @pytest.fixture
    def setup_pipeline(self, tmp_path):
        """Setup pipeline for scenario tests"""
        config = PipelineConfig(
            data_path=str(tmp_path / "scenario_data.csv"),
            elasticity_samples=200,
            elasticity_tune=100,
            elasticity_chains=1,
            forecast_horizon=3,
            track_with_mlflow=False,
        )
        pipeline = PricingPipeline(config=config, enable_mlflow=False)
        pipeline.load_or_generate_data()
        pipeline.fit_models()
        return pipeline

    def test_normal_day_scenario(self, setup_pipeline):
        """Test pricing on a normal day"""
        workflow = PricingGovernanceWorkflow(
            pipeline=setup_pipeline,
            confidence_scorer=ConfidenceScorer(high_threshold=0.50, medium_threshold=0.30),
        )

        result = workflow.execute(GovernanceState())

        assert result.approved is True
        assert result.final_price is not None
        assert result.final_price > 0

    def test_high_uncertainty_scenario(self, setup_pipeline):
        """Test pricing with high uncertainty (pending review)"""
        workflow = PricingGovernanceWorkflow(
            pipeline=setup_pipeline,
            confidence_scorer=ConfidenceScorer(high_threshold=0.99, medium_threshold=0.95),
        )

        result = workflow.execute(GovernanceState())

        # Should be pending review
        assert result.approved is False
        assert result.human_reviewed is True
        assert result.current_state == WorkflowState.PENDING_REVIEW

    def test_human_override_scenario(self, setup_pipeline):
        """Test scenario with human override"""
        workflow = PricingGovernanceWorkflow(
            pipeline=setup_pipeline,
            confidence_scorer=ConfidenceScorer(high_threshold=0.99, medium_threshold=0.95),
        )

        state = GovernanceState()
        state.human_override_price = 19.99

        result = workflow.execute(state)

        assert result.final_price == 19.99
        assert result.human_reviewed is True
        assert result.current_state == WorkflowState.COMPLETED

    def test_complete_decision_log(self, setup_pipeline, tmp_path):
        """Test complete decision logging"""
        workflow = PricingGovernanceWorkflow(
            pipeline=setup_pipeline,
            confidence_scorer=ConfidenceScorer(high_threshold=0.50, medium_threshold=0.30),
        )

        result = workflow.execute(GovernanceState())

        # Create decision log
        decision_log = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "final_price": result.final_price,
            "approved": result.approved,
            "confidence_score": result.confidence_score,
            "confidence_level": result.confidence_level.value if result.confidence_level else None,
            "forecasted_demand": result.forecasted_demand,
            "optimal_price": result.optimal_price,
            "anomaly_status": result.anomaly_status,
        }

        # Save log
        log_path = tmp_path / "decision_log.json"
        with open(log_path, "w") as f:
            json.dump(decision_log, f, indent=2)

        # Verify log
        assert log_path.exists()
        with open(log_path) as f:
            loaded = json.load(f)

        assert loaded["final_price"] == result.final_price
        assert loaded["approved"] == result.approved
