"""End-to-end integration tests for governance workflow"""

from pathlib import Path

import pandas as pd
import pytest

from pricepilot.api.feedback import FeedbackStore
from pricepilot.governance.confidence import ConfidenceScorer, ConfidenceThresholds
from pricepilot.governance.pricing_workflow import PricingGovernanceWorkflow
from pricepilot.governance.state import ConfidenceLevel, GovernanceState, WorkflowState
from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PricingPipeline


@pytest.fixture
def pipeline_config(tmp_path):
    """Create pipeline config for testing"""
    return PipelineConfig(
        data_path=str(tmp_path / "test_data.csv"),
        elasticity_samples=200,
        elasticity_tune=100,
        elasticity_chains=1,
        forecast_horizon=3,
        anomaly_contamination=0.05,
        track_with_mlflow=False,
    )


@pytest.fixture
def fitted_pipeline(pipeline_config):
    """Create and fit pipeline"""
    pipeline = PricingPipeline(
        config=pipeline_config,
        enable_mlflow=False,
    )
    pipeline.load_or_generate_data()
    pipeline.fit_models()
    return pipeline


@pytest.fixture
def governance_workflow(fitted_pipeline):
    """Create governance workflow with fitted pipeline"""
    return PricingGovernanceWorkflow(
        pipeline=fitted_pipeline,
        confidence_scorer=ConfidenceThresholds.balanced(),
    )


class TestGovernanceWorkflowE2E:
    """End-to-end tests for governance workflow"""

    def test_complete_workflow_high_confidence(self, governance_workflow):
        """Test complete workflow with high confidence"""
        # Force high confidence
        governance_workflow.confidence_scorer = ConfidenceScorer(
            high_threshold=0.50,
            medium_threshold=0.30,
        )

        # Execute
        result = governance_workflow.execute(GovernanceState())

        # Validate result
        assert result is not None
        assert result.current_state == WorkflowState.COMPLETED
        assert result.error_message is None
        assert result.final_price is not None
        assert result.approved is True
        assert result.human_reviewed is False

        # Validate state fields
        assert result.forecasted_demand is not None
        assert result.optimal_price is not None
        assert result.confidence_score is not None

    def test_complete_workflow_low_confidence(self, governance_workflow):
        """Test complete workflow with low confidence"""
        # Force low confidence
        governance_workflow.confidence_scorer = ConfidenceScorer(
            high_threshold=0.99,
            medium_threshold=0.95,
        )

        # Execute
        result = governance_workflow.execute(GovernanceState())

        # Validate
        assert result.current_state == WorkflowState.COMPLETED
        assert result.approved is False
        assert result.human_reviewed is True

    def test_workflow_state_history(self, governance_workflow):
        """Test that state history is tracked"""
        governance_workflow.confidence_scorer = ConfidenceScorer(
            high_threshold=0.50,
            medium_threshold=0.30,
        )

        governance_workflow.execute(GovernanceState())

        history = governance_workflow.get_history()
        assert len(history) > 0

        # Last state should be completed
        assert history[-1].current_state == WorkflowState.COMPLETED


class TestFeedbackLoop:
    """Tests for continuous learning feedback loop"""

    def test_feedback_logging(self, tmp_path):
        """Test feedback logging"""
        store = FeedbackStore(storage_path=str(tmp_path / "feedback.jsonl"))

        record = {
            "override_id": "test-001",
            "recommendation_id": "rec-001",
            "original_price": 15.0,
            "human_price": 18.0,
            "timestamp": "2024-01-01T12:00:00",
        }

        store.log_override(record)

        records = store.get_all_feedback()
        assert len(records) == 1
        assert records[0]["override_id"] == "test-001"

    def test_feedback_to_dataframe(self, tmp_path):
        """Test converting feedback to DataFrame"""
        store = FeedbackStore(storage_path=str(tmp_path / "feedback.jsonl"))

        # Log multiple records
        for i in range(3):
            store.log_override(
                {
                    "override_id": f"test-{i}",
                    "original_price": 15.0 + i,
                    "human_price": 18.0 + i,
                }
            )

        df = store.get_feedback_dataframe()
        assert len(df) == 3
        assert "override_id" in df.columns
        assert "original_price" in df.columns
        assert "human_price" in df.columns

    def test_append_to_training_data(self, tmp_path):
        """Test appending feedback to training data"""
        # Create training data
        training_path = tmp_path / "training.csv"
        training_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "price": [15.0] * 10,
                "quantity_sold": [100] * 10,
                "revenue": [1500.0] * 10,
                "is_raining": [False] * 10,
                "is_sunny": [True] * 10,
                "temperature": [70.0] * 10,
                "day_of_week": range(10),
                "is_weekend": [False] * 10,
                "month": [1] * 10,
                "year": [2023] * 10,
            }
        )
        training_data.to_csv(training_path, index=False)

        # Create feedback store
        store = FeedbackStore(storage_path=str(tmp_path / "feedback.jsonl"))

        # Log feedback with actual demand
        store.log_override(
            {
                "timestamp": "2024-01-01T12:00:00",
                "date": "2024-01-11",
                "original_price": 15.0,
                "final_price": 18.0,
                "forecasted_demand": 100,
                "actual_demand": 95,
                "confidence_score": 0.85,
                "anomaly_detected": False,
                "human_override": True,
            }
        )

        # Append to training
        updated = store.append_to_training_data(str(training_path))

        assert len(updated) > 10  # Should have new rows
        assert updated.iloc[-1]["price"] == 18.0
        assert updated.iloc[-1]["quantity_sold"] == 95


class TestSecurity:
    """Security validation tests"""

    def test_dockerfile_non_root(self):
        """Test that Dockerfile creates non-root user"""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()

        assert "useradd" in content
        assert "USER appuser" in content

    def test_dockerfile_multistage(self):
        """Test that Dockerfile uses multi-stage build"""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()

        assert "AS builder" in content
        assert "COPY --from=builder" in content

    def test_dockerfile_healthcheck(self):
        """Test that Dockerfile has health check"""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()

        assert "HEALTHCHECK" in content

    def test_dockerignore_no_secrets(self):
        """Test that .dockerignore excludes secrets"""
        dockerignore = Path(".dockerignore")
        content = dockerignore.read_text()

        assert ".env" in content
        assert ".git" in content

    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in source"""
        from pathlib import Path

        secret_patterns = [
            "api_key",
            "password",
            "secret",
            "token",
        ]

        src_dir = Path("src")
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in secret_patterns:
                # Look for hardcoded assignments
                if f'{pattern} = "' in content or f"{pattern} = '" in content:
                    pytest.fail(f"Potential secret found in {py_file}: {pattern}")


class TestAPIIntegration:
    """API integration tests"""

    @pytest.mark.asyncio
    async def test_api_workflow_integration(self):
        """Test API with governance workflow"""
        from httpx import ASGITransport, AsyncClient

        from pricepilot.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            # Health check
            health_response = await client.get("/health")
            assert health_response.status_code == 200

            # Get recommendation
            rec_response = await client.get("/recommendation")
            assert rec_response.status_code == 200
            rec_data = rec_response.json()

            # Validate recommendation structure
            assert "recommendation_id" in rec_data
            assert "optimal_price" in rec_data
            assert "confidence_score" in rec_data
            assert "requires_review" in rec_data

            # Submit override if review required
            if rec_data["requires_review"]:
                override_data = {
                    "recommendation_id": rec_data["recommendation_id"],
                    "human_price": 18.50,
                    "notes": "Integration test override",
                    "reviewer_name": "Test User",
                }

                override_response = await client.post("/override", json=override_data)
                assert override_response.status_code == 200
                override_data_result = override_response.json()

                assert override_data_result["human_price"] == 18.50
                assert override_data_result["status"] == "OVERRIDDEN"


class TestPipelineGovernanceIntegration:
    """Pipeline and governance integration tests"""

    def test_pipeline_feeds_governance(self, fitted_pipeline, governance_workflow):
        """Test that pipeline data flows into governance"""
        # Execute workflow
        result = governance_workflow.execute(GovernanceState())

        # Validate that pipeline data was used
        assert result.forecasted_demand is not None
        assert result.optimal_price is not None

        # Pipeline should have data
        assert fitted_pipeline.data is not None
        assert len(fitted_pipeline.data) > 0

    def test_confidence_from_pipeline(self, governance_workflow):
        """Test that confidence uses pipeline parameters"""
        governance_workflow.confidence_scorer = ConfidenceScorer(
            high_threshold=0.50,
            medium_threshold=0.30,
        )

        result = governance_workflow.execute(GovernanceState())

        assert result.confidence_score is not None
        assert 0 <= result.confidence_score <= 1
        assert result.confidence_level in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
        ]
