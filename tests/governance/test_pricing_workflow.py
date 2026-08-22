"""Tests for pricing governance workflow"""

import pytest

from pricepilot.governance.confidence import ConfidenceScorer
from pricepilot.governance.pricing_workflow import PricingGovernanceWorkflow
from pricepilot.governance.state import GovernanceState, WorkflowState
from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PricingPipeline


@pytest.fixture
def pipeline_config(tmp_path):
    """Create pipeline config for testing"""
    return PipelineConfig(
        data_path=str(tmp_path / "test_data.csv"),
        elasticity_samples=200,  # Reduced for testing
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
def workflow(fitted_pipeline):
    """Create governance workflow with fitted pipeline"""
    return PricingGovernanceWorkflow(
        pipeline=fitted_pipeline,
        confidence_scorer=ConfidenceScorer(),
    )


def test_workflow_initialization(workflow):
    """Test workflow initialization"""
    assert workflow.pipeline is not None
    assert workflow.confidence_scorer is not None


def test_workflow_build(workflow):
    """Test workflow graph building"""
    graph = workflow.build_graph()
    assert graph is not None


def test_workflow_execute_high_confidence(workflow):
    """Test workflow with high confidence"""
    # Set high confidence thresholds for test
    workflow.confidence_scorer = ConfidenceScorer(
        high_threshold=0.50,  # Very low threshold to force approval
        medium_threshold=0.30,
    )

    initial_state = GovernanceState()

    result = workflow.execute(initial_state)

    assert result.final_price is not None
    assert result.current_state == WorkflowState.COMPLETED
    assert result.error_message is None


def test_workflow_execute_low_confidence(workflow):
    """Test workflow with low confidence (should request review)"""
    # Set very high thresholds to force review
    workflow.confidence_scorer = ConfidenceScorer(
        high_threshold=0.99,  # Very high threshold to force review
        medium_threshold=0.95,
    )

    initial_state = GovernanceState()

    result = workflow.execute(initial_state)

    assert result.current_state == WorkflowState.COMPLETED
    assert result.human_reviewed is True or result.approved is False


def test_workflow_with_human_override(workflow):
    """Test workflow with human override"""
    # Set high thresholds to force review
    workflow.confidence_scorer = ConfidenceScorer(
        high_threshold=0.99,
        medium_threshold=0.95,
    )

    initial_state = GovernanceState()
    initial_state.human_override_price = 18.50  # Human sets price

    result = workflow.execute(initial_state)

    assert result.final_price == 18.50
    assert result.human_reviewed is True


def test_workflow_result_dict(workflow):
    """Test workflow result conversion"""
    workflow.confidence_scorer = ConfidenceScorer(
        high_threshold=0.50,
        medium_threshold=0.30,
    )

    result = workflow.execute(GovernanceState())

    result_dict = result.to_dict()
    assert "final_price" in result_dict
    assert "approved" in result_dict
    assert "confidence_score" in result_dict


def test_workflow_state_transitions(workflow):
    """Test workflow state transitions"""
    workflow.confidence_scorer = ConfidenceScorer(
        high_threshold=0.50,
        medium_threshold=0.30,
    )

    result = workflow.execute(GovernanceState())

    # Should end in COMPLETED state
    assert result.current_state == WorkflowState.COMPLETED

    # History should have been recorded
    history = workflow.get_history()
    assert len(history) > 0
