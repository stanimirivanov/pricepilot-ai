"""Tests for LangGraph workflow with Pydantic state"""

from pricepilot.governance.simple_workflow import SimpleGovernanceWorkflow
from pricepilot.governance.state import GovernanceState, WorkflowState


def test_workflow_build():
    """Test workflow graph building"""
    workflow = SimpleGovernanceWorkflow()
    graph = workflow.build_graph()
    assert graph is not None


def test_workflow_execute_high_confidence():
    """Test workflow with high confidence"""
    workflow = SimpleGovernanceWorkflow()

    initial_state = GovernanceState(confidence_score=0.95)

    result = workflow.execute(initial_state)

    assert result.approved is True
    assert result.current_state == WorkflowState.COMPLETED
    assert result.error_message is None


def test_workflow_execute_low_confidence():
    """Test workflow with low confidence"""
    workflow = SimpleGovernanceWorkflow()

    initial_state = GovernanceState(confidence_score=0.50)

    result = workflow.execute(initial_state)

    assert result.approved is False
    assert result.current_state == WorkflowState.COMPLETED


def test_workflow_history():
    """Test workflow history tracking"""
    workflow = SimpleGovernanceWorkflow()
    initial_state = GovernanceState(confidence_score=0.95)

    workflow.execute(initial_state)

    history = workflow.get_history()
    assert len(history) > 0
    assert history[-1].current_state == WorkflowState.COMPLETED


def test_state_serialization():
    """Test state to dict and back"""
    state = GovernanceState(
        current_price=15.0,
        optimal_price=20.0,
        confidence_score=0.95,
    )

    # To dict
    state_dict = state.to_dict()
    assert state_dict["current_price"] == 15.0

    # From dict
    restored = GovernanceState.from_dict(state_dict)
    assert restored.current_price == 15.0
    assert restored.optimal_price == 20.0
