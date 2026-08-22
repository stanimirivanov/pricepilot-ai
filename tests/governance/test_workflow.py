"""Tests for LangGraph workflow"""

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

    initial_state = GovernanceState()
    initial_state.confidence_score = 0.95

    result = workflow.execute(initial_state)

    assert result.approved is True
    assert result.current_state == WorkflowState.COMPLETED
    assert result.error_message is None


def test_workflow_execute_low_confidence():
    """Test workflow with low confidence"""
    workflow = SimpleGovernanceWorkflow()

    initial_state = GovernanceState()
    initial_state.confidence_score = 0.50

    result = workflow.execute(initial_state)

    assert result.approved is False
    assert result.current_state == WorkflowState.COMPLETED
    assert result.error_message is None


def test_workflow_history():
    """Test workflow history tracking"""
    workflow = SimpleGovernanceWorkflow()
    initial_state = GovernanceState()
    initial_state.confidence_score = 0.95

    workflow.execute(initial_state)

    history = workflow.get_history()
    assert len(history) > 0
    assert history[-1].current_state == WorkflowState.COMPLETED
