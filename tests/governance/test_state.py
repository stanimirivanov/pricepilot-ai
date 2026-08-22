"""Tests for governance state"""

from pricepilot.governance.state import ConfidenceLevel, GovernanceState, WorkflowState


def test_state_initialization():
    """Test initial state"""
    state = GovernanceState()

    assert state.current_state == WorkflowState.INGEST_DATA
    assert state.previous_state is None
    assert state.current_price is None
    assert state.optimal_price is None
    assert state.approved is False
    assert state.human_reviewed is False


def test_state_transition():
    """Test state transitions"""
    state = GovernanceState()

    state.transition_to(WorkflowState.FORECAST)
    assert state.current_state == WorkflowState.FORECAST
    assert state.previous_state == WorkflowState.INGEST_DATA

    state.transition_to(WorkflowState.OPTIMIZE)
    assert state.current_state == WorkflowState.OPTIMIZE
    assert state.previous_state == WorkflowState.FORECAST


def test_state_to_dict():
    """Test state to dictionary conversion"""
    state = GovernanceState()
    state.current_price = 15.0
    state.optimal_price = 20.0
    state.confidence_score = 0.95
    state.confidence_level = ConfidenceLevel.HIGH

    state_dict = state.to_dict()

    assert state_dict["current_state"] == "ingest_data"
    assert state_dict["current_price"] == 15.0
    assert state_dict["optimal_price"] == 20.0
    assert state_dict["confidence_score"] == 0.95
    assert state_dict["confidence_level"] == "high"
