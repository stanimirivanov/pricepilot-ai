"""Simple workflow example for testing"""

from langgraph.graph import END, START, StateGraph
from loguru import logger

from pricepilot.governance.state import GovernanceState, WorkflowState
from pricepilot.governance.workflow import BaseGovernanceWorkflow


class SimpleGovernanceWorkflow(BaseGovernanceWorkflow):
    """Simple governance workflow for testing"""

    def _add_nodes(self, graph: StateGraph) -> None:
        """Add nodes to the graph"""
        graph.add_node("ingest", self._ingest_data)
        graph.add_node("forecast", self._forecast)
        graph.add_node("optimize", self._optimize)
        graph.add_node("check_confidence", self._check_confidence)
        graph.add_node("approve", self._approve)
        graph.add_node("request_review", self._request_review)

    def _add_edges(self, graph: StateGraph) -> None:
        """Add edges to the graph"""
        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", "forecast")
        graph.add_edge("forecast", "optimize")
        graph.add_edge("optimize", "check_confidence")

        # Conditional routing based on confidence
        graph.add_conditional_edges(
            "check_confidence",
            self._route_decision,
            {
                "approve": "approve",
                "review": "request_review",
            },
        )

        graph.add_edge("approve", END)
        graph.add_edge("request_review", END)

    def _ingest_data(self, state: GovernanceState) -> GovernanceState:
        """Ingest data step"""
        logger.info("State: Ingest Data")
        state.transition_to(WorkflowState.INGEST_DATA)
        return state

    def _forecast(self, state: GovernanceState) -> GovernanceState:
        """Forecast step"""
        logger.info("State: Forecast")
        state.transition_to(WorkflowState.FORECAST)
        return state

    def _optimize(self, state: GovernanceState) -> GovernanceState:
        """Optimize step"""
        logger.info("State: Optimize")
        state.transition_to(WorkflowState.OPTIMIZE)
        return state

    def _check_confidence(self, state: GovernanceState) -> GovernanceState:
        """Check confidence step"""
        logger.info("State: Check Confidence")
        state.transition_to(WorkflowState.CHECK_CONFIDENCE)

        if state.confidence_score is None:
            state.confidence_score = 0.95

        return state

    def _route_decision(self, state: GovernanceState) -> str:
        """Route based on confidence"""
        if state.confidence_score and state.confidence_score >= 0.90:
            return "approve"
        return "review"

    def _approve(self, state: GovernanceState) -> GovernanceState:
        """Approve decision"""
        logger.info("State: Approved")
        state.approved = True
        state.transition_to(WorkflowState.APPROVED)
        return state

    def _request_review(self, state: GovernanceState) -> GovernanceState:
        """Request review"""
        logger.info("State: Request Review")
        state.approved = False
        state.transition_to(WorkflowState.REQUEST_REVIEW)
        return state
