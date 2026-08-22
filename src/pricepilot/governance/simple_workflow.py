"""Simple workflow example for testing"""

from typing import Any

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

    def _ingest_data(self, state: GovernanceState) -> dict[str, Any]:
        """Ingest data step"""
        logger.info("State: Ingest Data")
        return {"current_state": WorkflowState.INGEST_DATA.value}

    def _forecast(self, state: GovernanceState) -> dict[str, Any]:
        """Forecast step"""
        logger.info("State: Forecast")
        return {"current_state": WorkflowState.FORECAST.value}

    def _optimize(self, state: GovernanceState) -> dict[str, Any]:
        """Optimize step"""
        logger.info("State: Optimize")
        return {"current_state": WorkflowState.OPTIMIZE.value}

    def _check_confidence(self, state: GovernanceState) -> dict[str, Any]:
        """Check confidence step"""
        logger.info("State: Check Confidence")

        # Explicitly type as Dict[str, Any]
        updates: dict[str, Any] = {"current_state": WorkflowState.CHECK_CONFIDENCE.value}

        # Set default confidence for testing if not provided
        if state.confidence_score is None:
            updates["confidence_score"] = 0.95

        return updates

    def _route_decision(self, state: GovernanceState) -> str:
        """Route based on confidence"""
        if state.confidence_score and state.confidence_score >= 0.90:
            return "approve"
        return "review"

    def _approve(self, state: GovernanceState) -> dict[str, Any]:
        """Approve decision"""
        logger.info("State: Approved")
        return {
            "approved": True,
            "current_state": WorkflowState.APPROVED.value,
        }

    def _request_review(self, state: GovernanceState) -> dict[str, Any]:
        """Request review"""
        logger.info("State: Request Review")
        return {
            "approved": False,
            "current_state": WorkflowState.REQUEST_REVIEW.value,
        }
