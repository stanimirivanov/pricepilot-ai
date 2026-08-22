"""Workflow definitions for governance using LangGraph 1.x"""

from abc import ABC, abstractmethod
from typing import Any, cast

from langgraph.graph import StateGraph
from loguru import logger

from pricepilot.governance.state import GovernanceState, WorkflowState


class BaseGovernanceWorkflow(ABC):
    """Base governance workflow using LangGraph"""

    def __init__(self):
        self.state = GovernanceState()
        self.history: list[GovernanceState] = []
        self.graph: Any | None = None  # Compiled graph

    def build_graph(self):
        """Build LangGraph state graph"""
        # Create graph
        graph = StateGraph(GovernanceState)

        # Add nodes (implemented by subclasses)
        self._add_nodes(graph)

        # Add edges (implemented by subclasses)
        self._add_edges(graph)

        # Compile graph
        self.graph = graph.compile()
        return self.graph

    @abstractmethod
    def _add_nodes(self, graph: StateGraph) -> None:
        """Add nodes to the graph"""
        pass

    @abstractmethod
    def _add_edges(self, graph: StateGraph) -> None:
        """Add edges to the graph"""
        pass

    def execute(self, initial_state: GovernanceState | None = None) -> GovernanceState:
        """Execute the workflow"""
        logger.info("Starting governance workflow")

        if self.graph is None:
            self.build_graph()

        if self.graph is None:
            raise RuntimeError("Failed to build graph")

        # Use provided state or create new
        if initial_state is None:
            initial_state = GovernanceState()

        try:
            # Execute graph - LangGraph returns dict
            result_dict = self.graph.invoke(initial_state)

            # Convert dict back to GovernanceState
            if isinstance(result_dict, dict):
                result_state = GovernanceState(**result_dict)
            else:
                result_state = cast(GovernanceState, result_dict)

            # Update state
            self.state = result_state
            self.state.transition_to(WorkflowState.COMPLETED)
            logger.info("Workflow completed successfully")
        except Exception as e:
            self.state = initial_state
            self.state.error_message = str(e)
            self.state.transition_to(WorkflowState.FAILED)
            logger.error(f"Workflow failed: {e}")

        self.history.append(self.state)
        return self.state

    def get_history(self) -> list[GovernanceState]:
        """Get workflow state history"""
        return self.history

    def get_current_state(self) -> GovernanceState:
        """Get current workflow state"""
        return self.state
