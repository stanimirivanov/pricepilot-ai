"""Base workflow class for governance"""

from abc import ABC, abstractmethod

from loguru import logger

from pricepilot.governance.state import GovernanceState, WorkflowState


class BaseGovernanceWorkflow(ABC):
    """Abstract base class for governance workflows"""

    def __init__(self):
        self.state = GovernanceState()
        self.history: list[GovernanceState] = []

    def execute(self) -> GovernanceState:
        """Execute the workflow"""
        logger.info("Starting governance workflow")

        try:
            self._run()
            self.state.transition_to(WorkflowState.COMPLETED)
            logger.info("Workflow completed successfully")
        except Exception as e:
            self.state.error_message = str(e)
            self.state.transition_to(WorkflowState.FAILED)
            logger.error(f"Workflow failed: {e}")

        self.history.append(self.state)
        return self.state

    @abstractmethod
    def _run(self) -> None:
        """Run the workflow steps (implemented by subclasses)"""
        pass

    def get_history(self) -> list[GovernanceState]:
        """Get workflow state history"""
        return self.history

    def get_current_state(self) -> GovernanceState:
        """Get current workflow state"""
        return self.state
