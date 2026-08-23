"""State definitions for governance workflow"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowState(str, Enum):
    """States in the governance workflow"""

    INGEST_DATA = "ingest_data"
    FORECAST = "forecast"
    OPTIMIZE = "optimize"
    CHECK_CONFIDENCE = "check_confidence"
    APPROVED = "approved"
    REQUEST_REVIEW = "request_review"
    HUMAN_OVERRIDE = "human_override"
    PENDING_REVIEW = "pending_review"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GovernanceState(BaseModel):
    """State for governance workflow (LangGraph compatible with Pydantic)"""

    # Data
    historical_data: Any | None = None  # DataFrame not directly supported
    forecast_result: Any | None = None

    # Pricing
    current_price: float | None = None
    optimal_price: float | None = None
    expected_revenue: float | None = None
    forecasted_demand: float | None = None
    demand_interval: tuple | None = None

    # Confidence
    confidence_score: float | None = None
    confidence_level: ConfidenceLevel | None = None
    confidence_details: dict[str, float] = Field(default_factory=dict)

    # Anomaly
    anomaly_detected: bool = False
    anomaly_status: str = "NORMAL"

    # Workflow
    current_state: WorkflowState = WorkflowState.INGEST_DATA
    previous_state: WorkflowState | None = None
    error_message: str | None = None

    # Decision
    approved: bool = False
    human_reviewed: bool = False
    human_override_price: float | None = None
    human_notes: str | None = None
    final_price: float | None = None

    # Metadata
    timestamp: str | None = None  # Use ISO format string for serialization
    execution_time: float | None = None

    def transition_to(self, new_state: WorkflowState) -> None:
        """Transition to a new state"""
        self.previous_state = self.current_state
        self.current_state = new_state

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string"""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GovernanceState":
        """Create from dictionary"""
        return cls(**data)
