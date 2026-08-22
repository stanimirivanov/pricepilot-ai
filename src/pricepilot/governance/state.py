"""State definitions for governance workflow"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class WorkflowState(str, Enum):
    """States in the governance workflow"""

    INGEST_DATA = "ingest_data"
    FORECAST = "forecast"
    OPTIMIZE = "optimize"
    CHECK_CONFIDENCE = "check_confidence"
    APPROVED = "approved"
    REQUEST_REVIEW = "request_review"
    HUMAN_OVERRIDE = "human_override"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class GovernanceState:
    """State object for the governance workflow"""

    # Data
    historical_data: pd.DataFrame | None = None
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
    confidence_details: dict[str, float] = field(default_factory=dict)

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
    timestamp: pd.Timestamp | None = None
    execution_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "current_price": self.current_price,
            "optimal_price": self.optimal_price,
            "final_price": self.final_price,
            "forecasted_demand": self.forecasted_demand,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value if self.confidence_level else None,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_status": self.anomaly_status,
            "approved": self.approved,
            "human_reviewed": self.human_reviewed,
            "human_override_price": self.human_override_price,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
        }

    def transition_to(self, new_state: WorkflowState) -> None:
        """Transition to a new state"""
        self.previous_state = self.current_state
        self.current_state = new_state
