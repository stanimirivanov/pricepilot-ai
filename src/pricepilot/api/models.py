"""Pydantic models for API requests and responses"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PricingRecommendation(BaseModel):
    """Pricing recommendation response"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendation_id": "rec-20240101-123456",
                "timestamp": "2024-01-01T12:34:56",
                "current_price": 15.0,
                "forecasted_demand": 104,
                "demand_lower": 85,
                "demand_upper": 122,
                "optimal_price": 16.45,
                "expected_revenue": 1710.80,
                "confidence_score": 0.85,
                "confidence_level": "medium",
                "anomaly_detected": False,
                "anomaly_status": "NORMAL",
                "requires_review": True,
            }
        }
    )

    recommendation_id: str = Field(..., description="Unique recommendation ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    current_price: float = Field(..., description="Current price")
    forecasted_demand: float = Field(..., description="Forecasted demand")
    demand_lower: float = Field(..., description="Lower bound of demand interval")
    demand_upper: float = Field(..., description="Upper bound of demand interval")
    optimal_price: float = Field(..., description="AI recommended price")
    expected_revenue: float = Field(..., description="Expected revenue at optimal price")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    confidence_level: str = Field(..., description="Confidence level (high/medium/low)")
    anomaly_detected: bool = Field(default=False, description="Whether anomaly detected")
    anomaly_status: str = Field(default="NORMAL", description="Anomaly status")
    requires_review: bool = Field(..., description="Whether human review is required")


class HumanOverrideRequest(BaseModel):
    """Human override request"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendation_id": "rec-20240101-123456",
                "human_price": 18.50,
                "notes": "Competitor price drop detected",
                "reviewer_name": "John Smith",
            }
        }
    )

    recommendation_id: str = Field(..., description="Recommendation ID being overridden")
    human_price: float = Field(..., description="Human-specified price", gt=0)
    notes: str | None = Field(None, description="Optional notes about override")
    reviewer_name: str | None = Field(None, description="Name of human reviewer")


class OverrideResponse(BaseModel):
    """Response after human override"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "override_id": "ovr-20240101-123456",
                "recommendation_id": "rec-20240101-123456",
                "original_price": 16.45,
                "human_price": 18.50,
                "price_difference": 2.05,
                "timestamp": "2024-01-01T12:34:56",
                "notes": "Competitor price drop detected",
                "reviewer_name": "John Smith",
                "status": "OVERRIDDEN",
            }
        }
    )

    override_id: str = Field(..., description="Unique override ID")
    recommendation_id: str = Field(..., description="Original recommendation ID")
    original_price: float = Field(..., description="AI recommended price")
    human_price: float = Field(..., description="Human override price")
    price_difference: float = Field(..., description="Difference between human and AI price")
    timestamp: datetime = Field(default_factory=datetime.now)
    notes: str | None = None
    reviewer_name: str | None = None
    status: str = Field(default="OVERRIDDEN", description="Override status")


class FeedbackRecord(BaseModel):
    """Feedback record for continuous learning"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-01T12:34:56",
                "date": "2024-01-02",
                "original_price": 16.45,
                "final_price": 18.50,
                "forecasted_demand": 104,
                "actual_demand": 98,
                "confidence_score": 0.85,
                "anomaly_detected": False,
                "human_override": True,
                "reviewer_name": "John Smith",
                "notes": "Competitor price drop detected",
            }
        }
    )

    timestamp: datetime = Field(default_factory=datetime.now)
    date: str = Field(..., description="Date of pricing decision")
    original_price: float = Field(..., description="AI recommended price")
    final_price: float = Field(..., description="Final executed price")
    forecasted_demand: float = Field(..., description="Forecasted demand")
    actual_demand: float | None = Field(None, description="Actual demand (if known)")
    confidence_score: float = Field(..., description="Confidence score")
    anomaly_detected: bool = Field(default=False)
    human_override: bool = Field(default=False)
    reviewer_name: str | None = None
    notes: str | None = None
