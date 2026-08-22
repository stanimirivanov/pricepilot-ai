"""FastAPI application for pricing review and override"""

from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from loguru import logger

from pricepilot.api.feedback import FeedbackStore
from pricepilot.api.models import (
    FeedbackRecord,
    HumanOverrideRequest,
    OverrideResponse,
    PricingRecommendation,
)
from pricepilot.governance.confidence import ConfidenceThresholds
from pricepilot.governance.pricing_workflow import PricingGovernanceWorkflow
from pricepilot.governance.state import GovernanceState
from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PricingPipeline

app = FastAPI(
    title="PricePilot AI - Review API",
    description="Human-on-the-loop pricing governance API",
    version="0.1.0",
)

# Global state
pipeline: PricingPipeline | None = None
workflow: PricingGovernanceWorkflow | None = None
feedback_store = FeedbackStore()
current_recommendation: PricingRecommendation | None = None


def get_pipeline() -> PricingPipeline:
    """Get or create pipeline"""
    global pipeline

    if pipeline is None:
        logger.info("Initializing pricing pipeline...")
        pipeline_config = PipelineConfig()
        pipeline = PricingPipeline(config=pipeline_config, enable_mlflow=False)
        pipeline.load_or_generate_data()
        pipeline.fit_models()
        logger.info("Pipeline initialized")

    return pipeline


def get_workflow() -> PricingGovernanceWorkflow:
    """Get or create workflow"""
    global workflow

    if workflow is None:
        logger.info("Initializing governance workflow...")
        pipeline = get_pipeline()
        workflow = PricingGovernanceWorkflow(
            pipeline=pipeline,
            confidence_scorer=ConfidenceThresholds.balanced(),
        )
        logger.info("Workflow initialized")

    return workflow


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PricePilot AI Review API",
        "version": "0.1.0",
        "endpoints": [
            "/health",
            "/recommendation",
            "/override",
            "/feedback",
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline is not None,
        "workflow_ready": workflow is not None,
    }


@app.get("/recommendation", response_model=PricingRecommendation)
async def get_recommendation():
    """Get pricing recommendation for tomorrow"""
    global current_recommendation

    try:
        workflow = get_workflow()

        # Execute workflow
        state = workflow.execute(GovernanceState())

        # Create recommendation
        recommendation_id = f"rec-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        recommendation = PricingRecommendation(
            recommendation_id=recommendation_id,
            current_price=state.current_price or 15.0,
            forecasted_demand=state.forecasted_demand or 0,
            demand_lower=state.demand_interval[0] if state.demand_interval else 0,
            demand_upper=state.demand_interval[1] if state.demand_interval else 0,
            optimal_price=state.optimal_price or 0,
            expected_revenue=state.expected_revenue or 0,
            confidence_score=state.confidence_score or 0,
            confidence_level=state.confidence_level.value if state.confidence_level else "low",
            anomaly_detected=state.anomaly_detected,
            anomaly_status=state.anomaly_status,
            requires_review=not state.approved,
        )

        current_recommendation = recommendation

        logger.info(f"Recommendation generated: {recommendation_id}")
        return recommendation

    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/override", response_model=OverrideResponse)
async def submit_override(request: HumanOverrideRequest):
    """Submit human override for pricing decision"""
    global current_recommendation

    try:
        if current_recommendation is None:
            raise HTTPException(status_code=400, detail="No active recommendation")

        if request.recommendation_id != current_recommendation.recommendation_id:
            raise HTTPException(
                status_code=400,
                detail=f"Recommendation ID mismatch. Expected: {current_recommendation.recommendation_id}",
            )

        # Create override response
        override_id = f"ovr-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        price_difference = request.human_price - current_recommendation.optimal_price

        override = OverrideResponse(
            override_id=override_id,
            recommendation_id=request.recommendation_id,
            original_price=current_recommendation.optimal_price,
            human_price=request.human_price,
            price_difference=price_difference,
            notes=request.notes,
            reviewer_name=request.reviewer_name,
        )

        # Log feedback
        feedback_record = FeedbackRecord(
            timestamp=datetime.now(),
            date=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            original_price=current_recommendation.optimal_price,
            final_price=request.human_price,
            forecasted_demand=current_recommendation.forecasted_demand,
            actual_demand=None,  # Will be updated later
            confidence_score=current_recommendation.confidence_score,
            anomaly_detected=current_recommendation.anomaly_detected,
            human_override=True,
            reviewer_name=request.reviewer_name,
            notes=request.notes,
        )

        feedback_store.log_override(feedback_record.model_dump())

        logger.info(f"Override submitted: {override_id}")
        return override

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit override: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback")
async def get_feedback(limit: int = 10):
    """Get recent feedback records"""
    try:
        records = feedback_store.get_recent_overrides(limit)
        return {
            "count": len(records),
            "records": records,
        }
    except Exception as e:
        logger.error(f"Failed to get feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/actual-demand")
async def log_actual_demand(
    recommendation_id: str,
    actual_demand: int,
):
    """Log actual demand for a previous recommendation"""
    try:
        # Find the feedback record
        records = feedback_store.get_all_feedback()

        for record in records:
            if record.get("recommendation_id") == recommendation_id:
                record["actual_demand"] = actual_demand
                feedback_store.log_override(record)
                return {"status": "updated", "recommendation_id": recommendation_id}

        raise HTTPException(status_code=404, detail="Recommendation not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to log actual demand: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/append-training")
async def append_to_training():
    """Append feedback data to training set"""
    try:
        updated_data = feedback_store.append_to_training_data()
        return {
            "status": "success",
            "records_appended": len(updated_data) if not updated_data.empty else 0,
        }
    except Exception as e:
        logger.error(f"Failed to append training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
