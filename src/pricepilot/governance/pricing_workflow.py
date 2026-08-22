"""Full governance workflow integrating pricing pipeline with LangGraph"""

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from pricepilot.governance.confidence import ConfidenceScorer
from pricepilot.governance.state import ConfidenceLevel, GovernanceState, WorkflowState
from pricepilot.governance.workflow import BaseGovernanceWorkflow
from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PipelineResult, PricingPipeline


class PricingGovernanceWorkflow(BaseGovernanceWorkflow):
    """Complete governance workflow for pricing decisions"""

    def __init__(
        self,
        pipeline: PricingPipeline | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
    ):
        """
        Initialize pricing governance workflow

        Args:
            pipeline: Fitted pricing pipeline (if None, will create)
            confidence_scorer: Confidence scorer for decision evaluation
        """
        super().__init__()
        self.pipeline = pipeline
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.pipeline_result: PipelineResult | None = None

    def _add_nodes(self, graph: StateGraph) -> None:
        """Add nodes to the graph"""
        graph.add_node("ingest_data", self._ingest_data)
        graph.add_node("forecast", self._forecast)
        graph.add_node("optimize", self._optimize)
        graph.add_node("check_confidence", self._check_confidence)
        graph.add_node("approve", self._approve)
        graph.add_node("request_review", self._request_review)
        graph.add_node("human_override", self._human_override)
        graph.add_node("finalize", self._finalize)

    def _add_edges(self, graph: StateGraph) -> None:
        """Add edges to the graph"""
        graph.add_edge(START, "ingest_data")
        graph.add_edge("ingest_data", "forecast")
        graph.add_edge("forecast", "optimize")
        graph.add_edge("optimize", "check_confidence")

        # Conditional routing based on confidence
        graph.add_conditional_edges(
            "check_confidence",
            self._route_by_confidence,
            {
                "approve": "approve",
                "review": "request_review",
            },
        )

        graph.add_edge("approve", "finalize")
        graph.add_edge("request_review", "human_override")
        graph.add_edge("human_override", "finalize")
        graph.add_edge("finalize", END)

    def _ingest_data(self, state: GovernanceState) -> GovernanceState:
        """Ingest data step"""
        logger.info("Step 1: Ingesting data")
        state.transition_to(WorkflowState.INGEST_DATA)
        state.timestamp = __import__("pandas").Timestamp.now()

        # Ensure pipeline exists
        if self.pipeline is None:
            logger.info("Creating new pipeline...")
            self.pipeline = PricingPipeline(
                config=PipelineConfig(),
                enable_mlflow=False,  # Disable MLflow for workflow execution
            )
            self.pipeline.load_or_generate_data()
            self.pipeline.fit_models()

        state.historical_data = self.pipeline.data
        return state

    def _forecast(self, state: GovernanceState) -> GovernanceState:
        """Forecast demand step"""
        logger.info("Step 2: Forecasting demand")
        state.transition_to(WorkflowState.FORECAST)

        if self.pipeline is None or self.pipeline.forecaster is None:
            raise ValueError("Pipeline forecaster not available")

        # Generate forecast
        forecast = self.pipeline.forecaster.predict(steps=1)

        state.forecast_result = forecast
        state.forecasted_demand = float(forecast.mean[0])
        state.demand_interval = (
            float(forecast.lower[0]),
            float(forecast.upper[0]),
        )

        logger.info(f"Forecasted demand: {state.forecasted_demand:.0f}")
        return state

    def _optimize(self, state: GovernanceState) -> GovernanceState:
        """Optimize price step"""
        logger.info("Step 3: Optimizing price")
        state.transition_to(WorkflowState.OPTIMIZE)

        if self.pipeline is None or self.pipeline.pricing_model is None:
            raise ValueError("Pipeline pricing model not available")

        # Get current price
        if state.current_price is None:
            state.current_price = float(self.pipeline.data["price"].iloc[-1])

        # Generate pricing result
        pricing_result = self.pipeline.pricing_model.price_for_tomorrow(
            current_price=state.current_price,
            forecast=state.forecast_result,
        )

        state.optimal_price = pricing_result.optimal_price
        state.expected_revenue = pricing_result.expected_revenue
        state.forecasted_demand = pricing_result.forecasted_demand

        self.pipeline_result = pricing_result

        logger.info(f"Optimal price: ${state.optimal_price:.2f}")
        return state

    def _check_confidence(self, state: GovernanceState) -> GovernanceState:
        """Check confidence step"""
        logger.info("Step 4: Checking confidence")
        state.transition_to(WorkflowState.CHECK_CONFIDENCE)

        if self.pipeline is None or self.pipeline.elasticity_model is None:
            raise ValueError("Pipeline elasticity model not available")

        # Get elasticity statistics
        elasticity_mean = self.pipeline.elasticity_model.results.posterior_mean
        elasticity_std = self.pipeline.elasticity_model.results.posterior_std

        # Get forecast interval width
        if state.demand_interval:
            interval_width = state.demand_interval[1] - state.demand_interval[0]
        else:
            interval_width = 50  # Default if not available

        # Check anomaly
        anomaly_detected = state.anomaly_detected

        # Calculate confidence
        confidence_score = self.confidence_scorer.calculate(
            forecast_interval_width=interval_width,
            forecasted_demand=state.forecasted_demand or 100,
            elasticity_std=elasticity_std,
            elasticity_mean=elasticity_mean,
            anomaly_detected=anomaly_detected,
        )

        state.confidence_score = confidence_score.score
        state.confidence_level = confidence_score.level
        state.confidence_details = confidence_score.details

        logger.info(f"Confidence: {confidence_score.score:.3f} ({confidence_score.level.value})")
        return state

    def _route_by_confidence(self, state: GovernanceState) -> str:
        """Route based on confidence level"""
        if state.confidence_level == ConfidenceLevel.HIGH:
            return "approve"
        return "review"

    def _approve(self, state: GovernanceState) -> GovernanceState:
        """Approve decision step"""
        logger.info("Step 5: Approving decision")
        state.transition_to(WorkflowState.APPROVED)
        state.approved = True
        state.final_price = state.optimal_price
        logger.info(f"Approved price: ${state.final_price:.2f}")
        return state

    def _request_review(self, state: GovernanceState) -> GovernanceState:
        """Request review step"""
        logger.info("Step 5: Requesting human review")
        state.transition_to(WorkflowState.REQUEST_REVIEW)
        state.approved = False
        state.human_reviewed = False
        logger.warning(
            f"Human review required. Confidence: {state.confidence_score:.3f} "
            f"(suggested price: ${state.optimal_price:.2f})"
        )
        return state

    def _human_override(self, state: GovernanceState) -> GovernanceState:
        """Human override step"""
        logger.info("Step 6: Processing human override")
        state.transition_to(WorkflowState.HUMAN_OVERRIDE)
        state.human_reviewed = True

        if state.human_override_price is not None:
            state.final_price = state.human_override_price
            logger.info(f"Human override price: ${state.final_price:.2f}")
        else:
            # Human approved the AI suggestion
            state.final_price = state.optimal_price
            state.approved = True
            logger.info(f"Human approved AI price: ${state.final_price:.2f}")

        return state

    def _finalize(self, state: GovernanceState) -> GovernanceState:
        """Finalize step"""
        logger.info("Step 7: Finalizing decision")
        state.transition_to(WorkflowState.COMPLETED)
        state.execution_time = (
            time.time() - state.timestamp.timestamp() if state.timestamp else None
        )

        logger.info("=" * 60)
        logger.info("GOVERNANCE WORKFLOW COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Final Price: ${state.final_price:.2f}")
        logger.info(f"Approved: {state.approved}")
        logger.info(f"Human Reviewed: {state.human_reviewed}")
        logger.info(
            f"Confidence: {state.confidence_score:.3f}"
            if state.confidence_score
            else "No confidence score"
        )
        logger.info("=" * 60)

        return state


class WorkflowResult:
    """Container for workflow execution results"""

    def __init__(self, state: GovernanceState):
        self.state = state
        self.final_price = state.final_price
        self.approved = state.approved
        self.human_reviewed = state.human_reviewed
        self.confidence_score = state.confidence_score
        self.confidence_level = state.confidence_level
        self.forecasted_demand = state.forecasted_demand
        self.optimal_price = state.optimal_price
        self.execution_time = state.execution_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "final_price": self.final_price,
            "approved": self.approved,
            "human_reviewed": self.human_reviewed,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value if self.confidence_level else None,
            "forecasted_demand": self.forecasted_demand,
            "optimal_price": self.optimal_price,
            "execution_time": self.execution_time,
        }
