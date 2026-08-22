"""Run governance workflow"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

import click
from loguru import logger

from pricepilot.governance.confidence import ConfidenceThresholds
from pricepilot.governance.pricing_workflow import PricingGovernanceWorkflow, WorkflowResult
from pricepilot.governance.state import GovernanceState
from pricepilot.pipeline.config import PipelineConfig
from pricepilot.pipeline.pricing_pipeline import PricingPipeline


@click.command()
@click.option("--data-path", default="data/raw/car_wash_transactions.csv")
@click.option("--current-price", default=None, type=float)
@click.option(
    "--threshold", default="balanced", type=click.Choice(["conservative", "balanced", "aggressive"])
)
@click.option("--human-override", default=None, type=float, help="Human override price")
@click.option("--output", default=None, help="Output JSON file")
def run_governance(
    data_path: str,
    current_price: float | None,
    threshold: str,
    human_override: float | None,
    output: str | None,
):
    """Run governance workflow for pricing decision"""

    # Create pipeline
    pipeline_config = PipelineConfig(data_path=data_path)
    pipeline = PricingPipeline(config=pipeline_config, enable_mlflow=False)
    pipeline.load_or_generate_data()
    pipeline.fit_models()

    # Create confidence scorer
    if threshold == "conservative":
        confidence_scorer = ConfidenceThresholds.conservative()
    elif threshold == "aggressive":
        confidence_scorer = ConfidenceThresholds.aggressive()
    else:
        confidence_scorer = ConfidenceThresholds.balanced()

    # Create workflow
    workflow = PricingGovernanceWorkflow(
        pipeline=pipeline,
        confidence_scorer=confidence_scorer,
    )

    # Set initial state
    initial_state = GovernanceState()
    if current_price:
        initial_state.current_price = current_price
    if human_override:
        initial_state.human_override_price = human_override

    # Execute workflow
    state = workflow.execute(initial_state)

    # Create workflow result (serializable)
    result = WorkflowResult(state)
    result_dict = result.to_json_serializable()

    # Print result
    logger.info("\n" + "=" * 60)
    logger.info("GOVERNANCE DECISION")
    logger.info("=" * 60)
    if result.final_price:
        logger.info(f"Final Price:     ${result.final_price:.2f}")
        logger.info(f"Approved:        {result.approved}")
        logger.info(f"Human Reviewed:  {result.human_reviewed}")
    if result.confidence_score:
        logger.info(f"Confidence:      {result.confidence_score:.3f}")
    if result.forecasted_demand:
        logger.info(f"Forecast Demand: {result.forecasted_demand:.0f}")
        logger.info("=" * 60)

    # Save result
    if output:
        with open(output, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.info(f"Result saved to {output}")

    return result_dict


if __name__ == "__main__":
    run_governance()
