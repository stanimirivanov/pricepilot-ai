"""Feedback storage for continuous learning"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class FeedbackStore:
    """Stores human overrides for continuous learning"""

    def __init__(self, storage_path: str = "data/feedback/overrides.jsonl"):
        """
        Initialize feedback store

        Args:
            storage_path: Path to feedback storage file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Feedback store initialized at {self.storage_path}")

    def log_override(self, record: dict[str, Any]) -> None:
        """
        Log an override record

        Args:
            record: Override record as dictionary
        """
        # Append to JSONL file
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        logger.info(f"Override logged: {record.get('override_id', 'unknown')}")

    def get_all_feedback(self) -> list[dict[str, Any]]:
        """
        Get all feedback records

        Returns:
            List of feedback records
        """
        if not self.storage_path.exists():
            return []

        records = []
        with open(self.storage_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        return records

    def get_feedback_dataframe(self) -> pd.DataFrame:
        """
        Get feedback as DataFrame

        Returns:
            DataFrame of feedback records
        """
        records = self.get_all_feedback()
        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def get_recent_overrides(self, n: int = 10) -> list[dict[str, Any]]:
        """
        Get recent overrides

        Args:
            n: Number of recent overrides to return

        Returns:
            List of recent override records
        """
        records = self.get_all_feedback()
        return records[-n:] if records else []

    def append_to_training_data(
        self,
        training_data_path: str = "data/raw/car_wash_transactions.csv",
    ) -> pd.DataFrame:
        """
        Append feedback records to training data for continuous learning

        Args:
            training_data_path: Path to existing training data

        Returns:
            Updated DataFrame with feedback appended
        """
        training_path = Path(training_data_path)

        if not training_path.exists():
            logger.warning(f"Training data not found at {training_data_path}")
            return pd.DataFrame()

        # Load existing data
        training_data = pd.read_csv(training_path, parse_dates=["date"])

        # Get feedback records
        feedback_df = self.get_feedback_dataframe()

        if feedback_df.empty:
            logger.info("No feedback to append")
            return training_data

        # Filter records with actual_demand
        feedback_with_demand = feedback_df[feedback_df["actual_demand"].notna()]

        if feedback_with_demand.empty:
            logger.info("No feedback with actual demand")
            return training_data

        # Create new rows for training data
        new_rows = []
        for _, record in feedback_with_demand.iterrows():
            new_row = {
                "date": pd.to_datetime(record["date"]),
                "price": float(record["final_price"]),
                "quantity_sold": int(record["actual_demand"]),
                "revenue": float(record["final_price"]) * int(record["actual_demand"]),
                "is_raining": False,  # Placeholder
                "is_sunny": True,  # Placeholder
                "temperature": 70.0,  # Placeholder
                "day_of_week": pd.to_datetime(record["date"]).dayofweek,
                "is_weekend": pd.to_datetime(record["date"]).dayofweek >= 5,
                "month": pd.to_datetime(record["date"]).month,
                "year": pd.to_datetime(record["date"]).year,
            }
            new_rows.append(new_row)

        if new_rows:
            new_data = pd.DataFrame(new_rows)
            updated_data = pd.concat([training_data, new_data], ignore_index=True)

            # Save updated data
            updated_data.to_csv(training_path, index=False)
            logger.info(f"Appended {len(new_rows)} feedback records to training data")
            return updated_data

        return training_data
