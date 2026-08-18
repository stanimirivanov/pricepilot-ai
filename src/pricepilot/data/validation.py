from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    """Data validation result"""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, float]


class DataValidator:
    """Validate synthetic data quality"""

    def __init__(self):
        self.required_columns = [
            "date",
            "price",
            "quantity_sold",
            "revenue",
            "is_raining",
            "is_sunny",
            "temperature",
            "day_of_week",
            "is_weekend",
            "month",
            "year",
        ]

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Run all validation checks"""
        errors = []
        warnings = []

        # Check columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # Check data types
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            errors.append("'date' column must be datetime type")

        # Check value ranges
        if "price" in df.columns:
            if df["price"].min() <= 0:
                errors.append("Prices must be positive")
            if df["price"].max() > 100:
                warnings.append("Some prices seem unusually high (>$100)")

        if "quantity_sold" in df.columns:
            if df["quantity_sold"].min() < 0:
                errors.append("Quantity sold cannot be negative")
            if df["quantity_sold"].max() > 200:
                warnings.append("Some demand values seem unusually high (>200 cars/day)")

        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        if missing_pct.any() > 0:
            warnings.append(f"Missing values detected: {missing_pct[missing_pct > 0].to_dict()}")

        # Calculate quality metrics
        metrics = {
            "n_records": len(df),
            "missing_values_pct": df.isnull().sum().sum() / len(df) * 100,
            "price_variation": df["price"].std() if "price" in df.columns else 0,
            "demand_variation": df["quantity_sold"].std() if "quantity_sold" in df.columns else 0,
            "rainy_day_pct": df["is_raining"].mean() * 100 if "is_raining" in df.columns else 0,
        }

        is_valid = len(errors) == 0

        if not is_valid:
            logger.error(f"Data validation failed with {len(errors)} errors")
        elif warnings:
            logger.warning(f"Data validation passed with {len(warnings)} warnings")
        else:
            logger.info("Data validation passed successfully")

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, metrics=metrics
        )
