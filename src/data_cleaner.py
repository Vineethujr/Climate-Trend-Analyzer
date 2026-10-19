"""
Climate Trend Analyzer - Data Quality, Validation & Cleaning Module.

Responsible for:
- Enforcing structural schema requirements.
- Identifying missing, invalid, or duplicate temporal records.
- Verifying physical boundary limits for temperature anomalies.
- Transparently logging cleaning transformations without arbitrary zero-filling.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ["Date", "Year", "Month", "Decimal_Year", "Anomaly"]
PHYSICAL_MIN_ANOMALY = -5.0  # °C
PHYSICAL_MAX_ANOMALY = +5.0  # °C


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the DataFrame satisfies schema expectations.

    Args:
        df: Input climate DataFrame.

    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
    """
    issues = []

    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Input is not a valid pandas DataFrame."]

    if df.empty:
        return False, ["DataFrame is empty (0 observations)."]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    if "Year" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Year"]):
            issues.append("'Year' column is not numeric.")
        elif (df["Year"] < 1700).any() or (df["Year"] > 2100).any():
            issues.append("Detected year values outside realistic historical range (1700-2100).")

    if "Month" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Month"]):
            issues.append("'Month' column is not numeric.")
        elif not df["Month"].dropna().between(1, 12).all():
            issues.append("Detected 'Month' values outside valid range [1, 12].")

    if "Anomaly" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Anomaly"]):
            issues.append("'Anomaly' column is not numeric.")

    is_valid = len(issues) == 0
    return is_valid, issues


def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect the dataset for missing or null values.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary summarizing missing value statistics.
    """
    total_rows = len(df)
    missing_by_col = df.isnull().sum().to_dict()
    missing_rows = int(df.isnull().any(axis=1).sum())

    missing_dates = []
    if "Date" in df.columns and "Anomaly" in df.columns:
        null_mask = df["Anomaly"].isnull()
        missing_dates = df.loc[null_mask, "Date"].tolist()

    return {
        "total_observations": total_rows,
        "missing_observations_count": missing_rows,
        "missing_percentage": round((missing_rows / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "missing_by_column": missing_by_col,
        "missing_dates": missing_dates,
    }


def check_physical_bounds(
    df: pd.DataFrame,
    min_val: float = PHYSICAL_MIN_ANOMALY,
    max_val: float = PHYSICAL_MAX_ANOMALY,
) -> Dict[str, Any]:
    """
    Verify that anomalies fall within plausible physical boundaries.

    Args:
        df: Input DataFrame.
        min_val: Minimum physically plausible anomaly (°C).
        max_val: Maximum physically plausible anomaly (°C).

    Returns:
        Dictionary reporting out-of-bound observations.
    """
    if "Anomaly" not in df.columns or df.empty:
        return {"out_of_bounds_count": 0, "out_of_bounds_rows": []}

    valid_mask = df["Anomaly"].notnull()
    outliers_mask = valid_mask & ((df["Anomaly"] < min_val) | (df["Anomaly"] > max_val))
    outlier_rows = df[outliers_mask]

    return {
        "out_of_bounds_count": len(outlier_rows),
        "min_allowed": min_val,
        "max_allowed": max_val,
        "out_of_bounds_rows": outlier_rows[["Date", "Year", "Month", "Anomaly"]].to_dict(orient="records"),
    }


def check_temporal_continuity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for duplicate months and identify temporal gaps.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with continuity diagnostics.
    """
    if df.empty or "Year" not in df.columns or "Month" not in df.columns:
        return {"is_continuous": True, "duplicate_months": 0, "missing_months_count": 0}

    # Duplicate check
    duplicates = df.duplicated(subset=["Year", "Month"], keep=False).sum()

    # Gap check
    sorted_df = df.sort_values(by=["Year", "Month"]).reset_index(drop=True)
    start_year = int(sorted_df["Year"].iloc[0])
    start_month = int(sorted_df["Month"].iloc[0])
    end_year = int(sorted_df["Year"].iloc[-1])
    end_month = int(sorted_df["Month"].iloc[-1])

    # Construct expected continuous monthly sequence
    expected_periods = (end_year - start_year) * 12 + (end_month - start_month + 1)
    actual_periods = len(sorted_df)
    missing_periods_count = max(0, expected_periods - actual_periods)

    return {
        "start_period": f"{start_year}-{start_month:02d}",
        "end_period": f"{end_year}-{end_month:02d}",
        "expected_total_months": expected_periods,
        "actual_total_months": actual_periods,
        "missing_months_count": missing_periods_count,
        "duplicate_months_count": int(duplicates),
        "is_continuous": missing_periods_count == 0 and duplicates == 0,
    }


def clean_dataset(
    df: pd.DataFrame,
    drop_nulls: bool = True,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform standardized data cleaning and record transformation metrics.

    Args:
        df: Input DataFrame.
        drop_nulls: Whether to drop rows where Anomaly is NaN (never fills with 0).
        drop_duplicates: Whether to keep only the first occurrence of duplicate year-month pairs.

    Returns:
        Tuple of (cleaned DataFrame, summary log).
    """
    initial_count = len(df)
    cleaned = df.copy()

    log: Dict[str, Any] = {
        "initial_observations": initial_count,
        "duplicates_removed": 0,
        "nulls_removed": 0,
        "final_observations": 0,
    }

    if drop_duplicates and "Year" in cleaned.columns and "Month" in cleaned.columns:
        before_dup = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["Year", "Month"], keep="first")
        log["duplicates_removed"] = before_dup - len(cleaned)

    if drop_nulls and "Anomaly" in cleaned.columns:
        before_null = len(cleaned)
        cleaned = cleaned.dropna(subset=["Anomaly"])
        log["nulls_removed"] = before_null - len(cleaned)

    # Sort chronologically and reset index
    cleaned = cleaned.sort_values(by=["Year", "Month"]).reset_index(drop=True)
    log["final_observations"] = len(cleaned)

    return cleaned, log
