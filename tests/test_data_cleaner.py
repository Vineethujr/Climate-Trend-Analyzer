"""
Unit tests for data_cleaner module.
"""

import pandas as pd
import numpy as np
from src.data_cleaner import (
    validate_schema,
    check_missing_values,
    check_physical_bounds,
    check_temporal_continuity,
    clean_dataset,
)


def test_validate_schema_valid():
    df = pd.DataFrame({
        "Date": ["190001", "190002"],
        "Year": [1900, 1900],
        "Month": [1, 2],
        "Decimal_Year": [1900.04, 1900.12],
        "Anomaly": [-0.15, -0.10],
    })
    is_valid, issues = validate_schema(df)
    assert is_valid is True
    assert len(issues) == 0


def test_validate_schema_missing_column():
    df = pd.DataFrame({
        "Year": [1900],
        "Month": [1],
    })
    is_valid, issues = validate_schema(df)
    assert is_valid is False
    assert any("Missing required columns" in s for s in issues)


def test_check_physical_bounds():
    df = pd.DataFrame({
        "Date": ["200001", "200002", "200003"],
        "Year": [2000, 2000, 2000],
        "Month": [1, 2, 3],
        "Anomaly": [0.5, 99.9, -12.0],  # 99.9 and -12.0 are out of bounds
    })
    res = check_physical_bounds(df)
    assert res["out_of_bounds_count"] == 2


def test_temporal_continuity():
    # Sequence with missing month 2
    df = pd.DataFrame({
        "Year": [2000, 2000],
        "Month": [1, 3],
    })
    diag = check_temporal_continuity(df)
    assert diag["missing_months_count"] == 1
    assert diag["is_continuous"] is False


def test_clean_dataset():
    df = pd.DataFrame({
        "Date": ["190001", "190001", "190002", "190003"],
        "Year": [1900, 1900, 1900, 1900],
        "Month": [1, 1, 2, 3],
        "Decimal_Year": [1900.04, 1900.04, 1900.12, 1900.21],
        "Anomaly": [0.1, 0.1, np.nan, 0.3],  # Duplicate and NaN
    })
    cleaned, log = clean_dataset(df)
    assert log["duplicates_removed"] == 1
    assert log["nulls_removed"] == 1
    assert log["final_observations"] == 2
    assert len(cleaned) == 2
