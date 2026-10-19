"""
Unit tests for predictive_model module.
"""

import pytest
import pandas as pd
from tests.fixtures.synthetic_series import generate_linear_trend_series
from src.predictive_model import (
    chronological_split,
    train_and_evaluate_models,
    generate_future_projection,
)


def test_chronological_split_ordering():
    df, _ = generate_linear_trend_series(start_year=1980, end_year=2020)
    train_df, test_df = chronological_split(df, split_year=2010)

    # Verify no overlap and strict chronological ordering
    assert train_df["Year"].max() < 2010
    assert test_df["Year"].min() >= 2010
    assert len(train_df) + len(test_df) == len(df)


def test_train_and_evaluate_models():
    df, _ = generate_linear_trend_series(start_year=1950, end_year=2020, slope_per_decade=0.25)
    results = train_and_evaluate_models(df, split_year=2010)

    assert "comparison_table" in results
    comp_df = results["comparison_table"]
    assert len(comp_df) == 3
    assert set(comp_df.columns) == {"Model", "MAE", "RMSE", "R2", "Max_Error"}
    
    # Regression model should perform better or equal to persistent mean baseline
    baseline_rmse = comp_df.loc[comp_df["Model"] == "Persistent Mean Baseline", "RMSE"].iloc[0]
    linear_rmse = comp_df.loc[comp_df["Model"] == "Linear Trend Extrapolation", "RMSE"].iloc[0]
    assert linear_rmse <= baseline_rmse


def test_generate_future_projection():
    df, _ = generate_linear_trend_series(start_year=1970, end_year=2020)
    proj = generate_future_projection(df, horizon_months=24)

    assert len(proj) == 24
    assert list(proj.columns) == [
        "Date", "Year", "Month", "Decimal_Year", "Projected_Anomaly", "Lower_95_CI", "Upper_95_CI"
    ]
    # Projected years should start in 2021
    assert proj["Year"].iloc[0] == 2021
    # Upper CI should be strictly greater than Lower CI
    assert (proj["Upper_95_CI"] > proj["Lower_95_CI"]).all()
