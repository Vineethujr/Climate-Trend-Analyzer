"""
Unit tests for visualizations module.
"""

import pandas as pd
import plotly.graph_objects as go
from src.visualizations import (
    plot_historical_series,
    plot_annual_bars,
    plot_trend_regression,
    plot_monthly_climatology,
    plot_seasonal_comparison,
    plot_decomposition,
    plot_anomaly_detection,
    plot_prediction_comparison,
)
from src.trend_analysis import compute_ols_trend
from src.seasonal_analysis import compute_monthly_climatology, decompose_time_series
from src.anomaly_detection import detect_zscore_anomalies
from src.predictive_model import train_and_evaluate_models
from tests.fixtures.synthetic_series import generate_linear_trend_series


def test_plot_historical_series():
    df, _ = generate_linear_trend_series(start_year=1980, end_year=2000)
    df["Rolling_12m"] = df["Anomaly"]
    df["Rolling_60m"] = df["Anomaly"]
    fig = plot_historical_series(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_annual_bars():
    annual_df = pd.DataFrame({
        "Year": [1990, 1991, 1992],
        "Annual_Mean": [-0.1, 0.2, 0.3],
    })
    fig = plot_annual_bars(annual_df)
    assert isinstance(fig, go.Figure)


def test_plot_trend_regression():
    df, _ = generate_linear_trend_series(start_year=1980, end_year=2000)
    trend_dict = compute_ols_trend(df)
    fig = plot_trend_regression(trend_dict)
    assert isinstance(fig, go.Figure)


def test_plot_monthly_climatology():
    df, _ = generate_linear_trend_series(start_year=1980, end_year=2000)
    monthly_df = compute_monthly_climatology(df)
    fig = plot_monthly_climatology(monthly_df)
    assert isinstance(fig, go.Figure)


def test_plot_decomposition():
    df, _ = generate_linear_trend_series(start_year=1980, end_year=2000)
    decomp_df = decompose_time_series(df)
    fig = plot_decomposition(decomp_df)
    assert isinstance(fig, go.Figure)


def test_plot_anomaly_detection():
    df, _ = generate_linear_trend_series(start_year=1980, end_year=2000)
    annotated = detect_zscore_anomalies(df)
    fig = plot_anomaly_detection(annotated, method_name="Z-Score")
    assert isinstance(fig, go.Figure)


def test_plot_prediction_comparison():
    df, _ = generate_linear_trend_series(start_year=1960, end_year=2020)
    eval_res = train_and_evaluate_models(df, split_year=2010)
    fig = plot_prediction_comparison(eval_res)
    assert isinstance(fig, go.Figure)
