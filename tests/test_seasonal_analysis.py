"""
Unit tests for seasonal_analysis module.
"""

from tests.fixtures.synthetic_series import generate_seasonal_test_series
from src.seasonal_analysis import (
    compute_monthly_climatology,
    compute_seasonal_summary,
    decompose_time_series,
    compare_decadal_seasonality,
)


def test_compute_monthly_climatology():
    df = generate_seasonal_test_series(n_years=10, seasonal_amplitude=1.5)
    clim = compute_monthly_climatology(df)
    assert len(clim) == 12
    assert list(clim["Month"]) == list(range(1, 13))
    assert "Mean" in clim.columns
    assert "Std" in clim.columns


def test_decompose_time_series():
    df = generate_seasonal_test_series(n_years=15, seasonal_amplitude=1.0)
    decomp = decompose_time_series(df, period=12)
    assert len(decomp) == len(df)
    assert list(decomp.columns) == [
        "Date", "Year", "Month", "Decimal_Year", "Observed", "Trend", "Seasonal", "Residual"
    ]
    # Check that seasonal component has non-zero variation
    assert decomp["Seasonal"].std() > 0.3
    # Check that seasonal components roughly average to 0
    assert abs(decomp["Seasonal"].iloc[:12].sum()) < 0.05


def test_seasonal_summary():
    df = generate_seasonal_test_series(n_years=12, seasonal_amplitude=1.0)
    overall, trends = compute_seasonal_summary(df)
    assert len(overall) == 4
    assert set(overall["Season"]) == {"DJF", "MAM", "JJA", "SON"}
