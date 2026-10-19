"""
Unit tests for trend_analysis module using synthetic known-truth series.
"""

import pytest
import numpy as np
from tests.fixtures.synthetic_series import generate_linear_trend_series
from src.trend_analysis import (
    compute_ols_trend,
    compute_theil_sen_trend,
    compute_mann_kendall_test,
    compute_autocorrelation_diagnostics,
    compare_era_trends,
)


def test_recover_known_trend_slope():
    # Known trend: +0.25 °C per decade
    known_rate = 0.25
    df, meta = generate_linear_trend_series(
        start_year=1900,
        end_year=2000,
        slope_per_decade=known_rate,
        noise_std=0.04,
        seed=42,
    )
    res = compute_ols_trend(df)
    
    # Must recover the true slope within 5% relative tolerance
    recovered_rate = res["rate_per_decade"]
    assert abs(recovered_rate - known_rate) < 0.02
    assert res["is_statistically_significant"] is True
    assert res["p_value"] < 0.0001
    assert res["ci_decade_lower"] <= known_rate <= res["ci_decade_upper"]


def test_theil_sen_robust_recovery():
    known_rate = 0.20
    df, meta = generate_linear_trend_series(
        start_year=1920,
        end_year=1980,
        slope_per_decade=known_rate,
        noise_std=0.03,
        seed=99,
    )
    res = compute_theil_sen_trend(df)
    recovered_rate = res["theil_sen_rate_per_decade"]
    assert abs(recovered_rate - known_rate) < 0.025


def test_mann_kendall_monotonic_trend():
    df, meta = generate_linear_trend_series(
        start_year=1950,
        end_year=2000,
        slope_per_decade=0.30,
        seed=101,
    )
    mk_res = compute_mann_kendall_test(df)
    assert mk_res["is_significant"] is True
    assert mk_res["kendall_tau"] > 0.5
    assert "Upward Trend" in mk_res["trend_classification"]


def test_autocorrelation_diagnostics():
    # Generate series with known positive serial correlation
    np.random.seed(42)
    n = 200
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + np.random.normal(0, 0.2)
    
    import pandas as pd
    df = pd.DataFrame({"Anomaly": y})
    diag = compute_autocorrelation_diagnostics(df)
    
    assert diag["lag_1_autocorrelation"] > 0.4
    assert diag["durbin_watson_stat"] < 1.8  # Positive autocorrelation gives DW < 2
    assert diag["effective_sample_size"] < n
