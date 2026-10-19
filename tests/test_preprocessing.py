"""
Unit tests for preprocessing module.
"""

import pandas as pd
import numpy as np
from src.preprocessing import (
    compute_rolling_averages,
    adjust_baseline,
    aggregate_annual,
    aggregate_decadal,
    add_meteorological_season,
    filter_by_date_range,
)


def test_compute_rolling_averages():
    vals = [1.0] * 12 + [2.0] * 12
    df = pd.DataFrame({"Anomaly": vals})
    res = compute_rolling_averages(df, windows=[12])
    assert "Rolling_12m" in res.columns
    # 12th element should be exactly 1.0
    assert abs(res["Rolling_12m"].iloc[11] - 1.0) < 1e-4
    # 24th element should be exactly 2.0
    assert abs(res["Rolling_12m"].iloc[23] - 2.0) < 1e-4


def test_adjust_baseline():
    df = pd.DataFrame({
        "Year": [1950, 1951, 1952, 1953],
        "Anomaly": [0.20, 0.30, 0.40, 0.50],
    })
    # Baseline 1950-1951 mean is (0.2 + 0.3) / 2 = 0.25
    re_centered, offset = adjust_baseline(df, base_start_year=1950, base_end_year=1951)
    assert abs(offset - 0.25) < 1e-4
    assert abs(re_centered["Anomaly"].iloc[0] - (-0.05)) < 1e-4
    assert abs(re_centered["Anomaly"].iloc[1] - (0.05)) < 1e-4


def test_aggregate_annual():
    records = []
    # 1990 has 12 months, 1991 has only 6 months
    for m in range(1, 13):
        records.append({"Year": 1990, "Month": m, "Anomaly": 0.5})
    for m in range(1, 7):
        records.append({"Year": 1991, "Month": m, "Anomaly": 0.8})
    
    df = pd.DataFrame(records)
    annual = aggregate_annual(df, min_months=12)
    assert len(annual) == 1
    assert annual["Year"].iloc[0] == 1990
    assert annual["Annual_Mean"].iloc[0] == 0.5


def test_add_meteorological_season():
    df = pd.DataFrame({
        "Year": [2020, 2020, 2020, 2020],
        "Month": [1, 4, 7, 12],
    })
    res = add_meteorological_season(df)
    assert list(res["Season"]) == ["DJF", "MAM", "JJA", "DJF"]
    # December 2020 belongs to Season_Year 2021 for continuous DJF
    assert res.loc[res["Month"] == 12, "Season_Year"].iloc[0] == 2021


def test_filter_by_date_range():
    df = pd.DataFrame({
        "Year": [1980, 1990, 2000, 2010],
        "Anomaly": [0.1, 0.2, 0.3, 0.4],
    })
    filtered = filter_by_date_range(df, 1990, 2005)
    assert list(filtered["Year"]) == [1990, 2000]
