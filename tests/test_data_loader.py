"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
from src.data_loader import parse_noaa_csv, load_dataset, get_available_domains


SAMPLE_RAW_CSV = """# Title: Global Land and Ocean Average Temperature Departures
# Units: Degrees Celsius
# Base Period: 1901-2000
Date,Departure from Average
188001,-0.19
188002,-0.25
188003,-0.09
188004,-0.16
"""


def test_parse_noaa_csv_structure():
    df, meta = parse_noaa_csv(SAMPLE_RAW_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df.columns) == ["Date", "Year", "Month", "Decimal_Year", "Anomaly"]
    assert meta["units"] == "Degrees Celsius"
    assert meta["base_period"] == "1901-2000"
    assert df["Year"].iloc[0] == 1880
    assert df["Month"].iloc[0] == 1
    assert df["Anomaly"].iloc[0] == -0.19


def test_parse_noaa_csv_empty():
    with pytest.raises(ValueError):
        parse_noaa_csv("# Just a comment\n# Another comment\n")


def test_load_real_cached_dataset():
    # Load global real dataset from cached data/raw
    df, meta = load_dataset("Global (Land & Ocean)")
    assert not df.empty
    assert len(df) > 1500  # Should have >170 years of monthly data
    assert meta["is_synthetic"] is False
    assert meta["domain"] == "Global (Land & Ocean)"
    assert "Anomaly" in df.columns
    assert "Year" in df.columns


def test_available_domains():
    domains = get_available_domains()
    assert "Global (Land & Ocean)" in domains
    assert "Northern Hemisphere (Land & Ocean)" in domains
    assert "Southern Hemisphere (Land & Ocean)" in domains
