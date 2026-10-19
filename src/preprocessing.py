"""
Climate Trend Analyzer - Data Preprocessing & Aggregation Module.

Responsible for:
- Computing trailing and centered rolling averages (12-month, 60-month).
- Re-centering temperature anomalies to custom reference baselines.
- Aggregating monthly series into annual and decadal climate summaries.
- Assigning meteorological seasons (DJF, MAM, JJA, SON).
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


def compute_rolling_averages(
    df: pd.DataFrame,
    windows: List[int] = (12, 60),
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute trailing moving averages over specified monthly window sizes.

    Args:
        df: Input DataFrame containing 'Anomaly'.
        windows: List of integer monthly window sizes (e.g. 12 for 1-year, 60 for 5-year).
        min_periods: Minimum observations in window required to have a value (default window // 2).

    Returns:
        DataFrame with added 'Rolling_<W>m' columns.
    """
    result = df.copy()
    if "Anomaly" not in result.columns:
        return result

    for w in windows:
        col_name = f"Rolling_{w}m"
        min_p = min_periods if min_periods is not None else max(1, w // 2)
        result[col_name] = (
            result["Anomaly"]
            .rolling(window=w, min_periods=min_p)
            .mean()
            .round(4)
        )

    return result


def adjust_baseline(
    df: pd.DataFrame,
    base_start_year: int,
    base_end_year: int,
    target_col: str = "Anomaly",
) -> Tuple[pd.DataFrame, float]:
    """
    Re-center anomalies relative to an arbitrary climatological baseline period.

    Formula:
        Anomaly_new(t) = Anomaly(t) - mean(Anomaly_{base_start .. base_end})

    Args:
        df: Input DataFrame.
        base_start_year: Beginning year of reference baseline (inclusive).
        base_end_year: Ending year of reference baseline (inclusive).
        target_col: Column to re-center.

    Returns:
        Tuple of (re-centered DataFrame, baseline_offset_applied).
    """
    result = df.copy()
    if target_col not in result.columns or "Year" not in result.columns:
        return result, 0.0

    mask = (result["Year"] >= base_start_year) & (result["Year"] <= base_end_year)
    baseline_slice = result.loc[mask, target_col].dropna()

    if baseline_slice.empty:
        raise ValueError(
            f"No observations available in baseline window {base_start_year}-{base_end_year}."
        )

    baseline_offset = float(baseline_slice.mean())

    result[target_col] = (result[target_col] - baseline_offset).round(4)

    # Recompute any rolling average columns that exist
    rolling_cols = [c for c in result.columns if c.startswith("Rolling_") and c.endswith("m")]
    if rolling_cols:
        windows = [int(c.replace("Rolling_", "").replace("m", "")) for c in rolling_cols]
        result = compute_rolling_averages(result, windows=windows)

    return result, baseline_offset


def aggregate_annual(
    df: pd.DataFrame,
    min_months: int = 12,
    anomaly_col: str = "Anomaly",
) -> pd.DataFrame:
    """
    Aggregate monthly records into annual climatological summaries.

    Args:
        df: Cleaned monthly DataFrame.
        min_months: Minimum months required in a year to calculate an annual average (default 12).
        anomaly_col: Column name containing anomalies.

    Returns:
        DataFrame with annual statistics: ['Year', 'Annual_Mean', 'Min', 'Max', 'Months_Count'].
    """
    if df.empty or "Year" not in df.columns or anomaly_col not in df.columns:
        return pd.DataFrame(columns=["Year", "Annual_Mean", "Min", "Max", "Months_Count"])

    grouped = df.groupby("Year")[anomaly_col].agg(
        Annual_Mean="mean",
        Min="min",
        Max="max",
        Months_Count="count",
    ).reset_index()

    # Filter out incomplete years to avoid seasonal bias
    complete_years = grouped[grouped["Months_Count"] >= min_months].copy()
    complete_years["Annual_Mean"] = complete_years["Annual_Mean"].round(4)
    complete_years["Min"] = complete_years["Min"].round(4)
    complete_years["Max"] = complete_years["Max"].round(4)

    return complete_years.reset_index(drop=True)


def aggregate_decadal(
    df: pd.DataFrame,
    anomaly_col: str = "Anomaly",
) -> pd.DataFrame:
    """
    Group observations into 10-year climatological decades (e.g. 1850-1859).

    Args:
        df: Monthly or annual DataFrame.
        anomaly_col: Temperature anomaly column.

    Returns:
        DataFrame with decadal means, standard deviations, and sample counts.
    """
    if df.empty or "Year" not in df.columns or anomaly_col not in df.columns:
        return pd.DataFrame(columns=["Decade", "Decade_Mean", "Std_Dev", "Count"])

    temp = df.copy()
    temp["Decade"] = (temp["Year"] // 10) * 10
    temp["Decade_Label"] = temp["Decade"].astype(str) + "s"

    decadal = temp.groupby(["Decade", "Decade_Label"])[anomaly_col].agg(
        Decade_Mean="mean",
        Std_Dev="std",
        Count="count",
    ).reset_index()

    decadal["Decade_Mean"] = decadal["Decade_Mean"].round(4)
    decadal["Std_Dev"] = decadal["Std_Dev"].round(4)

    return decadal.sort_values(by="Decade").reset_index(drop=True)


def add_meteorological_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign standard meteorological season labels to monthly records.
    - DJF: Dec, Jan, Feb (Winter in NH, Summer in SH)
    - MAM: Mar, Apr, May (Spring in NH, Autumn in SH)
    - JJA: Jun, Jul, Aug (Summer in NH, Winter in SH)
    - SON: Sep, Oct, Nov (Autumn in NH, Spring in SH)

    For December, Season_Year is assigned to Year + 1 so DJF forms a continuous winter season.

    Args:
        df: DataFrame with 'Year' and 'Month'.

    Returns:
        DataFrame with 'Season' and 'Season_Year' columns added.
    """
    if df.empty or "Month" not in df.columns:
        return df

    result = df.copy()
    season_map = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    result["Season"] = result["Month"].map(season_map)

    if "Year" in result.columns:
        # December belongs to the next year's DJF season
        result["Season_Year"] = np.where(
            result["Month"] == 12, result["Year"] + 1, result["Year"]
        )

    return result


def filter_by_date_range(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Filter DataFrame strictly within the [start_year, end_year] window.

    Args:
        df: Input DataFrame.
        start_year: Minimum year (inclusive).
        end_year: Maximum year (inclusive).

    Returns:
        Filtered DataFrame.
    """
    if df.empty or "Year" not in df.columns:
        return df

    mask = (df["Year"] >= start_year) & (df["Year"] <= end_year)
    return df[mask].copy().reset_index(drop=True)
