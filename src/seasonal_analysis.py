"""
Climate Trend Analyzer - Seasonal Dynamics & Time Series Decomposition Module.

Responsible for:
- Computing monthly climatological anomaly profiles and spread metrics.
- Grouping meteorological seasons (DJF, MAM, JJA, SON) and evaluating seasonal warming contrasts.
- Performing classical additive time-series decomposition (Observed = Trend + Seasonal + Residual).
- Comparing monthly distributions across historical climatological epochs.
"""

from typing import Dict, Any, List, Optional, Tuple
import calendar
import numpy as np
import pandas as pd


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


def compute_monthly_climatology(
    df: pd.DataFrame,
    anomaly_col: str = "Anomaly",
) -> pd.DataFrame:
    """
    Calculate climatological statistics across each calendar month.

    Args:
        df: Input DataFrame with 'Month' and target anomaly column.
        anomaly_col: Column name containing anomalies.

    Returns:
        DataFrame with rows 1..12 and summary statistics.
    """
    if df.empty or "Month" not in df.columns or anomaly_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("Month")[anomaly_col].agg(
        Mean="mean",
        Std="std",
        Median="median",
        Min="min",
        Max="max",
        Q25=lambda s: s.quantile(0.25),
        Q75=lambda s: s.quantile(0.75),
        Count="count",
    ).reset_index()

    grouped["Month_Name"] = grouped["Month"].apply(lambda m: calendar.month_abbr[m])
    
    # Reorder columns
    cols = ["Month", "Month_Name", "Mean", "Std", "Median", "Min", "Max", "Q25", "Q75", "Count"]
    res = grouped[cols].copy()
    for col in ["Mean", "Std", "Median", "Min", "Max", "Q25", "Q75"]:
        res[col] = res[col].round(4)

    return res.sort_values(by="Month").reset_index(drop=True)


def compute_seasonal_summary(
    df: pd.DataFrame,
    anomaly_col: str = "Anomaly",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute aggregate seasonal metrics and analyze warming trends per individual season.

    Args:
        df: Input DataFrame with 'Year' and 'Month'.
        anomaly_col: Target anomaly column.

    Returns:
        Tuple of (seasonal_overall_summary_df, seasonal_trend_slopes_df).
    """
    from src.preprocessing import add_meteorological_season
    from src.trend_analysis import compute_ols_trend

    data = add_meteorological_season(df)

    # 1. Overall season summary
    overall = data.groupby("Season")[anomaly_col].agg(
        Mean="mean",
        Std="std",
        Min="min",
        Max="max",
        Count="count",
    ).reset_index()

    # Order seasons: DJF, MAM, JJA, SON
    season_order = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    overall["Order"] = overall["Season"].map(season_order)
    overall = overall.sort_values(by="Order").drop(columns=["Order"]).reset_index(drop=True)
    for c in ["Mean", "Std", "Min", "Max"]:
        overall[c] = overall[c].round(4)

    # 2. Season-specific warming rate per decade
    trend_records = []
    for s_name in ["DJF", "MAM", "JJA", "SON"]:
        season_df = data[data["Season"] == s_name].dropna(subset=[anomaly_col, "Year"])
        if len(season_df) >= 24:
            # Group by year for that season
            season_annual = season_df.groupby("Season_Year")[anomaly_col].mean().reset_index()
            season_annual = season_annual.rename(columns={"Season_Year": "Year"})
            if len(season_annual) >= 10:
                t_res = compute_ols_trend(season_annual, x_col="Year", y_col=anomaly_col)
                trend_records.append({
                    "Season": s_name,
                    "Description": {
                        "DJF": "Dec-Jan-Feb (Boreal Winter / Austral Summer)",
                        "MAM": "Mar-Apr-May (Boreal Spring / Austral Autumn)",
                        "JJA": "Jun-Jul-Aug (Boreal Summer / Austral Winter)",
                        "SON": "Sep-Oct-Nov (Boreal Autumn / Austral Spring)",
                    }[s_name],
                    "Warming Rate (°C/Decade)": t_res["rate_per_decade"],
                    "95% CI Lower": t_res["ci_decade_lower"],
                    "95% CI Upper": t_res["ci_decade_upper"],
                    "R-Squared": t_res["r_squared"],
                    "p-value": t_res["p_value"],
                })

    trend_df = pd.DataFrame(trend_records)
    return overall, trend_df


def decompose_time_series(
    df: pd.DataFrame,
    anomaly_col: str = "Anomaly",
    period: int = 12,
) -> pd.DataFrame:
    """
    Perform classical additive time-series decomposition:
        Observed = Trend + Seasonal + Residual

    1. Trend (T_t): Centered 12-month moving average filter.
    2. Detrended: D_t = Observed - T_t.
    3. Seasonal (S_t): Mean of detrended observations for each calendar month.
    4. Residual (R_t): Observed - Trend - Seasonal.

    Args:
        df: Input DataFrame in chronological monthly order.
        anomaly_col: Target column name.
        period: Seasonal period (12 for monthly).

    Returns:
        DataFrame with columns: ['Date', 'Decimal_Year', 'Observed', 'Trend', 'Seasonal', 'Residual'].
    """
    valid = df.dropna(subset=[anomaly_col]).sort_values(by=["Year", "Month"]).copy()
    n = len(valid)

    if n < period * 2:
        raise ValueError(
            f"Decomposition requires at least {period * 2} continuous observations (received {n})."
        )

    observed = valid[anomaly_col].to_numpy(dtype=float)

    # Centered moving average for Trend (period 12: 0.5 at ends, 1 in middle / 12)
    # Using pandas centered rolling mean
    s = pd.Series(observed)
    trend_series = s.rolling(window=period, center=True).mean()
    # Apply a 2-period moving average to center exactly on months
    trend_centered = trend_series.rolling(window=2, center=True).mean().to_numpy()

    # Detrend
    detrended = observed - trend_centered

    # Compute seasonal index per calendar month (1..12)
    months = valid["Month"].to_numpy()
    seasonal_component = np.zeros(n)
    monthly_season_means = {}

    for m in range(1, 13):
        m_mask = (months == m) & (~np.isnan(detrended))
        if np.any(m_mask):
            monthly_season_means[m] = float(np.mean(detrended[m_mask]))
        else:
            monthly_season_means[m] = 0.0

    # Ensure seasonal components sum to 0 across the 12 months
    mean_correction = np.mean(list(monthly_season_means.values()))
    for m in monthly_season_means:
        monthly_season_means[m] -= mean_correction

    # Assign seasonal values
    for i, m in enumerate(months):
        seasonal_component[i] = monthly_season_means[m]

    # Compute residuals: Residual = Observed - Trend - Seasonal
    residuals = observed - trend_centered - seasonal_component

    result_df = pd.DataFrame({
        "Date": valid["Date"].values,
        "Year": valid["Year"].values,
        "Month": valid["Month"].values,
        "Decimal_Year": valid["Decimal_Year"].values,
        "Observed": np.round(observed, 4),
        "Trend": np.round(trend_centered, 4),
        "Seasonal": np.round(seasonal_component, 4),
        "Residual": np.round(residuals, 4),
    })

    return result_df


def compare_decadal_seasonality(
    df: pd.DataFrame,
    anomaly_col: str = "Anomaly",
) -> pd.DataFrame:
    """
    Contrast monthly climatological means across 4 distinct epochs:
    - 1850–1899 (Early Industrial)
    - 1900–1949 (Early 20th Century)
    - 1950–1999 (Mid-to-Late 20th Century)
    - 2000–Present (21st Century)

    Args:
        df: Input monthly DataFrame.
        anomaly_col: Target column.

    Returns:
        Pivoted DataFrame with Month as index and Epochs as columns.
    """
    if df.empty or "Year" not in df.columns or "Month" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    conditions = [
        (temp["Year"] < 1900),
        (temp["Year"] >= 1900) & (temp["Year"] < 1950),
        (temp["Year"] >= 1950) & (temp["Year"] < 2000),
        (temp["Year"] >= 2000),
    ]
    choices = [
        "1850-1899 (Early)",
        "1900-1949 (Mid-Early)",
        "1950-1999 (Mid-Late)",
        "2000-Present (Recent)",
    ]
    temp["Epoch"] = np.select(conditions, choices, default="Other")

    epoch_pivot = temp.pivot_table(
        index="Month",
        columns="Epoch",
        values=anomaly_col,
        aggfunc="mean",
    )
    
    # Map index to month names
    epoch_pivot.index = [calendar.month_abbr[m] for m in epoch_pivot.index]
    return epoch_pivot.round(4)
