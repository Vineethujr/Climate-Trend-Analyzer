"""
Climate Trend Analyzer - Long-Term Trend & Statistical Significance Module.

Responsible for:
- Parametric trend estimation via Ordinary Least Squares (OLS) regression.
- Non-parametric robust slope estimation via the Theil-Sen estimator.
- Monotonic trend verification via the Mann-Kendall test.
- Autocorrelation diagnostics (Lag-1 ACF, Durbin-Watson, effective sample size).
- Multi-era trend comparisons (°C/decade) with 95% confidence intervals.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_ols_trend(
    df: pd.DataFrame,
    x_col: str = "Decimal_Year",
    y_col: str = "Anomaly",
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Estimate long-term temperature trend using Ordinary Least Squares (OLS).

    Args:
        df: Input DataFrame with x and y columns.
        x_col: Time coordinate column (e.g. 'Decimal_Year' or 'Year').
        y_col: Temperature anomaly column.
        confidence_level: Desired two-tailed confidence level (default 0.95).

    Returns:
        Dictionary of trend metrics, rates per decade/century, and 95% CIs.
    """
    valid = df[[x_col, y_col]].dropna()
    n = len(valid)

    if n < 3:
        raise ValueError(f"Insufficient observations for trend regression (n={n}).")

    x = valid[x_col].to_numpy(dtype=float)
    y = valid[y_col].to_numpy(dtype=float)

    reg = stats.linregress(x, y)
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    r_value = float(reg.rvalue)
    r_squared = float(r_value ** 2)
    p_value = float(reg.pvalue)
    stderr = float(reg.stderr)

    # Student's t critical value for two-tailed (n - 2) degrees of freedom
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 2))
    ci_margin_per_year = t_crit * stderr

    # Climatological units (per decade and per century)
    rate_per_decade = slope * 10.0
    ci_decade_lower = (slope - ci_margin_per_year) * 10.0
    ci_decade_upper = (slope + ci_margin_per_year) * 10.0

    rate_per_century = slope * 100.0
    total_warming = slope * (x.max() - x.min())

    # Predicted trend values
    trend_pred = intercept + slope * x

    return {
        "observations_count": n,
        "start_time": float(x.min()),
        "end_time": float(x.max()),
        "duration_years": float(x.max() - x.min()),
        "slope_per_year": round(slope, 6),
        "rate_per_decade": round(rate_per_decade, 4),
        "ci_decade_lower": round(ci_decade_lower, 4),
        "ci_decade_upper": round(ci_decade_upper, 4),
        "rate_per_century": round(rate_per_century, 3),
        "total_warming_estimated": round(total_warming, 3),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "p_value": p_value,
        "is_statistically_significant": bool(p_value < 0.05),
        "std_error_slope": round(stderr, 6),
        "trend_predictions": trend_pred,
        "x_values": x,
        "y_values": y,
    }


def compute_theil_sen_trend(
    df: pd.DataFrame,
    x_col: str = "Decimal_Year",
    y_col: str = "Anomaly",
    alpha: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute non-parametric robust linear slope using the Theil-Sen estimator.
    The slope is defined as the median of all pairwise slopes:
        beta_TS = median{ (y_j - y_i) / (x_j - x_i) }

    Args:
        df: Input DataFrame.
        x_col: Continuous time column.
        y_col: Temperature anomaly column.
        alpha: Confidence level.

    Returns:
        Dictionary with Theil-Sen median slope and decadal warming rate.
    """
    valid = df[[x_col, y_col]].dropna()
    if len(valid) < 3:
        raise ValueError("Insufficient observations for Theil-Sen estimation.")

    x = valid[x_col].to_numpy(dtype=float)
    y = valid[y_col].to_numpy(dtype=float)

    # Scipy theilslopes returns (medslope, medintercept, lo_slope, up_slope)
    res = stats.theilslopes(y, x, alpha=alpha)
    med_slope = float(res.slope)
    med_intercept = float(res.intercept)
    lo_slope = float(res.low_slope)
    up_slope = float(res.high_slope)

    return {
        "theil_sen_slope_per_year": round(med_slope, 6),
        "theil_sen_rate_per_decade": round(med_slope * 10.0, 4),
        "theil_sen_ci_lower_decade": round(lo_slope * 10.0, 4),
        "theil_sen_ci_upper_decade": round(up_slope * 10.0, 4),
        "theil_sen_intercept": round(med_intercept, 4),
    }


def compute_mann_kendall_test(
    df: pd.DataFrame,
    y_col: str = "Anomaly",
) -> Dict[str, Any]:
    """
    Execute Mann-Kendall non-parametric monotonic trend test.
    Tests null hypothesis H0 (no trend) vs H1 (monotonic trend exists).

    Args:
        df: Input DataFrame in chronological order.
        y_col: Target series column name.

    Returns:
        Dictionary with Kendall's tau, p-value, and trend interpretation.
    """
    y = df[y_col].dropna().to_numpy(dtype=float)
    n = len(y)

    if n < 4:
        return {"trend": "Insufficient data", "p_value": 1.0, "tau": 0.0}

    # Rank correlation against sequential index
    x = np.arange(n)
    tau, p_val = stats.kendalltau(x, y)

    if p_val < 0.05:
        trend_label = "Statistically Significant Upward Trend" if tau > 0 else "Statistically Significant Downward Trend"
    else:
        trend_label = "No Statistically Significant Monotonic Trend"

    return {
        "kendall_tau": round(float(tau), 4),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "trend_classification": trend_label,
        "sample_size": n,
    }


def compute_autocorrelation_diagnostics(
    df: pd.DataFrame,
    y_col: str = "Anomaly",
    max_lag: int = 12,
) -> Dict[str, Any]:
    """
    Evaluate serial autocorrelation and effective degrees of freedom.

    Args:
        df: Input DataFrame.
        y_col: Target series.
        max_lag: Number of trailing monthly lags to compute.

    Returns:
        Dictionary with Durbin-Watson statistic, lag-1 ACF, and effective sample size.
    """
    y = df[y_col].dropna().to_numpy(dtype=float)
    n = len(y)

    if n < max_lag + 2:
        return {"lag_1_autocorrelation": 0.0, "durbin_watson": 2.0, "effective_n": n}

    y_mean = np.mean(y)
    y_centered = y - y_mean
    variance = np.sum(y_centered ** 2)

    acf_lags = {}
    for lag in range(1, max_lag + 1):
        num = np.sum(y_centered[:-lag] * y_centered[lag:])
        acf_val = float(num / variance) if variance > 0 else 0.0
        acf_lags[f"lag_{lag}"] = round(acf_val, 4)

    r1 = acf_lags["lag_1"]

    # Durbin-Watson statistic: d ~ 2*(1 - r1)
    diff = np.diff(y)
    dw_stat = float(np.sum(diff ** 2) / variance) if variance > 0 else 2.0

    # Effective sample size: N_eff = N * (1 - r1) / (1 + r1)
    if -1.0 < r1 < 1.0:
        n_eff = int(max(3, round(n * ((1.0 - r1) / (1.0 + r1)))))
    else:
        n_eff = n

    return {
        "lag_1_autocorrelation": r1,
        "durbin_watson_stat": round(dw_stat, 3),
        "effective_sample_size": n_eff,
        "nominal_sample_size": n,
        "acf_by_lag": acf_lags,
        "autocorrelation_note": (
            "Positive serial correlation (r1 > 0) inflates standard OLS t-statistics. "
            "The effective degrees of freedom should be referenced when assessing statistical significance."
        ),
    }


def compare_era_trends(
    df: pd.DataFrame,
    custom_eras: Optional[Dict[str, Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Calculate and contrast warming rates across landmark climatological eras.

    Args:
        df: DataFrame with 'Year' and 'Decimal_Year' and 'Anomaly'.
        custom_eras: Optional dict of era names to (start_year, end_year).

    Returns:
        Summary DataFrame comparing warming rates (°C/decade) across eras.
    """
    eras = custom_eras or {
        "Full Historical Record": (1850, 2026),
        "Pre-Industrial / Early": (1850, 1900),
        "Mid-20th Century Transition": (1901, 1975),
        "Modern Acceleration Era": (1976, 2026),
        "21st Century Only": (2000, 2026),
    }

    records = []
    for era_name, (start_y, end_y) in eras.items():
        sub = df[(df["Year"] >= start_y) & (df["Year"] <= end_y)]
        if len(sub) >= 24:  # At least 2 years of observations
            res = compute_ols_trend(sub)
            records.append({
                "Era": era_name,
                "Period": f"{int(res['start_time'])}-{int(res['end_time'])}",
                "Months": res["observations_count"],
                "Rate (°C/Decade)": res["rate_per_decade"],
                "95% CI Lower": res["ci_decade_lower"],
                "95% CI Upper": res["ci_decade_upper"],
                "R-Squared": res["r_squared"],
                "p-value": "<0.0001" if res["p_value"] < 0.0001 else round(res["p_value"], 4),
            })

    return pd.DataFrame(records)
