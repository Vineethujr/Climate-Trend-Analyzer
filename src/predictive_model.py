"""
Climate Trend Analyzer - Responsible Predictive Modeling & Evaluation Module.

Responsible for:
- Enforcing strict chronological train/test temporal splitting (no random shuffling).
- Training defensible baseline regression models (Linear Trend and Autoregressive lags).
- Computing out-of-sample evaluation metrics (MAE, RMSE, R-squared).
- Generating short-term projections with uncertainty intervals and educational limitations.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_DISCLAIMER = (
    "EDUCATIONAL NOTICE: The predictive models implemented here are empirical time-series baselines "
    "intended for statistical evaluation and student learning. Empirical statistical models cannot "
    "simulate complex atmospheric fluid dynamics, oceanic circulation changes, or non-linear greenhouse "
    "gas feedback mechanisms. They must NOT be treated as authoritative IPCC-class climate projections."
)


def chronological_split(
    df: pd.DataFrame,
    split_year: int = 2015,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series strictly chronologically by calendar year to avoid temporal leakage.

    Args:
        df: Input DataFrame with 'Year'.
        split_year: First year of the holdout test set.

    Returns:
        Tuple of (train_df, test_df).
    """
    if df.empty or "Year" not in df.columns:
        raise ValueError("Cannot perform chronological split on empty dataset without 'Year'.")

    sorted_df = df.sort_values(by=["Year", "Month"]).reset_index(drop=True)
    train_mask = sorted_df["Year"] < split_year
    test_mask = sorted_df["Year"] >= split_year

    train_df = sorted_df[train_mask].copy().reset_index(drop=True)
    test_df = sorted_df[test_mask].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Split year {split_year} resulted in an empty train or test partition. "
            f"Available years: {int(sorted_df['Year'].min())}-{int(sorted_df['Year'].max())}."
        )

    return train_df, test_df


def train_and_evaluate_models(
    df: pd.DataFrame,
    split_year: int = 2015,
    anomaly_col: str = "Anomaly",
) -> Dict[str, Any]:
    """
    Train and evaluate time-series forecasting models using a chronological holdout set.

    Models evaluated:
    1. Naive Mean Baseline (constant prediction = train mean)
    2. Linear Trend Model (OLS on Decimal_Year)
    3. Autoregressive Model with Trend + Seasonal Harmonics (Ridge Regression)

    Args:
        df: Input DataFrame.
        split_year: Holdout split year.
        anomaly_col: Target anomaly column.

    Returns:
        Dictionary with model performance metrics, predictions, and test coordinates.
    """
    clean_df = df.dropna(subset=[anomaly_col, "Decimal_Year", "Month"]).copy()
    train_df, test_df = chronological_split(clean_df, split_year=split_year)

    y_train = train_df[anomaly_col].to_numpy()
    y_test = test_df[anomaly_col].to_numpy()

    # 1. Baseline: Persistent Mean
    mean_pred = float(np.mean(y_train))
    y_pred_baseline = np.full_like(y_test, fill_value=mean_pred)

    # 2. Linear Trend Model
    X_train_lin = train_df[["Decimal_Year"]].to_numpy()
    X_test_lin = test_df[["Decimal_Year"]].to_numpy()

    lin_model = LinearRegression()
    lin_model.fit(X_train_lin, y_train)
    y_pred_linear = lin_model.predict(X_test_lin)

    # 3. Autoregressive + Seasonal Harmonics Model
    def build_features(data_subset: pd.DataFrame) -> np.ndarray:
        t = data_subset["Decimal_Year"].to_numpy()
        m = data_subset["Month"].to_numpy()
        # Seasonal harmonics
        sin_m = np.sin(2 * np.pi * m / 12.0)
        cos_m = np.cos(2 * np.pi * m / 12.0)
        # Normalized time
        t_norm = (t - 1850.0) / 100.0
        return np.column_stack([t_norm, t_norm**2, sin_m, cos_m])

    X_train_harm = build_features(train_df)
    X_test_harm = build_features(test_df)

    harm_model = Ridge(alpha=1.0)
    harm_model.fit(X_train_harm, y_train)
    y_pred_harm = harm_model.predict(X_test_harm)

    # Compute Metrics Helper
    def get_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(actual, pred))
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        r2 = float(r2_score(actual, pred))
        max_err = float(np.max(np.abs(actual - pred)))
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Max_Error": round(max_err, 4),
        }

    metrics_baseline = get_metrics(y_test, y_pred_baseline)
    metrics_linear = get_metrics(y_test, y_pred_linear)
    metrics_harm = get_metrics(y_test, y_pred_harm)

    summary_comparison = pd.DataFrame([
        {"Model": "Persistent Mean Baseline", **metrics_baseline},
        {"Model": "Linear Trend Extrapolation", **metrics_linear},
        {"Model": "Polynomial + Seasonal Harmonics (Ridge)", **metrics_harm},
    ])

    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_period": f"{int(train_df['Year'].min())}-{int(train_df['Year'].max())}",
        "test_period": f"{int(test_df['Year'].min())}-{int(test_df['Year'].max())}",
        "comparison_table": summary_comparison,
        "test_dates": test_df["Date"].tolist(),
        "test_decimal_years": test_df["Decimal_Year"].tolist(),
        "actual_test_values": y_test.tolist(),
        "predictions": {
            "Baseline Mean": y_pred_baseline.tolist(),
            "Linear Trend": y_pred_linear.tolist(),
            "Harmonic Trend": y_pred_harm.tolist(),
        },
        "disclaimer": MODEL_DISCLAIMER,
    }


def generate_future_projection(
    df: pd.DataFrame,
    horizon_months: int = 60,
    anomaly_col: str = "Anomaly",
) -> pd.DataFrame:
    """
    Project baseline trend into future months with empirical uncertainty intervals.

    Args:
        df: Input DataFrame.
        horizon_months: Months to extrapolate forward (default 60 = 5 years).
        anomaly_col: Target column.

    Returns:
        DataFrame with projected future decimal years, point estimates, and lower/upper 95% bounds.
    """
    clean_df = df.dropna(subset=[anomaly_col, "Decimal_Year", "Month"]).sort_values(
        by=["Year", "Month"]
    ).copy()
    
    last_year = int(clean_df["Year"].iloc[-1])
    last_month = int(clean_df["Month"].iloc[-1])

    # Fit linear trend on modern acceleration era (1976 onwards)
    modern_df = clean_df[clean_df["Year"] >= 1976]
    X_mod = modern_df[["Decimal_Year"]].to_numpy()
    y_mod = modern_df[anomaly_col].to_numpy()

    lin = LinearRegression()
    lin.fit(X_mod, y_mod)

    # Estimate standard error of residuals
    residuals = y_mod - lin.predict(X_mod)
    sigma_resid = float(np.std(residuals, ddof=2))

    # Construct future dates
    future_records = []
    curr_y = last_year
    curr_m = last_month

    for _ in range(horizon_months):
        curr_m += 1
        if curr_m > 12:
            curr_m = 1
            curr_y += 1
        
        dec_y = curr_y + (curr_m - 0.5) / 12.0
        pred_val = float(lin.predict([[dec_y]])[0])
        # 95% empirical forecast interval: +/- 1.96 * sigma
        future_records.append({
            "Date": f"{curr_y}{curr_m:02d}",
            "Year": curr_y,
            "Month": curr_m,
            "Decimal_Year": round(dec_y, 4),
            "Projected_Anomaly": round(pred_val, 4),
            "Lower_95_CI": round(pred_val - 1.96 * sigma_resid, 4),
            "Upper_95_CI": round(pred_val + 1.96 * sigma_resid, 4),
        })

    return pd.DataFrame(future_records)
