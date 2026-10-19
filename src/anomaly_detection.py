"""
Climate Trend Analyzer - Statistical Anomaly Detection Module.

Responsible for:
- Detecting unusual observations using standardized deviation (Z-score).
- Detecting non-parametric outliers using Tukey's Interquartile Range (IQR) fences.
- Applying unsupervised machine learning via Scikit-learn Isolation Forest.
- Generating exportable anomaly inspection tables with clear scientific disclaimers.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ANOMALY_DISCLAIMER = (
    "SCIENTIFIC NOTICE: In this analytical tool, an 'anomaly' refers strictly to a mathematical "
    "outlier relative to the statistical properties of the chosen historical dataset and method. "
    "Flagged points must NOT be interpreted as confirmed extreme weather disasters, local catastrophic events, "
    "or instrument failures without corroborating meteorological and station metadata."
)


def detect_zscore_anomalies(
    df: pd.DataFrame,
    threshold: float = 2.5,
    anomaly_col: str = "Anomaly",
) -> pd.DataFrame:
    """
    Identify statistical anomalies using the standardized deviation (Z-Score) method.
        Z = (x - mean) / std

    Args:
        df: Input DataFrame.
        threshold: Standard deviation cutoff (|z| > threshold).
        anomaly_col: Target column name.

    Returns:
        DataFrame with added 'Z_Score', 'Is_Anomaly', and 'Severity' columns.
    """
    result = df.copy()
    if anomaly_col not in result.columns or result.empty:
        return result

    valid = result[anomaly_col].dropna()
    mean_val = float(valid.mean())
    std_val = float(valid.std(ddof=1))

    if std_val == 0.0 or np.isnan(std_val):
        result["Z_Score"] = 0.0
        result["Is_Anomaly"] = False
        result["Severity"] = "Normal"
        return result

    result["Z_Score"] = ((result[anomaly_col] - mean_val) / std_val).round(3)
    result["Is_Anomaly"] = result["Z_Score"].abs() > threshold

    # Categorize severity
    conditions = [
        result["Z_Score"].abs() > 3.5,
        result["Z_Score"].abs() > threshold,
    ]
    choices = ["Extreme Deviation", "Moderate Deviation"]
    result["Severity"] = np.select(conditions, choices, default="Normal")

    return result


def detect_iqr_anomalies(
    df: pd.DataFrame,
    multiplier: float = 1.5,
    anomaly_col: str = "Anomaly",
) -> pd.DataFrame:
    """
    Identify statistical anomalies using Tukey's Interquartile Range (IQR) fences.
        Lower Fence = Q1 - multiplier * IQR
        Upper Fence = Q3 + multiplier * IQR

    Args:
        df: Input DataFrame.
        multiplier: Tukey fence multiplier (typically 1.5 for outliers, 3.0 for extreme outliers).
        anomaly_col: Target column name.

    Returns:
        DataFrame with added 'Is_Anomaly' and 'IQR_Fence' columns.
    """
    result = df.copy()
    if anomaly_col not in result.columns or result.empty:
        return result

    valid = result[anomaly_col].dropna()
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1

    lower_fence = round(q1 - multiplier * iqr, 4)
    upper_fence = round(q3 + multiplier * iqr, 4)

    is_outlier = (result[anomaly_col] < lower_fence) | (result[anomaly_col] > upper_fence)
    result["Is_Anomaly"] = is_outlier.fillna(False)
    result["IQR_Lower_Fence"] = lower_fence
    result["IQR_Upper_Fence"] = upper_fence

    conditions = [
        result[anomaly_col] < lower_fence,
        result[anomaly_col] > upper_fence,
    ]
    choices = ["Cold Departure Outlier", "Warm Departure Outlier"]
    result["Severity"] = np.select(conditions, choices, default="Normal")

    return result


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
    anomaly_col: str = "Anomaly",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect multivariate anomalies using an unsupervised Isolation Forest.

    Engineers temporal context features:
    - Target anomaly value
    - Monthly seasonal cycle sine/cosine projections
    - Rolling 12-month residual departure

    Args:
        df: Input DataFrame.
        contamination: Expected proportion of outliers in the dataset (0.01 - 0.05).
        anomaly_col: Target column name.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with 'Is_Anomaly' and 'Anomaly_Score'.
    """
    result = df.copy()
    if anomaly_col not in result.columns or len(result.dropna(subset=[anomaly_col])) < 24:
        result["Is_Anomaly"] = False
        result["Anomaly_Score"] = 0.0
        return result

    # Feature engineering for temporal context
    features = pd.DataFrame(index=result.index)
    features["val"] = result[anomaly_col].fillna(result[anomaly_col].mean())
    
    # Cyclic month encoding
    if "Month" in result.columns:
        m = result["Month"].fillna(6)
        features["m_sin"] = np.sin(2 * np.pi * m / 12.0)
        features["m_cos"] = np.cos(2 * np.pi * m / 12.0)
    else:
        features["m_sin"] = 0.0
        features["m_cos"] = 0.0

    # Local trend residual (difference from 12-month moving average)
    rolling_12 = features["val"].rolling(window=12, min_periods=1).mean()
    features["diff_from_local_mean"] = features["val"] - rolling_12

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    preds = iso.fit_predict(features)
    # Decision function: lower values mean more anomalous
    scores = iso.decision_function(features)

    result["Is_Anomaly"] = preds == -1
    result["Anomaly_Score"] = np.round(scores, 4)
    result["Severity"] = np.where(result["Is_Anomaly"], "Isolation Forest Outlier", "Normal")

    return result


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "zscore",
    z_threshold: float = 2.5,
    iqr_multiplier: float = 1.5,
    if_contamination: float = 0.02,
    anomaly_col: str = "Anomaly",
) -> Dict[str, Any]:
    """
    Unified entry point for statistical anomaly detection.

    Args:
        df: Input DataFrame.
        method: One of ['zscore', 'iqr', 'isolation_forest'].
        z_threshold: Z-score threshold.
        iqr_multiplier: IQR multiplier.
        if_contamination: Isolation Forest contamination fraction.
        anomaly_col: Target anomaly column.

    Returns:
        Dictionary containing:
        - 'annotated_df': Full DataFrame with anomaly flags.
        - 'flagged_table': DataFrame containing only flagged records.
        - 'total_observations': Total count.
        - 'flagged_count': Number of flagged records.
        - 'flagged_percentage': Percentage flagged.
        - 'method_used': Name of method.
        - 'disclaimer': Scientific interpretation warning.
    """
    method_clean = method.lower().strip()

    if method_clean in ["zscore", "z-score", "standard_deviation"]:
        annotated = detect_zscore_anomalies(df, threshold=z_threshold, anomaly_col=anomaly_col)
        method_name = f"Z-Score (|z| > {z_threshold}σ)"
        params = {"threshold": z_threshold}
    elif method_clean in ["iqr", "tukey", "interquartile_range"]:
        annotated = detect_iqr_anomalies(df, multiplier=iqr_multiplier, anomaly_col=anomaly_col)
        method_name = f"Tukey's IQR Fences (multiplier = {iqr_multiplier})"
        params = {"multiplier": iqr_multiplier}
    elif method_clean in ["isolation_forest", "iforest", "ml"]:
        annotated = detect_isolation_forest_anomalies(
            df, contamination=if_contamination, anomaly_col=anomaly_col
        )
        method_name = f"Isolation Forest (contamination = {if_contamination * 100:.1f}%)"
        params = {"contamination": if_contamination}
    else:
        raise ValueError(
            f"Unsupported anomaly method: '{method}'. Choose from ['zscore', 'iqr', 'isolation_forest']."
        )

    flagged_mask = annotated["Is_Anomaly"] == True
    flagged_df = annotated[flagged_mask].copy()

    total_n = len(annotated)
    flagged_n = len(flagged_df)
    pct = round((flagged_n / total_n * 100), 2) if total_n > 0 else 0.0

    return {
        "annotated_df": annotated,
        "flagged_table": flagged_df,
        "total_observations": total_n,
        "flagged_count": flagged_n,
        "flagged_percentage": pct,
        "method_used": method_name,
        "parameters": params,
        "disclaimer": ANOMALY_DISCLAIMER,
    }
