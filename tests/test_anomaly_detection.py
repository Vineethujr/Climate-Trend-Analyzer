"""
Unit tests for anomaly_detection module using known injected outliers.
"""

from tests.fixtures.synthetic_series import generate_anomaly_test_series
from src.anomaly_detection import (
    detect_zscore_anomalies,
    detect_iqr_anomalies,
    detect_isolation_forest_anomalies,
    detect_anomalies,
)


def test_zscore_detects_known_outliers():
    # Inject extreme spikes at indices 30 and 80
    injected = [(30, 5.0), (80, -4.5)]
    df, outlier_indices = generate_anomaly_test_series(n_months=120, injected_outliers=injected)
    
    annotated = detect_zscore_anomalies(df, threshold=3.0)
    flagged_indices = annotated.index[annotated["Is_Anomaly"] == True].tolist()
    
    # Injected outlier indices must be flagged
    for idx in outlier_indices:
        assert idx in flagged_indices
        assert annotated.loc[idx, "Severity"] == "Extreme Deviation"


def test_iqr_detects_known_outliers():
    injected = [(25, 4.0), (60, -3.8)]
    df, outlier_indices = generate_anomaly_test_series(n_months=100, injected_outliers=injected)
    
    annotated = detect_iqr_anomalies(df, multiplier=1.5)
    flagged_indices = annotated.index[annotated["Is_Anomaly"] == True].tolist()
    
    for idx in outlier_indices:
        assert idx in flagged_indices


def test_isolation_forest_detection():
    injected = [(15, 6.0), (45, -5.0)]
    df, outlier_indices = generate_anomaly_test_series(n_months=80, injected_outliers=injected)
    
    annotated = detect_isolation_forest_anomalies(df, contamination=0.05)
    flagged_indices = annotated.index[annotated["Is_Anomaly"] == True].tolist()
    
    # Must identify at least one of the major spikes
    assert any(idx in flagged_indices for idx in outlier_indices)


def test_detect_anomalies_wrapper():
    injected = [(10, 4.2)]
    df, _ = generate_anomaly_test_series(n_months=50, injected_outliers=injected)
    
    res = detect_anomalies(df, method="zscore", z_threshold=2.5)
    assert res["flagged_count"] >= 1
    assert "disclaimer" in res
    assert "Z-Score" in res["method_used"]
