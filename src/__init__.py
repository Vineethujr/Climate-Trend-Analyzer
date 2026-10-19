"""
Climate Trend Analyzer - Core Analytical Engine.

Modules:
    data_loader: Ingestion, network fetching, and local cache management.
    data_cleaner: Schema validation, null handling, and domain sanity checks.
    preprocessing: Baseline adjustments, moving averages, and temporal aggregation.
    trend_analysis: OLS linear regression, Theil-Sen robust slope, Mann-Kendall test.
    seasonal_analysis: Climatological cycles, meteorological seasons, decomposition.
    anomaly_detection: Z-score thresholds, Tukey IQR fences, Isolation Forest.
    predictive_model: Chronological train/test validation and baseline autoregression.
    visualizations: Publication-quality interactive Plotly chart builders.
"""

__version__ = "1.0.0"
__author__ = "Climate Trend Analyzer Team"
