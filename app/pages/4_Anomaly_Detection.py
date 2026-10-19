"""
Climate Trend Analyzer - Page 4: Statistical Anomaly Detection.
"""

import sys
import os
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset
from src.preprocessing import filter_by_date_range
from src.anomaly_detection import detect_anomalies, ANOMALY_DISCLAIMER
from src.visualizations import plot_anomaly_detection


st.set_page_config(page_title="Anomaly Detection | Climate Trend Analyzer", page_icon="🚨", layout="wide")

st.title("🚨 Statistical Anomaly Detection")
st.markdown("Identify mathematically unusual observations using standardized deviation, non-parametric fences, and machine learning.")

if "df" not in st.session_state:
    df, _ = load_dataset("Global (Land & Ocean)")
    st.session_state["df"] = df
else:
    df = st.session_state["df"]

selected_domain = st.session_state.get("selected_domain", "Global (Land & Ocean)")

min_yr = int(df["Year"].min())
max_yr = int(df["Year"].max())

# Sidebar Method Controls
st.sidebar.header("Anomaly Detection Setup")

method_choice = st.sidebar.selectbox(
    "Detection Method",
    [
        "Z-Score (Standard Deviation Threshold)",
        "Tukey's IQR Fences (Non-Parametric)",
        "Isolation Forest (Unsupervised ML)",
    ],
    index=0,
    help="Select the statistical or machine learning algorithm for identifying outliers."
)

z_thresh = 2.5
iqr_mult = 1.5
if_contam = 0.02

if "Z-Score" in method_choice:
    method_key = "zscore"
    z_thresh = st.sidebar.slider("Z-Score Threshold (σ)", min_value=2.0, max_value=4.0, value=2.5, step=0.1)
elif "Tukey" in method_choice:
    method_key = "iqr"
    iqr_mult = st.sidebar.slider("IQR Multiplier", min_value=1.5, max_value=3.0, value=1.5, step=0.5)
else:
    method_key = "isolation_forest"
    if_contam = st.sidebar.slider("Contamination Fraction (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100.0

# Scientific Disclaimer Banner
st.warning(f"⚠️ **{ANOMALY_DISCLAIMER}**")

# Run Detection
active_df = df.copy()
results = detect_anomalies(
    active_df,
    method=method_key,
    z_threshold=z_thresh,
    iqr_multiplier=iqr_mult,
    if_contamination=if_contam,
)

annotated_df = results["annotated_df"]
flagged_df = results["flagged_table"]

# Metric Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Observations", f"{results['total_observations']:,}")
with col2:
    st.metric("Flagged Observations", f"{results['flagged_count']:,}")
with col3:
    st.metric("Outlier Rate", f"{results['flagged_percentage']:.2f}%")
with col4:
    st.metric("Algorithm Applied", results["method_used"].split("(")[0].strip())

st.markdown("---")

# Main Interactive Chart with Highlighted Points
st.subheader("📍 Anomaly Scatter & Time-Series Plot")
st.caption("Points in coral/red deviate beyond the configured statistical threshold.")
fig_anom = plot_anomaly_detection(
    annotated_df,
    method_name=results["method_used"],
    title=f"{selected_domain} Flagged Anomalies"
)
st.plotly_chart(fig_anom, use_container_width=True)

# Table of Flagged Observations
st.subheader(f"📋 Table of Flagged Observations ({results['flagged_count']} Records)")

if not flagged_df.empty:
    display_cols = ["Date", "Year", "Month", "Anomaly", "Severity"]
    if "Z_Score" in flagged_df.columns:
        display_cols.append("Z_Score")
    if "Anomaly_Score" in flagged_df.columns:
        display_cols.append("Anomaly_Score")

    st.dataframe(
        flagged_df[display_cols].sort_values(by="Date", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    # Download flagged table as CSV
    csv_anom = flagged_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Flagged Anomalies Table (CSV)",
        data=csv_anom,
        file_name=f"flagged_anomalies_{method_key}.csv",
        mime="text/csv",
    )
else:
    st.success("No statistical anomalies flagged under the selected threshold settings.")
