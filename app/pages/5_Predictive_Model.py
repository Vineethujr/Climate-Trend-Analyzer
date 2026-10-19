"""
Climate Trend Analyzer - Page 5: Responsible Predictive Modeling & Evaluation.
"""

import sys
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset
from src.predictive_model import (
    train_and_evaluate_models,
    generate_future_projection,
    MODEL_DISCLAIMER,
)
from src.visualizations import plot_prediction_comparison


st.set_page_config(page_title="Predictive Modeling | Climate Trend Analyzer", page_icon="🔮", layout="wide")

st.title("🔮 Responsible Predictive Modeling & Evaluation")
st.markdown("Chronological train/test splitting, baseline benchmarking, and out-of-sample error evaluation.")

# Prominent Educational Notice
st.info(f"ℹ️ **{MODEL_DISCLAIMER}**")

if "df" not in st.session_state:
    df, _ = load_dataset("Global (Land & Ocean)")
    st.session_state["df"] = df
else:
    df = st.session_state["df"]

selected_domain = st.session_state.get("selected_domain", "Global (Land & Ocean)")

min_yr = int(df["Year"].min())
max_yr = int(df["Year"].max())

# Sidebar Split Controls
st.sidebar.header("Evaluation Partitioning")
split_year = st.sidebar.slider(
    "Chronological Split Year",
    min_value=1990,
    max_value=max_yr - 3,
    value=2015,
    step=1,
    help="Observations before this year form the training set; observations on or after form the out-of-sample test set."
)

horizon_months = st.sidebar.slider(
    "Projection Horizon (Months)",
    min_value=12,
    max_value=120,
    value=60,
    step=12,
    help="Number of future months to extrapolate baseline trend."
)

# Train & Evaluate Models
eval_results = train_and_evaluate_models(df, split_year=split_year)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Period", f"{eval_results['train_period']} ({eval_results['train_size']:,} months)")
with col2:
    st.metric("Holdout Test Period", f"{eval_results['test_period']} ({eval_results['test_size']:,} months)")
with col3:
    st.metric("Models Evaluated", "3 (Baseline, Linear, Harmonic)")

st.markdown("---")

# Main Comparison Chart
st.subheader("📊 Out-of-Sample Model Predictions vs. Actual Observations")
st.caption(f"Testing holdout from {split_year} to {max_yr}. No future data was leaked during training.")
fig_eval = plot_prediction_comparison(
    eval_results,
    title=f"Holdout Evaluation ({split_year}–{max_yr})"
)
st.plotly_chart(fig_eval, use_container_width=True)

# Performance Metrics Table
st.subheader("📋 Out-of-Sample Performance Benchmarking")
st.dataframe(eval_results["comparison_table"], use_container_width=True, hide_index=True)

st.markdown("---")

# Future Projection Section
st.subheader(f"🔭 Short-Term Baseline Projection (Next {horizon_months} Months)")
st.caption(
    "Extrapolation of the modern era (1976–Present) trend with empirical 95% forecast intervals "
    "(±1.96 × residual standard deviation)."
)

proj_df = generate_future_projection(df, horizon_months=horizon_months)

fig_proj = go.Figure()

# Recent actuals (last 5 years)
recent_actuals = df[df["Year"] >= (max_yr - 5)]
fig_proj.add_trace(
    go.Scatter(
        x=recent_actuals["Decimal_Year"],
        y=recent_actuals["Anomaly"],
        mode="lines+markers",
        name="Recent Actuals",
        line={"color": "#f8fafc", "width": 2.0},
    )
)

# Projected trend line
fig_proj.add_trace(
    go.Scatter(
        x=proj_df["Decimal_Year"],
        y=proj_df["Projected_Anomaly"],
        mode="lines",
        name="Projected Trend",
        line={"color": "#ef4444", "width": 2.5, "dash": "dash"},
    )
)

# Upper and Lower CI
fig_proj.add_trace(
    go.Scatter(
        x=proj_df["Decimal_Year"],
        y=proj_df["Upper_95_CI"],
        mode="lines",
        name="Upper 95% Bound",
        line={"color": "rgba(239, 68, 68, 0.2)", "width": 0},
        showlegend=False,
    )
)
fig_proj.add_trace(
    go.Scatter(
        x=proj_df["Decimal_Year"],
        y=proj_df["Lower_95_CI"],
        mode="lines",
        name="95% Empirical Bound",
        line={"color": "rgba(239, 68, 68, 0.2)", "width": 0},
        fill="tonexty",
        fillcolor="rgba(239, 68, 68, 0.15)",
    )
)

fig_proj.update_layout(
    paper_bgcolor="#1e293b",
    plot_bgcolor="#0f172a",
    font={"color": "#f8fafc"},
    xaxis={"gridcolor": "#334155", "title": "Year"},
    yaxis={"gridcolor": "#334155", "title": "Projected Anomaly (°C)"},
    height=480,
)
st.plotly_chart(fig_proj, use_container_width=True)

# Table of Projected Values
with st.expander("🔍 View Projected Monthly Values Table"):
    st.dataframe(proj_df, use_container_width=True, hide_index=True)
