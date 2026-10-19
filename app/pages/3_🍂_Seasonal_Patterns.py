"""
Climate Trend Analyzer - Page 3: Seasonal Patterns & Decomposition.
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
from src.seasonal_analysis import (
    compute_monthly_climatology,
    compute_seasonal_summary,
    decompose_time_series,
    compare_decadal_seasonality,
)
from src.visualizations import (
    plot_monthly_climatology,
    plot_seasonal_comparison,
    plot_decomposition,
)
import plotly.express as px


st.set_page_config(page_title="Seasonal Patterns | Climate Trend Analyzer", page_icon="🍂", layout="wide")

st.title("🍂 Seasonal Dynamics & Climatological Decomposition")
st.markdown("Analyze intra-annual monthly distributions, meteorological seasonal trends, and 4-component time-series decomposition.")

if "df" not in st.session_state:
    df, _ = load_dataset("Global (Land & Ocean)")
    st.session_state["df"] = df
else:
    df = st.session_state["df"]

selected_domain = st.session_state.get("selected_domain", "Global (Land & Ocean)")

min_yr = int(df["Year"].min())
max_yr = int(df["Year"].max())

st.sidebar.header("Filter Horizon")
selected_range = st.sidebar.slider(
    "Analysis Window",
    min_value=min_yr,
    max_value=max_yr,
    value=(min_yr, max_yr),
    step=1,
)

active_df = filter_by_date_range(df, selected_range[0], selected_range[1])

if len(active_df) < 24:
    st.warning("Please select a range containing at least 24 months for seasonal analysis.")
    st.stop()

# 1. Monthly Climatology Distribution
st.subheader("📅 Climatological Monthly Profile")
st.caption("Mean departure and 1-sigma standard deviation spread for each calendar month.")
monthly_df = compute_monthly_climatology(active_df)

col_m1, col_m2 = st.columns([3, 2])
with col_m1:
    fig_month = plot_monthly_climatology(
        monthly_df,
        title=f"Monthly Anomaly Profile ({selected_range[0]}–{selected_range[1]})"
    )
    st.plotly_chart(fig_month, use_container_width=True)

with col_m2:
    st.markdown("#### Monthly Statistical Distribution")
    st.dataframe(
        monthly_df[["Month_Name", "Mean", "Std", "Median", "Min", "Max"]].rename(columns={
            "Month_Name": "Month",
            "Mean": "Mean (°C)",
            "Std": "Std (°C)",
            "Median": "Median (°C)",
        }),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

st.markdown("---")

# 2. Meteorological Seasonality & Differential Warming
st.subheader("❄️ Meteorological Seasonal Trends")
st.caption("Evaluating whether specific seasons are experiencing accelerated warming rates.")

overall_season, season_trends = compute_seasonal_summary(active_df)

col_s1, col_s2 = st.columns([3, 2])
with col_s1:
    if not season_trends.empty:
        fig_season = plot_seasonal_comparison(season_trends)
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.info("Insufficient observations per season to compute individual decadal slopes.")

with col_s2:
    st.markdown("#### Decadal Warming by Season")
    if not season_trends.empty:
        st.dataframe(
            season_trends[["Season", "Warming Rate (°C/Decade)", "95% CI Lower", "95% CI Upper", "R-Squared"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No seasonal trends available.")

st.markdown("---")

# 3. Additive Time Series Decomposition
st.subheader("🧩 4-Component Time Series Decomposition")
st.caption("Classical additive decomposition: Observed (Y) = Long-Term Trend (T) + Seasonal Cycle (S) + Irregular Residual (R).")

try:
    decomp_df = decompose_time_series(active_df)
    fig_decomp = plot_decomposition(decomp_df)
    st.plotly_chart(fig_decomp, use_container_width=True)
except Exception as e:
    st.warning(f"Unable to run decomposition: {e}")

st.markdown("---")

# 4. Decadal Seasonal Shift Matrix
st.subheader("📊 Inter-Decadal Seasonal Shift Matrix")
st.caption("Contrasting monthly mean anomalies across 4 historical epochs to identify seasonal warming shifts.")
epoch_pivot = compare_decadal_seasonality(df)
st.dataframe(epoch_pivot, use_container_width=True)
