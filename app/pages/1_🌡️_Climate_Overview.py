"""
Climate Trend Analyzer - Page 1: Climate Overview.
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset
from src.preprocessing import aggregate_annual, aggregate_decadal, filter_by_date_range, compute_rolling_averages
from src.visualizations import plot_historical_series, plot_annual_bars
import plotly.express as px


st.set_page_config(page_title="Climate Overview | Climate Trend Analyzer", page_icon="🌡️", layout="wide")

st.title("🌡️ Climate Overview Dashboard")
st.markdown("Explore overall statistical distributions, annual departures, and filtered climate time series.")

# Retrieve dataset from session state or load default
if "df" not in st.session_state:
    df, meta = load_dataset("Global (Land & Ocean)")
    df = compute_rolling_averages(df)
    st.session_state["df"] = df
    st.session_state["selected_domain"] = "Global (Land & Ocean)"
else:
    df = st.session_state["df"]

selected_domain = st.session_state.get("selected_domain", "Global (Land & Ocean)")

min_yr = int(df["Year"].min())
max_yr = int(df["Year"].max())

# Filter Section
st.sidebar.header("Filter Time Horizon")
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=min_yr,
    max_value=max_yr,
    value=(min_yr, max_yr),
    step=1,
    help="Adjust temporal analysis bounds dynamically."
)

filtered_df = filter_by_date_range(df, selected_years[0], selected_years[1])
filtered_df = compute_rolling_averages(filtered_df)

# Key Summary Metrics for selected filter
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Filtered Records", f"{len(filtered_df):,}")
with col2:
    st.metric("Mean Anomaly", f"{filtered_df['Anomaly'].mean():+.3f} °C")
with col3:
    st.metric("Std Deviation", f"{filtered_df['Anomaly'].std():.3f} °C")
with col4:
    st.metric("Max Anomaly", f"{filtered_df['Anomaly'].max():+.2f} °C")
with col5:
    st.metric("Min Anomaly", f"{filtered_df['Anomaly'].min():+.2f} °C")

st.markdown("---")

# Main Interactive Chart
st.subheader("📊 Time Series Departure with Moving Averages")
fig_series = plot_historical_series(
    filtered_df,
    show_rolling=True,
    title=f"{selected_domain} ({selected_years[0]}–{selected_years[1]})"
)
st.plotly_chart(fig_series, use_container_width=True)

# Annual Departures & Distribution
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📅 Annual Mean Temperature Departures")
    st.caption("Complete 12-month calendar averages colored by departure (Red = Above normal, Blue = Below normal).")
    annual_df = aggregate_annual(filtered_df, min_months=12)
    fig_annual = plot_annual_bars(
        annual_df,
        title=f"Annual Mean Departures ({selected_years[0]}–{selected_years[1]})"
    )
    st.plotly_chart(fig_annual, use_container_width=True)

with col_right:
    st.subheader("📈 Anomaly Distribution")
    st.caption("Empirical density and spread of monthly departures.")
    fig_hist = px.histogram(
        filtered_df,
        x="Anomaly",
        nbins=40,
        marginal="box",
        color_discrete_sequence=["#38bdf8"],
        title="Anomaly Frequency Distribution",
    )
    fig_hist.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font={"color": "#f8fafc"},
        xaxis={"gridcolor": "#334155", "title": "Anomaly (°C)"},
        yaxis={"gridcolor": "#334155", "title": "Observation Count"},
        height=480,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Decadal Summary Table & Bar Chart
st.subheader("🏛️ Decadal Climatological Epochs")
decadal_df = aggregate_decadal(filtered_df)
col_dec1, col_dec2 = st.columns([3, 2])

with col_dec1:
    fig_dec = px.bar(
        decadal_df,
        x="Decade_Label",
        y="Decade_Mean",
        color="Decade_Mean",
        color_continuous_scale="RdBu_r",
        title="Mean Anomaly by 10-Year Climatological Decade",
    )
    fig_dec.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font={"color": "#f8fafc"},
        xaxis={"gridcolor": "#334155", "title": "Decade"},
        yaxis={"gridcolor": "#334155", "title": "Decade Mean (°C)"},
        height=380,
    )
    st.plotly_chart(fig_dec, use_container_width=True)

with col_dec2:
    st.dataframe(
        decadal_df.rename(columns={
            "Decade_Label": "Decade",
            "Decade_Mean": "Mean (°C)",
            "Std_Dev": "Std Dev (°C)",
            "Count": "Months",
        }),
        use_container_width=True,
        height=360,
    )

# Export Data Section
st.markdown("---")
st.subheader("📥 Export Filtered Data")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    csv_monthly = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Monthly Observations (CSV)",
        data=csv_monthly,
        file_name=f"climate_monthly_{selected_years[0]}_{selected_years[1]}.csv",
        mime="text/csv",
    )

with col_exp2:
    csv_annual = annual_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Annual Climatological Summaries (CSV)",
        data=csv_annual,
        file_name=f"climate_annual_{selected_years[0]}_{selected_years[1]}.csv",
        mime="text/csv",
    )
