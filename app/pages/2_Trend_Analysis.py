"""
Climate Trend Analyzer - Page 2: Statistical Trend Analysis.
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
from src.trend_analysis import (
    compute_ols_trend,
    compute_theil_sen_trend,
    compute_mann_kendall_test,
    compute_autocorrelation_diagnostics,
    compare_era_trends,
)
from src.visualizations import plot_trend_regression


st.set_page_config(page_title="Trend Analysis | Climate Trend Analyzer", page_icon="📈", layout="wide")

st.title("📈 Statistical Trend & Warming Rate Analysis")
st.markdown("Quantify long-term temperature changes using parametric (OLS) and non-parametric (Theil-Sen, Mann-Kendall) models.")

if "df" not in st.session_state:
    df, _ = load_dataset("Global (Land & Ocean)")
    st.session_state["df"] = df
else:
    df = st.session_state["df"]

selected_domain = st.session_state.get("selected_domain", "Global (Land & Ocean)")

min_yr = int(df["Year"].min())
max_yr = int(df["Year"].max())

st.sidebar.header("Trend Parameters")
selected_range = st.sidebar.slider(
    "Analysis Window",
    min_value=min_yr,
    max_value=max_yr,
    value=(min_yr, max_yr),
    step=1,
    help="Define the temporal window for regression fitting."
)

active_df = filter_by_date_range(df, selected_range[0], selected_range[1])

if len(active_df) < 24:
    st.warning("Please select a date range containing at least 24 monthly observations for reliable trend analysis.")
    st.stop()

# Compute Statistical Models
ols_res = compute_ols_trend(active_df)
ts_res = compute_theil_sen_trend(active_df)
mk_res = compute_mann_kendall_test(active_df)
ac_res = compute_autocorrelation_diagnostics(active_df)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "OLS Warming Rate",
        f"+{ols_res['rate_per_decade']:.3f} °C / dec",
        help="Ordinary Least Squares linear regression slope expressed per decade."
    )
with col2:
    st.metric(
        "95% Confidence Interval",
        f"[{ols_res['ci_decade_lower']:+.3f}, {ols_res['ci_decade_upper']:+.3f}]",
        help="Student's t two-tailed 95% confidence interval for the decadal rate."
    )
with col3:
    st.metric(
        "Theil-Sen Robust Rate",
        f"+{ts_res['theil_sen_rate_per_decade']:.3f} °C / dec",
        help="Median pairwise slope (non-parametric, robust to extreme outliers)."
    )
with col4:
    st.metric(
        "Mann-Kendall Tau",
        f"{mk_res['kendall_tau']:+.3f} (p < 0.0001)" if mk_res['p_value'] < 0.0001 else f"{mk_res['kendall_tau']:+.3f}",
        help="Monotonic trend rank correlation coefficient."
    )

st.markdown("---")

# Main Regression Chart
st.subheader("📉 Long-Term Linear Trend Estimation")
fig_trend = plot_trend_regression(
    ols_res,
    title=f"{selected_domain} Linear Trend ({selected_range[0]}–{selected_range[1]})"
)
st.plotly_chart(fig_trend, use_container_width=True)

# Landmark Era Comparison
st.subheader("🏛️ Historical Era Warming Rate Comparison")
st.caption("Contrasting warming rates across widely referenced climatological periods.")
era_df = compare_era_trends(df)
st.dataframe(era_df, use_container_width=True, hide_index=True)

# Autocorrelation & Statistical Limitations Panel
st.markdown("---")
st.subheader("🔍 Time Series Diagnostics & Serial Correlation")

col_ac1, col_ac2 = st.columns([1, 1])

with col_ac1:
    st.markdown("#### Autocorrelation Metrics")
    st.write(f"• **Lag-1 Autocorrelation ($r_1$)**: `{ac_res['lag_1_autocorrelation']:+.3f}`")
    st.write(f"• **Durbin-Watson Statistic ($d$)**: `{ac_res['durbin_watson_stat']:.3f}`")
    st.write(f"• **Nominal Sample Size ($N$)**: `{ac_res['nominal_sample_size']:,}` months")
    st.write(f"• **Effective Degrees of Freedom ($N_{{eff}}$)**: `{ac_res['effective_sample_size']:,}` months")

with col_ac2:
    st.markdown("#### Scientific & Statistical Guidance")
    st.warning(
        "**Serial Correlation Notice**: Geophysical and meteorological time series exhibit strong positive "
        "autocorrelation ($r_1 > 0$), meaning consecutive months are correlated. "
        "Standard OLS assumes independent observations; positive autocorrelation shrinks nominal standard errors, "
        "making conventional $p$-values overly optimistic. For this reason, we report both the non-parametric "
        "Theil-Sen estimator and Mann-Kendall rank test."
    )
    st.info(
        "**Causation Disclaimer**: A positive statistical trend measures persistent upward temperature departure over "
        "time. It documents empirical change but does not, in isolation, prove specific physical causation without "
        "coupled atmosphere-ocean radiative physics models."
    )
