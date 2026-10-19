"""
Climate Trend Analyzer - Streamlit Application Entrypoint & Overview.
"""

import sys
import os
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset, get_available_domains
from src.preprocessing import compute_rolling_averages, adjust_baseline
from src.visualizations import plot_historical_series


st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for KPI cards and dark theme
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.8rem;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar Global Controls
    st.sidebar.title("🌍 Climate Controls")
    st.sidebar.markdown("---")

    domains = get_available_domains()
    selected_domain = st.sidebar.selectbox(
        "Select Geographic Domain",
        domains,
        index=0,
        help="Select spatial observation scope from NOAA NCEI historical tables."
    )

    baseline_choice = st.sidebar.selectbox(
        "Reference Baseline Period",
        [
            "1901-2000 (NOAA 20th Century Default)",
            "1951-1980 (NASA GISTEMP Climatology)",
            "1850-1900 (Pre-Industrial Baseline)",
        ],
        index=0,
        help="Climatological baseline normal against which departures are calculated."
    )

    force_refresh = st.sidebar.checkbox(
        "Force Online Data Refresh",
        value=False,
        help="Query NOAA NCEI servers directly instead of utilizing local cache."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Data Provenance**: Public historical monthly observations provided by NOAA NCEI "
        "(GHCNm v4 & ERSSTv5). 1850–Present."
    )

    # Load Data with caching
    with st.spinner("Loading historical climate observations..."):
        try:
            df_raw, metadata = load_dataset(selected_domain, force_refresh=force_refresh)
        except Exception as e:
            st.error(f"Error loading climate data: {e}")
            st.stop()

    # Apply Baseline Adjustment if selected
    df = df_raw.copy()
    offset_applied = 0.0
    if "1951-1980" in baseline_choice:
        df, offset_applied = adjust_baseline(df, 1951, 1980)
    elif "1850-1900" in baseline_choice:
        df, offset_applied = adjust_baseline(df, 1850, 1900)

    # Precompute rolling averages for display
    df = compute_rolling_averages(df, windows=[12, 60])

    # Store in session state for other pages
    st.session_state["df"] = df
    st.session_state["metadata"] = metadata
    st.session_state["selected_domain"] = selected_domain
    st.session_state["baseline_choice"] = baseline_choice
    st.session_state["offset_applied"] = offset_applied

    # Top Banner Header
    st.title("🌍 Climate Trend Analyzer")
    st.markdown(
        f"**Empirical Analysis of Historical Surface Temperature Observations** | "
        f"Domain: `{selected_domain}` | Baseline: `{baseline_choice.split(' ')[0]}`"
    )

    # Key Metrics KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)

    n_obs = len(df)
    start_yr = int(df["Year"].min())
    end_yr = int(df["Year"].max())
    mean_anom = float(df["Anomaly"].mean())
    max_row = df.loc[df["Anomaly"].idxmax()]
    min_row = df.loc[df["Anomaly"].idxmin()]
    latest_row = df.iloc[-1]

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Observation Span</div>
            <div class="metric-value">{start_yr}–{end_yr}</div>
            <div class="metric-sub">{n_obs:,} Monthly Records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Anomaly</div>
            <div class="metric-value">{mean_anom:+.2f} °C</div>
            <div class="metric-sub">Across full period</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Recorded</div>
            <div class="metric-value">{latest_row['Anomaly']:+.2f} °C</div>
            <div class="metric-sub">Date: {latest_row['Date']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Highest Anomaly</div>
            <div class="metric-value" style="color: #ef4444;">{max_row['Anomaly']:+.2f} °C</div>
            <div class="metric-sub">Record: {max_row['Date']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Lowest Anomaly</div>
            <div class="metric-value" style="color: #38bdf8;">{min_row['Anomaly']:+.2f} °C</div>
            <div class="metric-sub">Record: {min_row['Date']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical Time Series Chart
    st.subheader("📈 Historical Surface Temperature Departure Series")
    st.caption("Monthly temperature anomalies with trailing 12-month (1-year) and 60-month (5-year) smoothing filters.")
    fig = plot_historical_series(df, show_rolling=True, title=f"{selected_domain} Historical Temperature Departures")
    st.plotly_chart(fig, use_container_width=True)

    # Guided Analytical Navigation Cards
    st.subheader("🧭 Guided Analytical Modules")
    nav1, nav2, nav3, nav4 = st.columns(4)

    with nav1:
        st.markdown("#### 1. 🌡️ Climate Overview")
        st.write("Examine the full dataset distribution, annual summaries, and interactive date window filters.")
        st.page_link("pages/1_🌡️_Climate_Overview.py", label="Explore Overview", icon="🌡️")

    with nav2:
        st.markdown("#### 2. 📈 Trend Analysis")
        st.write("Quantify long-term warming rates (°C/decade) via OLS and Theil-Sen regression with 95% CIs.")
        st.page_link("pages/2_📈_Trend_Analysis.py", label="Analyze Trends", icon="📈")

    with nav3:
        st.markdown("#### 3. 🍂 Seasonal Patterns")
        st.write("Deconstruct monthly climatology, intra-annual shifts, and classical 4-component decomposition.")
        st.page_link("pages/3_🍂_Seasonal_Patterns.py", label="Inspect Seasonality", icon="🍂")

    with nav4:
        st.markdown("#### 4. 🚨 Anomaly Detection")
        st.write("Detect statistical outliers via Z-score, Tukey IQR fences, and unsupervised Isolation Forest.")
        st.page_link("pages/4_🚨_Anomaly_Detection.py", label="Detect Anomalies", icon="🚨")

    st.markdown("---")
    st.markdown(
        "Built with Python, Pandas, SciPy, Statsmodels, Scikit-learn, Plotly, and Streamlit. "
        "Source code and methodology documented under MIT License."
    )


if __name__ == "__main__":
    main()
