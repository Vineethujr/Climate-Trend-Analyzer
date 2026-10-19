"""
Climate Trend Analyzer - Root Streamlit Entrypoint.

Registers the application pages stored in app/ for root-level execution.
Allows running with either:
    streamlit run app.py
or
    streamlit run app/Home.py
"""

import os
import sys

import streamlit as st

# Ensure root directory is on Python module search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PAGES_DIR = os.path.join(BASE_DIR, "app", "pages")
pages = [
    st.Page(os.path.join(BASE_DIR, "app", "Home.py"), title="Home", icon="🌍", default=True),
    st.Page(os.path.join(PAGES_DIR, "1_Climate_Overview.py"), title="Climate Overview", icon="🌡️"),
    st.Page(os.path.join(PAGES_DIR, "2_Trend_Analysis.py"), title="Trend Analysis", icon="📈"),
    st.Page(os.path.join(PAGES_DIR, "3_Seasonal_Patterns.py"), title="Seasonal Patterns", icon="🍂"),
    st.Page(os.path.join(PAGES_DIR, "4_Anomaly_Detection.py"), title="Anomaly Detection", icon="🚨"),
    st.Page(os.path.join(PAGES_DIR, "5_Predictive_Model.py"), title="Predictive Model", icon="🔮"),
    st.Page(os.path.join(PAGES_DIR, "6_Methodology_and_Data.py"), title="Methodology and Data", icon="📚"),
]

st.navigation(pages).run()
