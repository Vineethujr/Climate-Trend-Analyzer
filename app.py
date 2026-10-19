"""
Climate Trend Analyzer - Root Streamlit Entrypoint.

Delegates execution to the main application in app/Home.py.
Allows running with either:
    streamlit run app.py
or
    streamlit run app/Home.py
"""

import os
import sys

# Ensure root directory is on Python module search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import and execute the main Home page module
from app.Home import main

if __name__ == "__main__":
    main()
