#!/usr/bin/env python3
"""
Climate Trend Analyzer - Main Project Launcher and CLI Tool.

This entrypoint provides convenient commands to run the Streamlit dashboard,
execute the test suite, download/verify datasets, or print project diagnostics.

Usage:
    python main.py             # Launch Streamlit interactive dashboard
    python main.py --app       # Launch Streamlit dashboard
    python main.py --test      # Run pytest automated test suite
    python main.py --status    # Check environment and dataset status
"""

import sys
import os
import subprocess
import argparse

def get_base_dir() -> str:
    """Return the absolute path of the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))

def run_streamlit_app():
    """Launch the interactive Streamlit application."""
    base_dir = get_base_dir()
    app_path = os.path.join(base_dir, "app", "Home.py")
    if not os.path.exists(app_path):
        app_path = os.path.join(base_dir, "app.py")
    
    print("\n" + "="*60)
    print("🌍 Starting Climate Trend Analyzer Streamlit Application...")
    print(f"📍 Entrypoint: {app_path}")
    print("="*60 + "\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd)

def run_tests():
    """Execute the pytest automated test suite."""
    print("\n" + "="*60)
    print("🧪 Running Climate Trend Analyzer Test Suite...")
    print("="*60 + "\n")
    cmd = [sys.executable, "-m", "pytest", "-v", "tests/"]
    sys.exit(subprocess.run(cmd).returncode)

def show_status():
    """Display environment, dataset, and package diagnostics."""
    base_dir = get_base_dir()
    print("\n" + "="*60)
    print("📋 Climate Trend Analyzer - System Diagnostics")
    print("="*60)
    print(f"Python Runtime : {sys.version.split()[0]} ({sys.executable})")
    print(f"Project Root   : {base_dir}")
    
    data_dir = os.path.join(base_dir, "data", "raw")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"Data directory : {data_dir} ({len(files)} files)")
        for f in files:
            size_kb = os.path.getsize(os.path.join(data_dir, f)) / 1024
            print(f"  • {f} ({size_kb:.1f} KB)")
    else:
        print(f"Data directory : {data_dir} (Not yet created)")
    
    print("\nDependency Check:")
    deps = ["streamlit", "pandas", "numpy", "scipy", "statsmodels", "sklearn", "plotly", "requests", "pytest"]
    for dep in deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [✓] {dep:15s} : {ver}")
        except ImportError:
            print(f"  [✗] {dep:15s} : NOT FOUND")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Climate Trend Analyzer - Historical Climate Analytics & Dashboard."
    )
    parser.add_argument(
        "--app", action="store_true", help="Launch the Streamlit dashboard."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the automated test suite."
    )
    parser.add_argument(
        "--status", action="store_true", help="Print environment and dataset status."
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.status:
        show_status()
    else:
        run_streamlit_app()

if __name__ == "__main__":
    main()
