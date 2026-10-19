# Exploratory Data Analysis & Notebooks

This directory contains standalone exploratory scripts and reproducible Jupyter notebooks used for initial research, data validation, and algorithm verification.

## Notebook Structure
* `01_exploratory_data_analysis.ipynb`: Systematic step-by-step EDA of NOAA and GISTEMP historical temperature series, including stationarity tests, seasonal cycles, and trend validations.

## Reproducibility Guidelines
To execute the notebooks locally:
```bash
# Ensure your virtual environment is active
.venv\Scripts\activate   # On Windows

# Install ipykernel if not present
pip install ipykernel

# Launch Jupyter
jupyter lab notebooks/
```
