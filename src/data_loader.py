"""
Climate Trend Analyzer - Data Ingestion & Loading Module.

Responsible for:
- Fetching and reading public historical climate datasets (NOAA NCEI Climate at a Glance).
- Managing local caching in data/raw/ to enable offline execution.
- Parsing metadata headers (# Title, # Units, # Base Period).
- Formatting date representations (Year, Month, Decimal Year).
"""

import os
import io
from typing import Dict, List, Tuple, Optional
import requests
import pandas as pd


# Canonical NOAA NCEI Climate at a Glance endpoints and local filenames
DOMAIN_MAPPING: Dict[str, Dict[str, str]] = {
    "Global (Land & Ocean)": {
        "filename": "globe_land_ocean.csv",
        "endpoint": "globe/land_ocean/1/0/1850-2026.csv",
        "description": "Global combined land surface air and sea surface temperature departures.",
    },
    "Northern Hemisphere (Land & Ocean)": {
        "filename": "nhem_land_ocean.csv",
        "endpoint": "nhem/land_ocean/1/0/1850-2026.csv",
        "description": "Northern Hemisphere land and ocean temperature departures (0°N - 90°N).",
    },
    "Southern Hemisphere (Land & Ocean)": {
        "filename": "shem_land_ocean.csv",
        "endpoint": "shem/land_ocean/1/0/1850-2026.csv",
        "description": "Southern Hemisphere land and ocean temperature departures (0°S - 90°S).",
    },
    "Global Land Only": {
        "filename": "globe_land.csv",
        "endpoint": "globe/land/1/0/1850-2026.csv",
        "description": "Global meteorological station land surface air temperature departures.",
    },
    "Global Ocean Only": {
        "filename": "globe_ocean.csv",
        "endpoint": "globe/ocean/1/0/1850-2026.csv",
        "description": "Global sea surface temperature departures from ship, buoy, and Argo observations.",
    },
}

NOAA_BASE_URL = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series"
)


def get_default_data_dir() -> str:
    """Return the absolute path to the local data/raw directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    return os.path.join(project_root, "data", "raw")


def get_available_domains() -> List[str]:
    """Return a list of supported climate geographic domains."""
    return list(DOMAIN_MAPPING.keys())


def parse_noaa_csv(raw_text_or_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parse NOAA Climate at a Glance CSV content or file.
    
    Extracts header comments containing Title, Units, and Base Period,
    and returns a structured DataFrame and a metadata dictionary.

    Args:
        raw_text_or_path: File path or raw CSV string.

    Returns:
        Tuple of (pd.DataFrame, metadata_dict).
    """
    metadata: Dict[str, str] = {
        "title": "Historical Climate Temperature Departures",
        "units": "Degrees Celsius",
        "base_period": "1901-2000",
        "source": "NOAA NCEI Climate at a Glance",
    }

    if os.path.exists(raw_text_or_path):
        with open(raw_text_or_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = raw_text_or_path.strip().splitlines(keepends=True)

    header_lines = []
    data_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            header_lines.append(stripped)
            # Parse key-value pairs from comment lines
            if "Title:" in stripped:
                metadata["title"] = stripped.split("Title:", 1)[1].strip().strip('"')
            elif "Units:" in stripped:
                metadata["units"] = stripped.split("Units:", 1)[1].strip().strip('"')
            elif "Base Period:" in stripped:
                metadata["base_period"] = stripped.split("Base Period:", 1)[1].strip().strip('"')
        elif stripped:
            data_lines.append(line)

    if not data_lines:
        raise ValueError("No data rows found in the provided NOAA CSV content.")

    csv_data = "".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_data))

    # Normalize column names
    col_mapping = {}
    for col in df.columns:
        c_clean = col.strip().lower()
        if "date" in c_clean:
            col_mapping[col] = "Date"
        elif "departure" in c_clean or "value" in c_clean or "mean" in c_clean or "anomaly" in c_clean:
            col_mapping[col] = "Anomaly"

    df = df.rename(columns=col_mapping)

    if "Date" not in df.columns or "Anomaly" not in df.columns:
        raise ValueError(
            f"Expected 'Date' and 'Anomaly/Departure' columns, found: {list(df.columns)}"
        )

    # Process Date to components
    df["Date"] = df["Date"].astype(str).str.strip()
    df["Year"] = df["Date"].str.slice(0, 4).astype(int)
    df["Month"] = df["Date"].str.slice(4, 6).astype(int)
    
    # Continuous decimal year for regression: Year + (Month - 0.5) / 12
    df["Decimal_Year"] = df["Year"] + (df["Month"] - 0.5) / 12.0
    df["Anomaly"] = pd.to_numeric(df["Anomaly"], errors="coerce")

    # Chronological sort
    df = df.sort_values(by=["Year", "Month"]).reset_index(drop=True)

    # Order standard columns
    standard_cols = ["Date", "Year", "Month", "Decimal_Year", "Anomaly"]
    other_cols = [c for c in df.columns if c not in standard_cols]
    df = df[standard_cols + other_cols]

    return df, metadata


def load_dataset(
    domain_name: str = "Global (Land & Ocean)",
    data_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load historical climate data for a given domain from local cache or remote endpoint.

    Args:
        domain_name: Domain name from get_available_domains().
        data_dir: Path to directory holding raw CSV files (defaults to data/raw).
        force_refresh: If True, forces redownloading from NOAA servers.

    Returns:
        Tuple of (clean DataFrame, metadata dictionary).
    """
    if domain_name not in DOMAIN_MAPPING:
        raise ValueError(
            f"Unknown domain: '{domain_name}'. Available: {get_available_domains()}"
        )

    domain_info = DOMAIN_MAPPING[domain_name]
    target_dir = data_dir or get_default_data_dir()
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, domain_info["filename"])

    content = None

    if not force_refresh and os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = None

    if content is None or force_refresh:
        url = f"{NOAA_BASE_URL}/{domain_info['endpoint']}"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Climate Trend Analyzer Researcher)"}
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200 and len(resp.text) > 200:
                content = resp.text
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                # If remote fails and local file exists, use local file
                if os.path.exists(local_path):
                    with open(local_path, "r", encoding="utf-8") as f:
                        content = f.read()
                else:
                    raise RuntimeError(
                        f"Failed to fetch data from NOAA ({resp.status_code}) and no local cache exists at {local_path}."
                    )
        except Exception as e:
            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                raise RuntimeError(
                    f"Unable to load climate data for '{domain_name}': {e}"
                )

    df, metadata = parse_noaa_csv(content)
    metadata["domain"] = domain_name
    metadata["description"] = domain_info["description"]
    metadata["is_synthetic"] = False

    return df, metadata
