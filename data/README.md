# Climate Data Directory

This directory houses historical climate observation datasets utilized by the **Climate Trend Analyzer**.

* `raw/`: Unmodified primary historical climate observation series directly ingested from NOAA NCEI (1850–Present). Real observations are processed dynamically in-memory.

## Primary Datasets

### 1. NOAA NCEI Climate at a Glance (CAG) Surface Temperature Departures
* **Source**: National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information.
* **Citation**: Huang et al. (2020), Extended Reconstructed Sea Surface Temperature version 5 (ERSSTv5); Menne et al. (2018), Global Historical Climatology Network - Monthly (GHCNm) Version 4.
* **Temporal Coverage**: Monthly from January 1850 to Present.
* **Reference Baseline**: 1901–2000 (20th-century mean).
* **Units**: Degrees Celsius (°C).
* **Included Domains**:
  * `globe_land_ocean.csv`: Global surface temperature departures (land + ocean).
  * `nhem_land_ocean.csv`: Northern Hemisphere departures.
  * `shem_land_ocean.csv`: Southern Hemisphere departures.
  * `globe_land.csv`: Global land-only surface temperature departures.
  * `globe_ocean.csv`: Global ocean-only sea surface temperature departures.

### 2. Secondary Cross-Validation Dataset: NASA GISTEMP v4
* **Source**: NASA Goddard Institute for Space Studies (GISS).
* **Citation**: Lenssen et al. (2019), Improvements in the GISTEMP Uncertainty Model.
* **Reference Baseline**: 1951–1980 mean.
* **Units**: Degrees Celsius (°C).

## Data Integrity & Licensing
* Public domain / open government data (US Government works under 17 U.S.C. § 105).
* Real observations are kept separate from synthetic testing data. Synthetic series used in unit tests reside exclusively in `tests/fixtures/synthetic_series.py` and are explicitly tagged with `is_synthetic: True`.
