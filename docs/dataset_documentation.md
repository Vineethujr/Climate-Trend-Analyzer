# Dataset Documentation: Historical Climate Observations

## 1. Overview
The **Climate Trend Analyzer** uses real-world, peer-reviewed public climate observation datasets to conduct temperature trend estimation, seasonal decomposition, and statistical anomaly detection.

## 2. Selected Primary Dataset: NOAA NCEI Climate at a Glance (CAG)
* **Authoritative Source**: NOAA National Centers for Environmental Information (NCEI), Asheville, NC, USA.
* **Portal**: [https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/)
* **Access Protocol**: Public REST / CSV endpoint with documented header comments:
  * `# Title: Global Land and Ocean Average Temperature Departures`
  * `# Units: Degrees Celsius`
  * `# Base Period: 1901-2000`
* **Underlying Observation Systems**:
  * **GHCNm v4**: Global Historical Climatology Network Monthly dataset containing monthly surface air temperatures recorded at thousands of land meteorological stations globally, quality-controlled for inhomogeneities and station relocations (Menne et al., 2018).
  * **ERSSTv5**: Extended Reconstructed Sea Surface Temperature dataset version 5, combining in-situ sea surface observations from ships, buoys, and Argo floats with spatio-temporal statistical smoothing (Huang et al., 2020).

## 3. Data Dictionary & Variables

| Column Name | Data Type | Units | Description | Valid Physical Range |
| :--- | :--- | :--- | :--- | :--- |
| `Date` | Integer / String | `YYYYMM` | Year and Month of observation (e.g. `185001` = Jan 1850) | `185001` – `Present` |
| `Year` | Integer | Year | Extracted 4-digit Gregorian calendar year | `1850` – `Present` |
| `Month` | Integer | Month | Calendar month index (1 = Jan, ..., 12 = Dec) | `1` – `12` |
| `Decimal_Year`| Float | Fractional Year | Time coordinate for continuous regression ($Year + \frac{Month - 0.5}{12}$) | `1850.0` – `Present` |
| `Anomaly` | Float | °C | Temperature departure relative to the 1901–2000 climatological average | $[-5.0, +5.0]$ |
| `Rolling_12m` | Float | °C | 12-month trailing moving average | Dynamic |
| `Rolling_60m` | Float | °C | 60-month (5-year) trailing moving average | Dynamic |

## 4. Climatological Baseline Periods
An **anomaly** represents the departure of observed temperature from a reference long-term average (climatological normal):
$$\text{Anomaly}_t = T_t - \overline{T}_{\text{baseline}}$$

* **NOAA 20th Century Baseline (1901–2000)**: Default reference period utilized in NOAA CAG tables.
* **NASA GISTEMP Baseline (1951–1980)**: Frequently used in satellite-era climate literature.
* **Pre-Industrial Baseline (1850–1900)**: Often utilized in IPCC international policy contexts to evaluate warming relative to pre-industrial conditions.
* The application provides a dynamic re-baselining toggle to convert between baseline definitions transparently.

## 5. Usage & Citation Guidelines
* **Attribution**: Cite NOAA National Centers for Environmental Information, State of the Climate: Global Climate Report.
* **License**: Open government data, free for public scientific, educational, and commercial application under US federal public domain regulations.
