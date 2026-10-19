"""
Climate Trend Analyzer - Page 6: Methodology, Data Provenance & Interview Guide.
"""

import streamlit as st


st.set_page_config(page_title="Methodology & Data | Climate Trend Analyzer", page_icon="📖", layout="wide")

st.title("📖 Analytical Methodology, Provenance & Technical Guide")
st.markdown("Detailed mathematical formulas, scientific references, data citations, and technical interview talking points.")

tab1, tab2, tab3 = st.tabs(["📐 Statistical Formulations", "🛰️ Dataset Provenance", "💼 Technical Interview Guide"])

with tab1:
    st.subheader("1. Ordinary Least Squares (OLS) Linear Regression")
    st.markdown(r"""
    The baseline parametric trend model estimates the linear warming slope $\beta_1$:
    $$y_i = \beta_0 + \beta_1 t_i + \varepsilon_i$$
    where $t_i$ is the continuous time coordinate in decimal years ($t_i = \text{Year}_i + \frac{\text{Month}_i - 0.5}{12}$) and $y_i$ is the temperature anomaly in °C.
    
    The decadal rate is computed as:
    $$\text{Rate per Decade} = \beta_1 \times 10 \quad (^\circ\text{C}/\text{decade})$$
    
    The 95% two-tailed confidence interval is derived from Student's $t$-distribution with $n - 2$ degrees of freedom:
    $$\text{CI}_{95\%} = \left(\beta_1 \pm t_{0.975, n-2} \cdot \text{SE}(\beta_1)\right) \times 10$$
    """)

    st.subheader("2. Theil-Sen Robust Estimator")
    st.markdown(r"""
    To guard against leverage outliers and non-normal residuals, the Theil-Sen slope is computed as the median of all slopes between pairs of points:
    $$\beta_{\text{TS}} = \text{median}\left\{ \frac{y_j - y_i}{t_j - t_i} : 1 \le i < j \le n \right\}$$
    """)

    st.subheader("3. Mann-Kendall Monotonic Trend Test")
    st.markdown(r"""
    A rank-based non-parametric test evaluating whether a monotonic upward or downward trend exists without assuming normality:
    $$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(y_j - y_k)$$
    Kendall's rank correlation coefficient $\tau$ evaluates the relative concordance of the pairs:
    $$\tau = \frac{2 S}{n(n - 1)}$$
    """)

    st.subheader("4. Serial Autocorrelation & Degrees of Freedom")
    st.markdown(r"""
    Geophysical time series frequently exhibit positive lag-1 autocorrelation:
    $$r_1 = \frac{\sum_{i=1}^{n-1} (y_i - \bar{y})(y_{i+1} - \bar{y})}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
    The Durbin-Watson statistic evaluates first-order autoregressive errors:
    $$d \approx 2(1 - r_1)$$
    Under positive autocorrelation ($r_1 > 0$), effective degrees of freedom decrease:
    $$N_{\text{eff}} = N \cdot \frac{1 - r_1}{1 + r_1}$$
    """)

    st.subheader("5. Classical Additive Decomposition")
    st.markdown(r"""
    $$Y_t = T_t + S_t + R_t$$
    * **$T_t$ (Trend)**: Centered 12-month moving average filter.
    * **$S_t$ (Seasonal)**: Monthly mean of detrended departures ($Y_t - T_t$).
    * **$R_t$ (Residual)**: Irregular unexplained component $Y_t - T_t - S_t$.
    """)

with tab2:
    st.subheader("Dataset Provenance & Observation Networks")
    st.markdown("""
    ### National Oceanic and Atmospheric Administration (NOAA NCEI)
    * **Program**: Climate at a Glance (CAG) Global Surface Temperature Analysis.
    * **Land Observations**: Global Historical Climatology Network Monthly (GHCNm version 4), incorporating over 25,000 weather stations worldwide with automated pairwise homogenization algorithms (Menne et al., 2018).
    * **Ocean Observations**: Extended Reconstructed Sea Surface Temperature version 5 (ERSSTv5), synthesizing ship logs, drifting buoys, and Argo profiling floats (Huang et al., 2020).
    * **Baseline Period**: 1901–2000 (20th Century Mean).
    * **Units**: Degrees Celsius (°C).
    * **Licensing**: US Federal Public Domain data (open government work under 17 U.S.C. § 105).
    """)

with tab3:
    st.subheader("💼 Five Key Technical Interview Questions & Answers")

    with st.expander("Q1: Why do we analyze temperature anomalies rather than raw absolute temperatures?"):
        st.markdown("""
        **Answer**:
        Absolute surface temperatures vary drastically over short geographic distances due to elevation, microclimates, and topography (e.g., a valley weather station vs. a mountain station 10 km away may differ by 10°C). 
        
        However, **temperature anomalies** (departures from a station's local long-term average) are highly correlated over distances exceeding 1,000 km. Utilizing anomalies allows spatial interpolation, eliminates elevation bias, and enables meaningful global averaging.
        """)

    with st.expander("Q2: How does serial autocorrelation affect ordinary least squares (OLS) trend analysis in climate data?"):
        st.markdown("""
        **Answer**:
        Standard OLS regression assumes that residual errors $\\varepsilon_i$ are independent and identically distributed (i.i.d.). Climate time series exhibit significant positive serial correlation ($r_1 > 0$) because heat storage in the oceans and atmosphere provides thermal inertia. 

        Positive autocorrelation causes OLS standard errors to be underestimated, artificially inflating $t$-statistics and yielding overly optimistic $p$-values. To counter this, we compute the Durbin-Watson statistic, calculate the effective sample size $N_{\\text{eff}}$, and cross-validate with the non-parametric **Theil-Sen estimator** and **Mann-Kendall test**.
        """)

    with st.expander("Q3: Why is random k-fold cross-validation invalid for evaluating time-series forecasting models?"):
        st.markdown("""
        **Answer**:
        Random shuffling in cross-validation breaks temporal dependency and leaks future information into the training set (lookahead bias). In climate data with strong autocorrelation, a randomly selected test observation is adjacent to training observations, artificially boosting test accuracy. 

        We enforce **strict chronological train/test splitting** (e.g., training exclusively on pre-2015 data and evaluating on 2015–present) to assess true out-of-sample generalization.
        """)

    with st.expander("Q4: What is the difference between a statistical temperature anomaly and an extreme climate event?"):
        st.markdown("""
        **Answer**:
        In statistics, an anomaly is merely an outlier that deviates significantly from the empirical distribution (e.g., $|z| > 2.5\\sigma$). 

        An **extreme climate event** (such as a heatwave, drought, or cyclone) is a complex meteorological phenomenon defined by duration, spatial extent, physical impacts, and local climatology. A single statistical outlier in a global average series cannot be automatically attributed to a specific disaster without localized meteorological attribution studies.
        """)

    with st.expander("Q5: How does this project structure separate real empirical observations from test data?"):
        st.markdown("""
        **Answer**:
        Real public observations from NOAA are stored in `data/raw/` with verified provenance. Synthetic datasets used for unit testing are strictly quarantined in `tests/fixtures/synthetic_series.py` and carry an explicit metadata flag (`is_synthetic: True`). 

        This ensures algorithmic test fixtures (with known slopes and injected spikes) never contaminate empirical climate analyses.
        """)
