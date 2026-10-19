# Methodology & Statistical Formulations

This document provides complete mathematical and statistical definitions for all analytical procedures implemented in the **Climate Trend Analyzer**.

---

## 1. Data Cleaning and Quality Validation
* **Date Parsing**: Date strings formatted as `YYYYMM` are parsed into integer year and month components, accompanied by a continuous decimal time index:
  $$t_i = \text{Year}_i + \frac{\text{Month}_i - 0.5}{12}$$
* **Domain Range Verification**: Observations outside $[-5.0^\circ\text{C}, +5.0^\circ\text{C}]$ are flagged as physically aberrant for global/hemispheric monthly anomalies.
* **Missing Value Policy**: Missing entries are flagged rather than filled with arbitrary zeros. Incomplete years are excluded from annual aggregations to avoid seasonal skewing.

---

## 2. Long-Term Trend Estimation

### Ordinary Least Squares (OLS) Linear Regression
The primary linear trend model is defined as:
$$y_i = \beta_0 + \beta_1 t_i + \varepsilon_i$$
where $y_i$ is the temperature anomaly at continuous time $t_i$ (in decimal years).
The slope $\beta_1$ is converted to climatological decadal warming rate:
$$\text{Decadal Rate} = \beta_1 \times 10 \quad (^\circ\text{C} / \text{decade})$$

The 95% confidence interval for the decadal rate is computed via:
$$\text{CI}_{95\%} = \left(\beta_1 \pm t_{0.975, n-2} \cdot \text{SE}(\beta_1)\right) \times 10$$

### Theil-Sen Robust Estimator
To guard against potential leverage points and non-Gaussian error distributions, the non-parametric Theil-Sen slope is computed as the median of all pairwise slopes:
$$\beta_{\text{TS}} = \text{median}\left\{ \frac{y_j - y_i}{t_j - t_i} : 1 \le i < j \le n \right\}$$

### Mann-Kendall Trend Test
A rank-based non-parametric hypothesis test evaluating whether a monotonic upward or downward trend exists without assuming normality:
$$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(y_j - y_k)$$
where $\text{sgn}(\theta)$ evaluates to $+1$ if $\theta > 0$, $0$ if $\theta = 0$, and $-1$ if $\theta < 0$.
The normalized test statistic $Z_{\text{MK}}$ and two-tailed $p$-value are computed to test the null hypothesis $H_0: \text{no trend}$.

### Autocorrelation & Serial Correlation Limitations
Geophysical time series typically exhibit positive lag-1 autocorrelation:
$$r_1 = \frac{\sum_{i=1}^{n-1} (y_i - \bar{y})(y_{i+1} - \bar{y})}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
Positive serial correlation reduces the effective degrees of freedom:
$$n_{\text{eff}} = n \frac{1 - r_1}{1 + r_1}$$
The Durbin-Watson statistic $d \approx 2(1 - r_1)$ is reported. The application explicitly documents that uncorrected OLS standard errors underestimate variance under positive serial correlation.

---

## 3. Seasonal & Intra-Annual Dynamics

### Meteorological Seasonal Aggregations
Standard meteorological seasons are defined as:
* **DJF**: December, January, February (Boreal Winter / Austral Summer)
* **MAM**: March, April, May (Boreal Spring / Austral Autumn)
* **JJA**: June, July, August (Boreal Summer / Austral Winter)
* **SON**: September, October, November (Boreal Autumn / Austral Spring)

### Classical Additive Decomposition
For an additive time series structure:
$$Y_t = T_t + S_t + R_t$$
* $T_t$: Long-term trend component extracted via centered moving average (12-month filter).
* $S_t$: Seasonal index computed by averaging detrended values ($Y_t - T_t$) for each calendar month across years.
* $R_t$: Residual irregular component $Y_t - T_t - S_t$.

---

## 4. Statistical Anomaly Detection

### Method 1: Standardized Deviation (Z-Score)
Identifies observations that deviate unusually from the sample mean:
$$z_t = \frac{x_t - \mu}{\sigma}$$
An observation is flagged if $|z_t| > \theta$ (selectable threshold $\theta \in \{2.0, 2.5, 3.0\}$).

### Method 2: Tukey's Interquartile Range (IQR) Fences
Non-parametric detection using the 25th percentile ($Q_1$) and 75th percentile ($Q_3$):
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

### Method 3: Isolation Forest
An unsupervised tree-based ensemble that isolates anomalous observations by randomly partitioning feature dimensions. Outliers require fewer recursive splits to be isolated.
* Features: `[Anomaly, Rolling_12m_Residual, Month_Sine, Month_Cosine]`.

### Scientific Disclaimer on Anomalies
A statistical anomaly indicates an observation that deviates significantly from the statistical properties of the chosen historical dataset. **It does not constitute definitive proof of an individual extreme weather event, measurement defect, or specific local disaster without independent corroborating meteorological station records.**

---

## 5. Responsible Predictive Modeling

### Chronological Train-Test Protocol
When forecasting time series, random shuffling introduces severe data leakage. A strict chronological cutoff is enforced:
* Training period: $t \le T_{\text{split}}$ (e.g. 1850 to 2014)
* Evaluation period: $t > T_{\text{split}}$ (e.g. 2015 to Present)

### Baseline Evaluation Metrics
* **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{m} \sum_{j=1}^m |y_j - \hat{y}_j|$$
* **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{j=1}^m (y_j - \hat{y}_j)^2}$$
* **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum (y_j - \hat{y}_j)^2}{\sum (y_j - \bar{y})^2}$$

* **Model Scope Notice**: Simple autoregressive and polynomial extrapolations are educational baselines. They cannot substitute for complex coupled General Circulation Models (GCMs, CMIP6) that model geophysical Navier-Stokes fluid dynamics, atmospheric chemistry, and carbon cycle feedbacks.
