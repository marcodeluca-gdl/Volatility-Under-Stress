# Volatility Under Stress: Pandemic Effects, Leverage, and Intraday Risk Measurement

## Overview

This project investigates how extreme market conditions affect financial
volatility dynamics, with a focus on three interconnected dimensions:

1.  Structural changes in volatility during the COVID-19 pandemic.
2.  Asymmetric volatility dynamics and the leverage effect using
    GARCH-type models.
3.  High-frequency volatility estimation under microstructure noise and
    intraday Value-at-Risk (VaR) backtesting.

The project combines econometric inference, conditional heteroskedastic
modeling, and high-frequency financial econometrics into a unified
risk-measurement framework.

------------------------------------------------------------------------

## Part I -- Pandemic Effect on Volatility (HAC / Newey--West)

### Research Question

Did COVID-19 produce a statistically significant structural increase in
volatility?

### Assets

-   Copper Futures (COMEX) -- industrial, shock-sensitive commodity
-   Bloomberg Commodity Index (BCOM) -- diversified commodity benchmark

### Methodology

-   Daily log-returns computed as $r_t = log(P_t / P\_{t-1})$
-   Volatility proxy: squared returns $r_t\^2$
-   Pre-pandemic sample: 2015--2019
-   Post-pandemic sample: 2020--2024
-   Weak stationarity diagnostics (ADF, Ljung--Box)
-   HAC / Newey--West robust mean comparison

### Main Findings

-   Volatility significantly increases after 2020 for both assets.
-   Copper remains more volatile than BCOM in both regimes.
-   The volatility gap widens in the post-pandemic period.

This confirms a statistically significant structural shift in
volatility.

------------------------------------------------------------------------

## Part II -- Leverage Effect: GARCH vs T-GARCH vs E-GARCH

### Modeling Framework

Returns are modeled as:

$$ε_t = σ_t Z_t, Z_t \~ i.i.d.(0,1)$$

Estimated via Gaussian Quasi-Maximum Likelihood.

### Models Considered

**GARCH(1,1)**\
Captures volatility clustering but assumes symmetric response to shocks.

**T-GARCH(1,1)**\
Introduces threshold asymmetry (α− vs α+).\
Likelihood Ratio tests show no significant leverage effect under this
specification.

**E-GARCH(1,1)**\
Models log-variance dynamics and allows flexible asymmetric response
through parameter λ.

### Results

-   T-GARCH fails to detect leverage effects.
-   E-GARCH strongly rejects symmetry (λ = 0).
-   λ \< 0 in all cases → negative shocks increase future volatility
    more than positive shocks.

Conclusion: E-GARCH provides a more flexible and statistically adequate
representation of volatility under stress conditions.

------------------------------------------------------------------------

## Part III -- High-Frequency Volatility and Intraday VaR

### Data

-   LOBSTER limit order book dataset
-   AMZN, 2012-06-21
-   600,000 observations
-   1-second sampling grid

### Problem

At very high frequencies, realized variance diverges due to
microstructure noise.

### Noise Model

Observed log-price: $Y_t = X_t + η ε_t$

Two estimators for noise scale η: - Lag-1 autocovariance method -
Realized variance explosion method

### Noise-Robust Estimator

Subsampled realized variance:

RVS = RV(avg) − (m̄ / n) RV(all)

This mitigates microstructure contamination.

------------------------------------------------------------------------

## Intraday VaR and Pseudo-Backtest

### VaR Approaches

-   Non-parametric VaR (empirical quantile)
-   Parametric VaR (naive realized variance)
-   Parametric VaR (noise-robust realized variance)

### Backtest Setup

-   Confidence level: 5%
-   Horizon: 60 seconds
-   Rolling training window: 2 hours
-   Test window: 30 minutes

### Findings

-   Parametric VaR using naive RV is overly conservative.
-   Robust realized variance improves coverage accuracy.
-   Non-parametric and robust parametric VaR align better with nominal
    levels.

------------------------------------------------------------------------

## Project Structure
```
├── AMZN_2012-06-21_34200000_57600000_message_1.csv 
│ 
├── AMZN_2012-06-21_34200000_57600000_orderbook_1.csv 
│ 
├── HighFrequencyVolatility.ipynb
│ 
├── PandemicEffects_Garch.ipynb 
│ 
├── garchVariationModels.py
│ 
├── pandemicEffectFcts.py
│ 
└── README.md
```
------------------------------------------------------------------------

## Technologies

-   Python 3.x
-   numpy
-   pandas
-   statsmodels
-   arch
-   matplotlib
-   seaborn

------------------------------------------------------------------------

## Authors

Marco De Luca\
Massil Gouachi
