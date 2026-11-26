# 📑 AI/ML Model Card: Celeris Automobili Demand Forecasting Engine

## 1. Model Overview
- **Model Architecture**: Multi-Paradigm AI/ML & Deep Learning Hybrid Ensemble + Scaled Conformal Quantiles (CQR)
- **Constituent Models**: Two-Stage Zero-Inflated Hurdle (`LGBMClassifier * LGBM Positive Tweedie`) + Global LightGBM Tweedie (`p=1.25`) + LightGBM Multiplicative Log-Ratio Residual + Scikit-Learn HistGB (`poisson`) + PyTorch Entity-Embedding Residual Network + Empirical Bayes Structural Anchor
- **Out-of-Sample Simplex Blend Weights**: Tweedie=0.056, LogRatio=0.386, Hurdle=0.257, HistGB=0.097, PyTorch-ResNet=0.107, Anchor=0.096 | **Hurdle Zero Gate**: `tau=0.1`
- **Uncertainty Quantification**: Horizon-stratified Scaled Conformalized Quantile Regression (`P10`, `P90`) calibrated toward ~80.5% empirical coverage with strict $0 \le P10 \le P50 \le P90$
- **Explainability**: Native LightGBM Tree-SHAP feature attributions (`pred_contrib=True`) exported to `outputs/forecast_drivers.parquet`
- **Specialized Sub-models**: Croston SBA, Seasonal Naive & Moving Average routed per Product Family tournament
- **Hierarchy Engine**: Exact bottom-up reconciliation ensuring zero discrepancy across SKU, Product Family, Business Unit, Region, and Global Portfolio

## 2. Intended Use & Scope
- **Forecast Horizon**: 19 months (November 2026 to May 2028)
- **Target Metric**: `Units_Sold_Unconstrained` (unconstrained customer order demand)
- **Business Impact**: Drives quarterly procurement, safety stock sizing, factory scheduling, and supplier capacity commitments across 6 global operating regions

## 3. Backtest Performance Summary

| Origin   | Model                    |   Horizon_Months |   WAPE_Pct |   Bias_Pct |   RMSE |   Raw_Coverage_Pct |   Calibrated_Coverage_Pct |   Pinball_Loss_Avg |   Winkler_80_Score |
|:---------|:-------------------------|-----------------:|-----------:|-----------:|-------:|-------------------:|--------------------------:|-------------------:|-------------------:|
| 2023-10  | Seasonal Naive           |               19 |      41.81 |      -8.6  |  35.56 |              nan   |                     nan   |               9.91 |             nan    |
| 2023-10  | Moving Average           |               19 |      52.57 |      -7.49 |  41.75 |              nan   |                     nan   |              12.46 |             nan    |
| 2023-10  | Intermittent SBA         |               19 |      53.06 |     -14.01 |  42.03 |              nan   |                     nan   |              12.58 |             nan    |
| 2023-10  | HistGB Poisson           |               19 |      31.28 |      -9.23 |  27.13 |              nan   |                     nan   |               7.41 |             nan    |
| 2023-10  | PyTorch Entity-ResNet    |               19 |      32.06 |      -2.83 |  27.26 |              nan   |                     nan   |               7.6  |             nan    |
| 2023-10  | Hurdle Two-Stage         |               19 |      31.96 |     -10.12 |  28.09 |              nan   |                     nan   |               7.57 |             nan    |
| 2023-10  | LightGBM Tweedie         |               19 |      32.6  |      -9.72 |  28.12 |              nan   |                     nan   |               7.73 |             nan    |
| 2023-10  | LightGBM LogRatio        |               19 |      32.2  |     -13.16 |  28.24 |              nan   |                     nan   |               7.63 |             nan    |
| 2023-10  | Grandmaster Ensemble P50 |               19 |      30.88 |      -7.16 |  27.02 |               82.7 |                      86.7 |               4.89 |              73.51 |
| 2024-10  | Seasonal Naive           |               19 |      38.75 |      -1.04 |  31.72 |              nan   |                     nan   |               8.89 |             nan    |
| 2024-10  | Moving Average           |               19 |      51.49 |       0.89 |  39.73 |              nan   |                     nan   |              11.81 |             nan    |
| 2024-10  | Intermittent SBA         |               19 |      51.47 |      -5.32 |  39.95 |              nan   |                     nan   |              11.8  |             nan    |
| 2024-10  | HistGB Poisson           |               19 |      30.2  |      -2.55 |  24.95 |              nan   |                     nan   |               6.92 |             nan    |
| 2024-10  | PyTorch Entity-ResNet    |               19 |      29.69 |      -4.55 |  24.98 |              nan   |                     nan   |               6.81 |             nan    |
| 2024-10  | Hurdle Two-Stage         |               19 |      30.26 |      -4.25 |  25.16 |              nan   |                     nan   |               6.94 |             nan    |
| 2024-10  | LightGBM Tweedie         |               19 |      30.9  |      -2.29 |  25.34 |              nan   |                     nan   |               7.09 |             nan    |
| 2024-10  | LightGBM LogRatio        |               19 |      30.52 |      -9.11 |  25.76 |              nan   |                     nan   |               7    |             nan    |
| 2024-10  | Grandmaster Ensemble P50 |               19 |      29.6  |      -2.32 |  24.75 |               84.3 |                      84.2 |               4.47 |              66.23 |
| 2025-03  | Seasonal Naive           |               19 |      39.64 |       4.42 |  33.71 |              nan   |                     nan   |               9.14 |             nan    |
| 2025-03  | Moving Average           |               19 |      52.48 |       4.67 |  41.14 |              nan   |                     nan   |              12.1  |             nan    |
| 2025-03  | Intermittent SBA         |               19 |      51.31 |      -5.41 |  40.93 |              nan   |                     nan   |              11.83 |             nan    |
| 2025-03  | HistGB Poisson           |               19 |      29.83 |       2.18 |  25.85 |              nan   |                     nan   |               6.88 |             nan    |
| 2025-03  | PyTorch Entity-ResNet    |               19 |      31.71 |       8.22 |  26.59 |              nan   |                     nan   |               7.31 |             nan    |
| 2025-03  | Hurdle Two-Stage         |               19 |      30.11 |       3.28 |  25.83 |              nan   |                     nan   |               6.94 |             nan    |
| 2025-03  | LightGBM Tweedie         |               19 |      30.22 |       3.08 |  25.93 |              nan   |                     nan   |               6.96 |             nan    |
| 2025-03  | LightGBM LogRatio        |               19 |      29.81 |      -1.36 |  26.23 |              nan   |                     nan   |               6.87 |             nan    |
| 2025-03  | Grandmaster Ensemble P50 |               19 |      29.59 |       3.1  |  25.6  |               84.4 |                      80.1 |               4.48 |              66.06 |

## 4. Per-Family Champion Model Selection

| Product_Family           | Selected_Champion_Model   |   Selection_WAPE_Pct |   Holdout_WAPE_Pct |
|:-------------------------|:--------------------------|---------------------:|-------------------:|
| Brake & Safety Systems   | Grandmaster Ensemble P50  |                29.19 |              27.83 |
| Chassis & Structural     | Grandmaster Ensemble P50  |                30.18 |              28.54 |
| Electrical Assemblies    | PyTorch Entity-ResNet     |                31.96 |              35.09 |
| Hydraulic Systems        | Grandmaster Ensemble P50  |                32.21 |              33.13 |
| Precision Machined Parts | Grandmaster Ensemble P50  |                30.15 |              30.23 |
| Powertrain Components    | Grandmaster Ensemble P50  |                29.42 |              29.93 |
| Thermal Management       | Grandmaster Ensemble P50  |                29.64 |              27.88 |

## 5. Top Predictive Drivers (LightGBM Feature Importance)

| Feature | Importance Score |
|---|---|
| `Customer_Name` | 644 |
| `months_since_launch` | 562 |
| `Product_Family` | 517 |
| `seasonal_index` | 494 |
| `family_region_prior` | 380 |
| `sku_price_log` | 351 |
| `sku_global_prior` | 342 |
| `mod2_phase_advantage` | 331 |

## 6. Ethical & Operational Governance
- **Lifecycle Boundary Safety**: Hard constraints prevent ghost forecasts post-EOL or prior to commercial launch.
- **Monotonicity Guarantee**: Upper procurement buffers (P90) strictly bound median operational forecasts (P50) and lower safety allocations (P10).
- **Audit Trail**: Every transaction-line adjustment is tracked with deterministic rule IDs in `reports/audit_trail.parquet`.
