# 📑 AI/ML Model Card: Celeris Automobili Demand Forecasting Engine

## 1. Model Overview
- **Model Architecture**: Multi-Paradigm AI/ML & Deep Learning Hybrid Ensemble + Scaled Conformal Quantiles (CQR)
- **Constituent Models**: Two-Stage Zero-Inflated Hurdle (`LGBMClassifier * LGBM Positive Tweedie`) + Global LightGBM Tweedie (`p=1.25`) + LightGBM Multiplicative Log-Ratio Residual + Scikit-Learn HistGB (`poisson`) + PyTorch Entity-Embedding Residual Network + Empirical Bayes Structural Anchor
- **Out-of-Sample Simplex Blend Weights**: Tweedie=0.038, LogRatio=0.404, Hurdle=0.264, HistGB=0.094, PyTorch-ResNet=0.072, Anchor=0.129 | **Hurdle Zero Gate**: `tau=0.1`
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
| 2023-10  | HistGB Poisson           |               19 |      31.37 |      -9.38 |  27.21 |              nan   |                     nan   |               7.43 |             nan    |
| 2023-10  | PyTorch Entity-ResNet    |               19 |      32.06 |      -2.83 |  27.26 |              nan   |                     nan   |               7.6  |             nan    |
| 2023-10  | Hurdle Two-Stage         |               19 |      31.8  |      -8.89 |  27.9  |              nan   |                     nan   |               7.54 |             nan    |
| 2023-10  | LightGBM Tweedie         |               19 |      32.52 |      -9.59 |  28.02 |              nan   |                     nan   |               7.71 |             nan    |
| 2023-10  | LightGBM LogRatio        |               19 |      32.18 |     -13.1  |  28.22 |              nan   |                     nan   |               7.63 |             nan    |
| 2023-10  | Grandmaster Ensemble P50 |               19 |      31.09 |      -8.58 |  27.28 |               82   |                      85.3 |               4.93 |              74.34 |
| 2024-10  | Seasonal Naive           |               19 |      38.75 |      -1.04 |  31.72 |              nan   |                     nan   |               8.89 |             nan    |
| 2024-10  | Moving Average           |               19 |      51.49 |       0.89 |  39.73 |              nan   |                     nan   |              11.81 |             nan    |
| 2024-10  | Intermittent SBA         |               19 |      51.47 |      -5.32 |  39.95 |              nan   |                     nan   |              11.8  |             nan    |
| 2024-10  | HistGB Poisson           |               19 |      30.27 |      -2.4  |  24.98 |              nan   |                     nan   |               6.94 |             nan    |
| 2024-10  | PyTorch Entity-ResNet    |               19 |      29.69 |      -4.57 |  24.99 |              nan   |                     nan   |               6.81 |             nan    |
| 2024-10  | Hurdle Two-Stage         |               19 |      30.14 |      -3.79 |  25.02 |              nan   |                     nan   |               6.91 |             nan    |
| 2024-10  | LightGBM Tweedie         |               19 |      30.91 |      -2.19 |  25.4  |              nan   |                     nan   |               7.09 |             nan    |
| 2024-10  | LightGBM LogRatio        |               19 |      30.25 |      -9.52 |  25.59 |              nan   |                     nan   |               6.94 |             nan    |
| 2024-10  | Grandmaster Ensemble P50 |               19 |      29.5  |      -3.94 |  24.81 |               84.2 |                      83.1 |               4.47 |              66.5  |
| 2025-03  | Seasonal Naive           |               19 |      39.64 |       4.42 |  33.71 |              nan   |                     nan   |               9.14 |             nan    |
| 2025-03  | Moving Average           |               19 |      52.48 |       4.67 |  41.14 |              nan   |                     nan   |              12.1  |             nan    |
| 2025-03  | Intermittent SBA         |               19 |      51.31 |      -5.41 |  40.93 |              nan   |                     nan   |              11.83 |             nan    |
| 2025-03  | HistGB Poisson           |               19 |      29.96 |       2.85 |  25.88 |              nan   |                     nan   |               6.91 |             nan    |
| 2025-03  | PyTorch Entity-ResNet    |               19 |      31.7  |       8.17 |  26.58 |              nan   |                     nan   |               7.31 |             nan    |
| 2025-03  | Hurdle Two-Stage         |               19 |      29.69 |       2.13 |  25.71 |              nan   |                     nan   |               6.84 |             nan    |
| 2025-03  | LightGBM Tweedie         |               19 |      30.71 |       4.09 |  26.12 |              nan   |                     nan   |               7.08 |             nan    |
| 2025-03  | LightGBM LogRatio        |               19 |      29.89 |      -1.82 |  26.3  |              nan   |                     nan   |               6.89 |             nan    |
| 2025-03  | Grandmaster Ensemble P50 |               19 |      29.76 |       3.95 |  25.63 |               84.1 |                      81   |               4.49 |              66.22 |

## 4. Per-Family Champion Model Selection

| Product_Family           | Selected_Champion_Model   |   Selection_WAPE_Pct |   Holdout_WAPE_Pct |
|:-------------------------|:--------------------------|---------------------:|-------------------:|
| Brake & Safety Systems   | Grandmaster Ensemble P50  |                29.31 |              27.67 |
| Chassis & Structural     | Grandmaster Ensemble P50  |                30.24 |              28.63 |
| Electrical Assemblies    | Grandmaster Ensemble P50  |                31.95 |              32.28 |
| Hydraulic Systems        | Grandmaster Ensemble P50  |                32.47 |              33.83 |
| Precision Machined Parts | Grandmaster Ensemble P50  |                30.29 |              30.43 |
| Powertrain Components    | Grandmaster Ensemble P50  |                29.27 |              29.79 |
| Thermal Management       | Grandmaster Ensemble P50  |                29.65 |              28    |

## 5. Top Predictive Drivers (LightGBM Feature Importance)

| Feature | Importance Score |
|---|---|
| `Customer_Name` | 763 |
| `months_since_launch` | 636 |
| `seasonal_index` | 614 |
| `Product_Family` | 601 |
| `sku_price_log` | 451 |
| `sku_global_prior` | 426 |
| `family_region_prior` | 420 |
| `cv2` | 405 |

## 6. Ethical & Operational Governance
- **Lifecycle Boundary Safety**: Hard constraints prevent ghost forecasts post-EOL or prior to commercial launch.
- **Monotonicity Guarantee**: Upper procurement buffers (P90) strictly bound median operational forecasts (P50) and lower safety allocations (P10).
- **Audit Trail**: Every transaction-line adjustment is tracked with deterministic rule IDs in `reports/audit_trail.parquet`.
