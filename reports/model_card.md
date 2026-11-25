# 📑 AI/ML Model Card: Celeris Automobili Demand Forecasting Engine

## 1. Model Overview
- **Model Architecture**: Multi-Paradigm AI/ML & Deep Learning Hybrid Ensemble + Scaled Conformal Quantiles (CQR)
- **Constituent Models**: Two-Stage Zero-Inflated Hurdle (`LGBMClassifier * LGBM Positive Tweedie`) + Global LightGBM Tweedie (`p=1.25`) + LightGBM Multiplicative Log-Ratio Residual + Scikit-Learn HistGB (`poisson`) + PyTorch Entity-Embedding Residual Network + Empirical Bayes Structural Anchor
- **Out-of-Sample Simplex Blend Weights**: Tweedie=0.0, LogRatio=0.528, Hurdle=0.326, HistGB=0.0, PyTorch-ResNet=0.0, Anchor=0.146 | **Hurdle Zero Gate**: `tau=0.1`
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
| 2023-10  | HistGB Poisson           |               19 |      31.82 |     -11.29 |  27.68 |              nan   |                     nan   |               7.54 |             nan    |
| 2023-10  | PyTorch Entity-ResNet    |               19 |      32.33 |      -3.04 |  27.35 |              nan   |                     nan   |               7.66 |             nan    |
| 2023-10  | Hurdle Two-Stage         |               19 |      32.12 |     -10.02 |  28.19 |              nan   |                     nan   |               7.61 |             nan    |
| 2023-10  | LightGBM Tweedie         |               19 |      32.8  |      -9.62 |  28.31 |              nan   |                     nan   |               7.78 |             nan    |
| 2023-10  | LightGBM LogRatio        |               19 |      32.65 |     -13.66 |  28.52 |              nan   |                     nan   |               7.74 |             nan    |
| 2023-10  | Grandmaster Ensemble P50 |               19 |      31.39 |      -8.97 |  27.53 |               81.2 |                      87.9 |               5.02 |              76.32 |
| 2024-10  | Seasonal Naive           |               19 |      38.75 |      -1.04 |  31.72 |              nan   |                     nan   |               8.89 |             nan    |
| 2024-10  | Moving Average           |               19 |      51.49 |       0.89 |  39.73 |              nan   |                     nan   |              11.81 |             nan    |
| 2024-10  | Intermittent SBA         |               19 |      51.47 |      -5.32 |  39.95 |              nan   |                     nan   |              11.8  |             nan    |
| 2024-10  | HistGB Poisson           |               19 |      30.74 |      -3.13 |  25.43 |              nan   |                     nan   |               7.05 |             nan    |
| 2024-10  | PyTorch Entity-ResNet    |               19 |      29.81 |      -1.66 |  24.78 |              nan   |                     nan   |               6.84 |             nan    |
| 2024-10  | Hurdle Two-Stage         |               19 |      30.49 |      -3.15 |  25.24 |              nan   |                     nan   |               6.99 |             nan    |
| 2024-10  | LightGBM Tweedie         |               19 |      31.17 |      -2.9  |  25.57 |              nan   |                     nan   |               7.15 |             nan    |
| 2024-10  | LightGBM LogRatio        |               19 |      30.03 |      -8.35 |  25.39 |              nan   |                     nan   |               6.88 |             nan    |
| 2024-10  | Grandmaster Ensemble P50 |               19 |      29.48 |      -2.74 |  24.68 |               84.3 |                      86.5 |               4.51 |              67.65 |
| 2025-03  | Seasonal Naive           |               19 |      39.64 |       4.42 |  33.71 |              nan   |                     nan   |               9.14 |             nan    |
| 2025-03  | Moving Average           |               19 |      52.48 |       4.67 |  41.14 |              nan   |                     nan   |              12.1  |             nan    |
| 2025-03  | Intermittent SBA         |               19 |      51.31 |      -5.41 |  40.93 |              nan   |                     nan   |              11.83 |             nan    |
| 2025-03  | HistGB Poisson           |               19 |      30.19 |       3.76 |  25.96 |              nan   |                     nan   |               6.96 |             nan    |
| 2025-03  | PyTorch Entity-ResNet    |               19 |      30.51 |       3.91 |  26.1  |              nan   |                     nan   |               7.03 |             nan    |
| 2025-03  | Hurdle Two-Stage         |               19 |      30.04 |       3.47 |  25.8  |              nan   |                     nan   |               6.92 |             nan    |
| 2025-03  | LightGBM Tweedie         |               19 |      30.24 |       3.15 |  25.96 |              nan   |                     nan   |               6.97 |             nan    |
| 2025-03  | LightGBM LogRatio        |               19 |      29.72 |      -1.46 |  26.18 |              nan   |                     nan   |               6.85 |             nan    |
| 2025-03  | Grandmaster Ensemble P50 |               19 |      29.89 |       3.7  |  25.74 |               84.1 |                      85.6 |               4.55 |              67.48 |

## 4. Per-Family Champion Model Selection

| Product_Family           | Selected_Champion_Model   |   Selection_WAPE_Pct |   Holdout_WAPE_Pct |
|:-------------------------|:--------------------------|---------------------:|-------------------:|
| Brake & Safety Systems   | Grandmaster Ensemble P50  |                29.3  |              27.64 |
| Chassis & Structural     | Grandmaster Ensemble P50  |                30.31 |              28.65 |
| Electrical Assemblies    | Grandmaster Ensemble P50  |                32.23 |              34.19 |
| Hydraulic Systems        | Grandmaster Ensemble P50  |                32.62 |              32.81 |
| Precision Machined Parts | Grandmaster Ensemble P50  |                30.38 |              30.39 |
| Powertrain Components    | Grandmaster Ensemble P50  |                29.43 |              30.11 |
| Thermal Management       | Grandmaster Ensemble P50  |                30.01 |              28.2  |

## 5. Top Predictive Drivers (LightGBM Feature Importance)

| Feature | Importance Score |
|---|---|
| `Customer_Name` | 576 |
| `months_since_launch` | 484 |
| `seasonal_index` | 451 |
| `Product_Family` | 426 |
| `analytical_anchor` | 334 |
| `family_region_prior` | 316 |
| `mod3_phase_advantage` | 307 |
| `cv2` | 292 |

## 6. Ethical & Operational Governance
- **Lifecycle Boundary Safety**: Hard constraints prevent ghost forecasts post-EOL or prior to commercial launch.
- **Monotonicity Guarantee**: Upper procurement buffers (P90) strictly bound median operational forecasts (P50) and lower safety allocations (P10).
- **Audit Trail**: Every transaction-line adjustment is tracked with deterministic rule IDs in `reports/audit_trail.parquet`.
