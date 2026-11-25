# 📑 AI/ML Model Card: Celeris Automobili Demand Forecasting Engine

## 1. Model Overview
- **Model Architecture**: Dual-Objective Hybrid Ensemble + Conformalized Quantiles (CQR)
- **Primary Point Forecasters**: Global LightGBM Tweedie (`p=1.25`) + Log1p-L1 MAE + Seasonal-EWM Anchor blend tuned out-of-sample per validation fold
- **Ensemble Blend Weights**: Tweedie=0.55, Log1p-L1=0.45, Seasonal-Anchor=-0.0
- **Uncertainty Quantification**: Horizon-stratified Conformalized Quantile Regression (`P10`, `P90`) calibrated toward 80% empirical coverage with strict $0 \le P10 \le P50 \le P90$
- **Explainability**: Native LightGBM Tree-SHAP feature attributions (`pred_contrib=True`) exported to `outputs/forecast_drivers.parquet`
- **Specialized Sub-models**: Croston SBA, Seasonal Naive & Moving Average routed per Product Family tournament
- **Hierarchy Engine**: Exact bottom-up reconciliation ensuring zero discrepancy across SKU, Product Family, Business Unit, Region, and Global Portfolio

## 2. Intended Use & Scope
- **Forecast Horizon**: 19 months (November 2026 to May 2028)
- **Target Metric**: `Units_Sold_Unconstrained` (unconstrained customer order demand)
- **Business Impact**: Drives quarterly procurement, safety stock sizing, factory scheduling, and supplier capacity commitments across 6 global operating regions

## 3. Backtest Performance Summary

| Origin   | Model            |   Horizon_Months |   WAPE_Pct |   Bias_Pct |   RMSE |   Raw_Coverage_Pct |   Calibrated_Coverage_Pct |
|:---------|:-----------------|-----------------:|-----------:|-----------:|-------:|-------------------:|--------------------------:|
| 2023-10  | Seasonal Naive   |               19 |      41.81 |      -8.6  |  35.56 |              nan   |                     nan   |
| 2023-10  | Moving Average   |               19 |      52.57 |      -7.49 |  41.75 |              nan   |                     nan   |
| 2023-10  | Intermittent SBA |               19 |      53.06 |     -14.01 |  42.03 |              nan   |                     nan   |
| 2023-10  | LightGBM P50     |               19 |      32.98 |     -13.96 |  28.64 |               82.2 |                      82.2 |
| 2024-10  | Seasonal Naive   |               19 |      38.75 |      -1.04 |  31.72 |              nan   |                     nan   |
| 2024-10  | Moving Average   |               19 |      51.49 |       0.89 |  39.73 |              nan   |                     nan   |
| 2024-10  | Intermittent SBA |               19 |      51.47 |      -5.32 |  39.95 |              nan   |                     nan   |
| 2024-10  | LightGBM P50     |               19 |      31.35 |      -4.91 |  25.84 |               85.5 |                      85.5 |
| 2025-03  | Seasonal Naive   |               19 |      39.64 |       4.42 |  33.71 |              nan   |                     nan   |
| 2025-03  | Moving Average   |               19 |      52.48 |       4.67 |  41.14 |              nan   |                     nan   |
| 2025-03  | Intermittent SBA |               19 |      51.31 |      -5.41 |  40.93 |              nan   |                     nan   |
| 2025-03  | LightGBM P50     |               19 |      31.22 |       2.58 |  26.47 |               83.8 |                      83.8 |

## 4. Per-Family Champion Model Selection

| Product_Family           | Selected_Champion_Model   |   Selection_WAPE_Pct |   Holdout_WAPE_Pct |
|:-------------------------|:--------------------------|---------------------:|-------------------:|
| Brake & Safety Systems   | LightGBM P50              |                30.51 |              29.33 |
| Chassis & Structural     | LightGBM P50              |                32.32 |              30.68 |
| Electrical Assemblies    | LightGBM P50              |                34.73 |              36.18 |
| Hydraulic Systems        | LightGBM P50              |                34.01 |              32.49 |
| Precision Machined Parts | LightGBM P50              |                32.06 |              31.52 |
| Powertrain Components    | LightGBM P50              |                30.93 |              30.69 |
| Thermal Management       | LightGBM P50              |                32.08 |              30.04 |

## 5. Top Predictive Drivers (LightGBM Feature Importance)

| Feature | Importance Score |
|---|---|
| `months_since_launch` | 272 |
| `customer_cycle_months` | 255 |
| `seasonal_index` | 238 |
| `same_month_hist_mean` | 227 |
| `origin_std_12m` | 210 |
| `Customer_Name` | 180 |
| `Product_Family` | 173 |
| `cv2` | 140 |

## 6. Ethical & Operational Governance
- **Lifecycle Boundary Safety**: Hard constraints prevent ghost forecasts post-EOL or prior to commercial launch.
- **Monotonicity Guarantee**: Upper procurement buffers (P90) strictly bound median operational forecasts (P50) and lower safety allocations (P10).
- **Audit Trail**: Every transaction-line adjustment is tracked with deterministic rule IDs in `reports/audit_trail.parquet`.
