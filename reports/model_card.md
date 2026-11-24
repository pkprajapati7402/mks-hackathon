# 📑 AI/ML Model Card: Celeris Automobili Demand Forecasting Engine

## 1. Model Overview
- **Model Architecture**: Multi-Tier Hybrid Demand Forecasting Engine
- **Primary Learner**: Global LightGBM Regressor with Tweedie Objective (Variance Power = 1.2) for zero-inflated demand
- **Uncertainty Quantification**: Pinball-loss Quantile Regression (P10, P50, P90) with monotonicity enforcement ($0 \le P10 \le P50 \le P90$)
- **Specialized Sub-models**: Croston SBA/TSB for lumpy/intermittent series, Seasonal Naive & Rolling Moving Average for baseline benchmarks
- **Hierarchy Engine**: Exact bottom-up reconciliation ensuring zero discrepancy across SKU, Product Family, Business Unit, Region, and Global Portfolio

## 2. Intended Use & Scope
- **Forecast Horizon**: 19 months (November 2026 to May 2028)
- **Target Metric**: `Units_Sold_Unconstrained` (unconstrained customer order demand)
- **Business Impact**: Drives quarterly procurement, safety stock sizing, factory scheduling, and supplier capacity commitments across 6 global operating regions

## 3. Backtest Performance Summary

| Origin   | Model            |   Horizon_Months |   WAPE_Pct |   Bias_Pct |   RMSE |
|:---------|:-----------------|-----------------:|-----------:|-----------:|-------:|
| 2023-10  | Seasonal Naive   |               19 |      43.14 |      -8.76 |  41.93 |
| 2023-10  | Moving Average   |               19 |      53.47 |      -7.49 |  45.36 |
| 2023-10  | Intermittent SBA |               19 |      53.85 |     -14.02 |  45.6  |
| 2023-10  | LightGBM P50     |               19 |      35.7  |      -3.9  |  34.4  |
| 2024-10  | Seasonal Naive   |               19 |      40.25 |      -1.51 |  39.13 |
| 2024-10  | Moving Average   |               19 |      52.53 |       0.29 |  43.7  |
| 2024-10  | Intermittent SBA |               19 |      52.35 |      -5.87 |  43.85 |
| 2024-10  | LightGBM P50     |               19 |      35.05 |       1.03 |  32.72 |
| 2025-03  | Seasonal Naive   |               19 |      41.19 |       4.39 |  40.51 |
| 2025-03  | Moving Average   |               19 |      53.67 |       4.78 |  43.86 |
| 2025-03  | Intermittent SBA |               19 |      52.31 |      -5.24 |  43.6  |
| 2025-03  | LightGBM P50     |               19 |      34.81 |       1.66 |  32.16 |

## 4. Per-Family Champion Model Selection

| Product_Family           | Selected_Champion_Model   |   Selection_WAPE_Pct |   Holdout_WAPE_Pct |
|:-------------------------|:--------------------------|---------------------:|-------------------:|
| Brake & Safety Systems   | LightGBM P50              |                33.31 |              32.57 |
| Chassis & Structural     | LightGBM P50              |                33.99 |              34    |
| Electrical Assemblies    | LightGBM P50              |                36.51 |              35.61 |
| Hydraulic Systems        | LightGBM P50              |                36.76 |              36.58 |
| Precision Machined Parts | LightGBM P50              |                36.61 |              36.65 |
| Powertrain Components    | LightGBM P50              |                35.67 |              35.66 |
| Thermal Management       | LightGBM P50              |                35.77 |              33.86 |

## 5. Top Predictive Drivers (LightGBM Feature Importance)

| Feature | Importance Score |
|---|---|
| `adi` | 572 |
| `croston_baseline_rate` | 496 |
| `lag_12` | 450 |
| `rolling_std_12` | 232 |
| `months_since_launch` | 230 |
| `cv2` | 213 |
| `Region` | 184 |
| `sba_baseline_rate` | 179 |

## 6. Ethical & Operational Governance
- **Lifecycle Boundary Safety**: Hard constraints prevent ghost forecasts post-EOL or prior to commercial launch.
- **Monotonicity Guarantee**: Upper procurement buffers (P90) strictly bound median operational forecasts (P50) and lower safety allocations (P10).
- **Audit Trail**: Every transaction-line adjustment is tracked with deterministic rule IDs in `reports/audit_trail.parquet`.
