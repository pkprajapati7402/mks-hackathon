"""
Global LightGBM Multi-Quantile Forecasting Engine (Workstream 4 Advanced & Expert Tier).
Trains Tweedie / L1 P50 models and P10/P90 quantile regression models with strict monotonicity enforcement.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import RANDOM_SEED, REPORTS_DIR

FEATURE_COLUMNS = [
    # Cyclical & Calendar
    "Month", "Quarter", "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    "is_europe_summer_shutdown", "is_golden_week_month", "is_cny_month",
    "is_year_end_holiday", "is_pre_price_hike_month", "is_post_price_hike_month",
    "is_calendar_price_change", "is_calendar_disruption",
    # Macroeconomic Drivers
    "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index", "EV_Adoption_Index",
    "ISM_Manufacturing_PMI_lag1", "Auto_Production_YoY_Pct_lag1",
    "New_Vehicle_Orders_Index_lag1", "New_Vehicle_Orders_Index_lag3",
    "Industrial_Util_Rate_Pct_lag1",
    # Intermittency & Baselines
    "adi", "cv2", "croston_baseline_rate", "sba_baseline_rate",
    # Lifecycle & Catalog
    "months_since_launch", "months_until_eol", "is_pre_launch", "is_post_eol",
    "has_replacement_sku",
    # EV Powertrain Interaction
    "is_powertrain_ev", "is_powertrain_ice", "ev_index_x_is_ev",
    # Seasonal Anchors
    "lag_12", "rolling_mean_12", "rolling_std_12"
]

CATEGORICAL_COLUMNS = [
    "Product_Family", "Region", "Business_Unit",
    "intermittency_category", "Powertrain_Application"
]

def prepare_lgb_data(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    cat_cols: List[str] = None
) -> pd.DataFrame:
    """Prepares and encodes feature matrix for LightGBM."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    X = df[feature_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X

def train_and_forecast_lgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Units_Sold_Unconstrained",
    n_estimators: int = 150
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains P50 (Tweedie/L1), P10 (Quantile alpha=0.1), and P90 (Quantile alpha=0.9) LightGBM models.
    Enforces strict quantile crossing monotonicity: 0 <= P10 <= P50 <= P90.
    """
    X_train = prepare_lgb_data(train_df)
    y_train = train_df[target_col].clip(lower=0.0).values
    X_test = prepare_lgb_data(test_df)

    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]

    # 1. Primary P50 Model (Tweedie variance power 1.2)
    params_p50 = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": n_estimators,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1
    }
    model_p50 = lgb.LGBMRegressor(**params_p50)
    model_p50.fit(X_train, y_train, categorical_feature=cat_cols)
    pred_p50 = np.clip(model_p50.predict(X_test), 0.0, None)

    # 2. Lower Quantile P10 Model
    params_p10 = {
        "objective": "quantile",
        "alpha": 0.1,
        "boosting_type": "gbdt",
        "n_estimators": max(80, int(n_estimators * 0.7)),
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1
    }
    model_p10 = lgb.LGBMRegressor(**params_p10)
    model_p10.fit(X_train, y_train, categorical_feature=cat_cols)
    pred_p10 = np.clip(model_p10.predict(X_test), 0.0, None)

    # 3. Upper Quantile P90 Model
    params_p90 = {
        "objective": "quantile",
        "alpha": 0.9,
        "boosting_type": "gbdt",
        "n_estimators": max(80, int(n_estimators * 0.7)),
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1
    }
    model_p90 = lgb.LGBMRegressor(**params_p90)
    model_p90.fit(X_train, y_train, categorical_feature=cat_cols)
    pred_p90 = np.clip(model_p90.predict(X_test), 0.0, None)

    # 4. Strict Quantile Monotonicity Enforcement: P10 <= P50 <= P90
    pred_p10_enforced = np.minimum(pred_p10, pred_p50)
    pred_p90_enforced = np.maximum(pred_p90, pred_p50)

    # Extract Feature Importances
    importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model_p50.feature_importances_
    }).sort_values("Importance", ascending=False)
    importances.to_csv(REPORTS_DIR / "feature_importances.csv", index=False)

    out_preds = pd.DataFrame({
        "P10": pred_p10_enforced.round(),
        "P50": pred_p50.round(),
        "P90": pred_p90_enforced.round()
    }, index=test_df.index)

    metadata = {
        "feature_count": len(X_train.columns),
        "top_features": importances.head(10).to_dict(orient="records"),
        "p50_mean": float(pred_p50.mean()),
        "p10_mean": float(pred_p10_enforced.mean()),
        "p90_mean": float(pred_p90_enforced.mean())
    }

    return out_preds, metadata
