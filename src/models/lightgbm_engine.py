"""
Global LightGBM Dual-Objective + Conformal Quantile + Tree-SHAP Engine (Workstream 4 Advanced & Expert Tier).
- Combines LightGBM Tweedie (volume-unbiased) + LightGBM Log1p-L1 (percentage-error / WAPE optimal) for P50.
- Applies Horizon-Stratified Conformalized Quantile Regression (CQR) on P10/P90 to target ~80% empirical coverage.
- Computes native LightGBM Tree-SHAP feature attributions (pred_contrib=True) for interactive explainability.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import RANDOM_SEED, REPORTS_DIR, OUTPUTS_DIR

FEATURE_COLUMNS = [
    # Origin-Anchored Level, Momentum & Seasonal Profiles (strictly causal <= origin)
    "origin_mean_3m", "origin_mean_6m", "origin_mean_12m", "origin_mean_24m",
    "origin_median_12m", "origin_std_12m", "origin_ewm", "origin_trend_ratio",
    "origin_nonzero_rate", "historical_stockout_rate",
    "same_month_hist_mean", "same_month_last_obs", "seasonal_index", "seasonal_ewm_anchor",
    "sku_global_prior", "family_region_prior", "customer_region_prior", "predecessor_prior",
    "horizon_months", "horizon_log",
    # Customer Ordering Cycle (Rule 6)
    "customer_cycle_months", "months_since_last_order", "is_expected_cycle_hit",
    # Cyclical & Calendar + Scoped Events
    "Month", "Quarter", "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    "is_europe_summer_shutdown", "is_na_retool_month", "is_golden_week_month",
    "is_cny_month", "is_ramadan_month", "is_year_end_holiday",
    "is_pre_price_hike_month", "is_post_price_hike_month", "announced_price_hike_pct",
    "is_calendar_price_change", "is_calendar_disruption", "is_supply_constraint_event",
    "is_orion_retool_event", "is_helios_expansion", "is_erp_migration_gap",
    "is_marketing_event", "is_mea_pre_entry",
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
    "is_powertrain_ev", "is_powertrain_ice", "ev_index_x_is_ev", "ev_index_x_is_ice",
    # Compatibility anchors
    "lag_12", "rolling_mean_12", "rolling_std_12",
]

CATEGORICAL_COLUMNS = [
    "Product_Family", "Region", "Business_Unit",
    "Customer_Name", "Sales_Channel", "intermittency_category", "Powertrain_Application"
]

DRIVER_GROUPS = {
    "Baseline & Seasonal Level": {
        "origin_mean_3m", "origin_mean_6m", "origin_mean_12m", "origin_mean_24m",
        "origin_median_12m", "origin_std_12m", "origin_ewm", "origin_trend_ratio",
        "origin_nonzero_rate", "historical_stockout_rate", "same_month_hist_mean",
        "same_month_last_obs", "seasonal_index", "seasonal_ewm_anchor",
        "sku_global_prior", "family_region_prior", "customer_region_prior",
        "horizon_months", "horizon_log", "adi", "cv2", "croston_baseline_rate",
        "sba_baseline_rate", "lag_12", "rolling_mean_12", "rolling_std_12",
        "SKU_ID", "Product_Family", "Region", "Business_Unit", "intermittency_category",
    },
    "Macro Economy": {
        "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
        "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
        "Fuel_Price_Index", "ISM_Manufacturing_PMI_lag1", "Auto_Production_YoY_Pct_lag1",
        "New_Vehicle_Orders_Index_lag1", "New_Vehicle_Orders_Index_lag3",
        "Industrial_Util_Rate_Pct_lag1",
    },
    "EV Transition": {
        "EV_Adoption_Index", "is_powertrain_ev", "is_powertrain_ice",
        "ev_index_x_is_ev", "ev_index_x_is_ice", "Powertrain_Application",
    },
    "Calendar & Price Hikes": {
        "Month", "Quarter", "month_sin", "month_cos", "quarter_sin", "quarter_cos",
        "is_europe_summer_shutdown", "is_na_retool_month", "is_golden_week_month",
        "is_cny_month", "is_ramadan_month", "is_year_end_holiday",
        "is_pre_price_hike_month", "is_post_price_hike_month", "announced_price_hike_pct",
        "is_calendar_price_change",
    },
    "Customer & Scoped Events": {
        "Customer_Name", "Sales_Channel", "customer_cycle_months",
        "months_since_last_order", "is_expected_cycle_hit",
        "is_calendar_disruption", "is_supply_constraint_event",
        "is_orion_retool_event", "is_helios_expansion", "is_erp_migration_gap",
        "is_marketing_event", "is_mea_pre_entry",
    },
    "Catalog & Lifecycle": {
        "months_since_launch", "months_until_eol", "is_pre_launch",
        "is_post_eol", "has_replacement_sku", "predecessor_prior",
    },
}


def prepare_lgb_data(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    cat_cols: List[str] = None,
    category_levels: Dict[str, pd.Index] = None,
) -> pd.DataFrame:
    """Prepares and aligns numeric/categorical columns identically between train and test."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    X = df[feature_cols + cat_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    for c in cat_cols:
        vals = X[c].astype(str).fillna("Unknown")
        if category_levels is not None and c in category_levels:
            X[c] = pd.Categorical(vals, categories=category_levels[c])
        else:
            X[c] = vals.astype("category")

    return X


def _compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Downweights structural anomalies (COVID lockdowns, ERP gaps) and boosts recent history."""
    w = np.ones(len(df), dtype=float)
    if "Date" in df.columns:
        dates = df["Date"].astype(str)
        w = np.where(dates.between("2020-03", "2020-05"), 0.25, w)
        w = np.where(dates >= "2024-01", 1.25, w)
        w = np.where(dates <= "2020-12", np.minimum(w, 0.70), w)
    if "is_erp_migration_gap" in df.columns:
        w = np.where(df["is_erp_migration_gap"].values == 1, 0.10, w)
    return w


def train_and_forecast_lgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Units_Sold_Unconstrained",
    n_estimators: int = 240,
    compute_shap: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains Hybrid P50 (Tweedie + Log1p-L1 + Seasonal Anchor blend) and CQR-calibrated P10/P90 models.
    Enforces strict quantile monotonicity 0 <= P10 <= P50 <= P90 and optionally exports Tree-SHAP attributions.
    """
    X_train = prepare_lgb_data(train_df)
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]
    cat_levels = {c: X_train[c].cat.categories for c in cat_cols}
    X_test = prepare_lgb_data(test_df, category_levels=cat_levels)

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).clip(lower=0.0).values
    weights = _compute_sample_weights(train_df)

    # Hold out most recent validation window (~last 20% of dates or latest _origin) for blend & CQR calibration
    if "_origin" in train_df.columns and train_df["_origin"].nunique() >= 3:
        latest_orig = sorted(train_df["_origin"].unique())[-1]
        val_mask = (train_df["_origin"] == latest_orig).values
    else:
        cutoff_idx = max(1, int(len(train_df) * 0.82))
        val_mask = np.zeros(len(train_df), dtype=bool)
        val_mask[cutoff_idx:] = True

    tr_mask = ~val_mask
    if tr_mask.sum() < 100 or val_mask.sum() < 50:
        tr_mask = np.ones(len(train_df), dtype=bool)
        val_mask = np.ones(len(train_df), dtype=bool)

    base_params = {
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 7,
        "max_bin": 127,
        "min_child_samples": 35,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    val_trees = max(80, int(n_estimators * 0.65))
    # --- 1. Validation fit for optimal blend weight & CQR conformity intervals ---
    m_tw_val = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.25, n_estimators=val_trees, **base_params)
    m_tw_val.fit(X_train.iloc[tr_mask], y_train[tr_mask], sample_weight=weights[tr_mask], categorical_feature=cat_cols)
    v_tw = np.clip(m_tw_val.predict(X_train.iloc[val_mask]), 0.0, None)

    m_log_val = lgb.LGBMRegressor(objective="regression_l1", n_estimators=val_trees, **base_params)
    m_log_val.fit(X_train.iloc[tr_mask], np.log1p(y_train[tr_mask]), sample_weight=weights[tr_mask], categorical_feature=cat_cols)
    v_log = np.clip(np.expm1(m_log_val.predict(X_train.iloc[val_mask])), 0.0, None)

    v_anchor = (
        pd.to_numeric(train_df.loc[val_mask, "seasonal_ewm_anchor"], errors="coerce").fillna(0.0).clip(lower=0.0).values
        if "seasonal_ewm_anchor" in train_df.columns else v_tw
    )
    y_val = y_train[val_mask]
    denom_val = max(1.0, float(np.sum(y_val)))

    best_w = (0.55, 0.35, 0.10)
    best_wape = float("inf")
    for w_tw in [0.35, 0.45, 0.55, 0.65, 0.75]:
        for w_lg in [0.15, 0.25, 0.35, 0.45]:
            w_an = round(1.0 - w_tw - w_lg, 2)
            if w_an < 0:
                continue
            blend = w_tw * v_tw + w_lg * v_log + w_an * v_anchor
            # Small bias alignment within +/- 3% permitted only from genuine blend
            wape = float(np.sum(np.abs(y_val - blend)) / denom_val)
            if wape < best_wape:
                best_wape = wape
                best_w = (w_tw, w_lg, w_an)

    # Validation Quantile Models for Conformalized Quantile Regression (CQR -> target 80% coverage)
    q_params = {**base_params, "num_leaves": 23, "max_depth": 6, "n_estimators": max(75, int(n_estimators * 0.55))}
    m_p10_val = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **q_params)
    m_p10_val.fit(X_train.iloc[tr_mask], y_train[tr_mask], sample_weight=weights[tr_mask], categorical_feature=cat_cols)
    v_p10 = np.clip(m_p10_val.predict(X_train.iloc[val_mask]), 0.0, None)

    m_p90_val = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **q_params)
    m_p90_val.fit(X_train.iloc[tr_mask], y_train[tr_mask], sample_weight=weights[tr_mask], categorical_feature=cat_cols)
    v_p90 = np.clip(m_p90_val.predict(X_train.iloc[val_mask]), 0.0, None)

    # CQR signed normalized nonconformity score stratified by horizon band
    val_scores = np.maximum(v_p10 - y_val, y_val - v_p90)
    cqr_q_all = float(np.quantile(val_scores, 0.80))
    cqr_margins = {"short": cqr_q_all, "mid": cqr_q_all, "long": cqr_q_all}
    if "horizon_months" in train_df.columns:
        vh = train_df.loc[val_mask, "horizon_months"].values
        for band, mask_b in [("short", vh <= 6), ("mid", (vh >= 7) & (vh <= 12)), ("long", vh >= 13)]:
            if mask_b.sum() >= 50:
                cqr_margins[band] = max(0.0, float(np.quantile(val_scores[mask_b], 0.80)))

    # --- 2. Full-History Refit for Maximum Predictive Power ---
    model_tweedie = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.25, n_estimators=n_estimators, **base_params)
    model_tweedie.fit(X_train, y_train, sample_weight=weights, categorical_feature=cat_cols)
    pred_tw = np.clip(model_tweedie.predict(X_test), 0.0, None)

    model_log1p = lgb.LGBMRegressor(objective="regression_l1", n_estimators=n_estimators, **base_params)
    model_log1p.fit(X_train, np.log1p(y_train), sample_weight=weights, categorical_feature=cat_cols)
    pred_log = np.clip(np.expm1(model_log1p.predict(X_test)), 0.0, None)

    pred_anchor = (
        pd.to_numeric(test_df["seasonal_ewm_anchor"], errors="coerce").fillna(0.0).clip(lower=0.0).values
        if "seasonal_ewm_anchor" in test_df.columns else pred_tw
    )
    pred_p50 = np.clip(best_w[0] * pred_tw + best_w[1] * pred_log + best_w[2] * pred_anchor, 0.0, None)

    model_p10 = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **q_params)
    model_p10.fit(X_train, y_train, sample_weight=weights, categorical_feature=cat_cols)
    raw_p10 = np.clip(model_p10.predict(X_test), 0.0, None)

    model_p90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **q_params)
    model_p90.fit(X_train, y_train, sample_weight=weights, categorical_feature=cat_cols)
    raw_p90 = np.clip(model_p90.predict(X_test), 0.0, None)

    # Apply CQR horizon-stratified calibration
    if "horizon_months" in test_df.columns:
        th = test_df["horizon_months"].values
        margin = np.where(
            th <= 6, cqr_margins["short"],
            np.where(th <= 12, cqr_margins["mid"], cqr_margins["long"])
        )
    else:
        margin = np.full(len(test_df), max(0.0, cqr_q_all))

    cal_p10 = np.clip(raw_p10 - margin, 0.0, None)
    cal_p90 = np.clip(raw_p90 + margin, 0.0, None)

    # Enforce strict monotonicity 0 <= P10 <= P50 <= P90
    pred_p10_enforced = np.minimum(cal_p10, pred_p50)
    pred_p90_enforced = np.maximum(cal_p90, pred_p50)
    raw_p10_enforced = np.minimum(raw_p10, pred_p50)
    raw_p90_enforced = np.maximum(raw_p90, pred_p50)

    # Extract Feature Importances
    importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model_tweedie.feature_importances_,
    }).sort_values("Importance", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importances.to_csv(REPORTS_DIR / "feature_importances.csv", index=False)

    # 3. Native Tree-SHAP Driver Attribution (pred_contrib=True on Tweedie backbone)
    if compute_shap and len(X_test) > 0:
        contrib = model_tweedie.predict(X_test, pred_contrib=True)
        feat_names = list(X_train.columns)
        # Map raw log-link Tweedie SHAP values cleanly to proportional unit share around P50
        raw_shap = contrib[:, :-1]
        abs_sum = np.maximum(np.sum(np.abs(raw_shap), axis=1, keepdims=True), 1e-6)
        unit_scale = pred_p50[:, None] / abs_sum

        driver_df = pd.DataFrame({
            "Date": test_df["Date"].values,
            "SKU_ID": test_df["SKU_ID"].values,
            "Region": test_df["Region"].values,
            "Product_Family": test_df["Product_Family"].values if "Product_Family" in test_df.columns else "",
            "P50_Forecast": np.round(pred_p50, 2),
        })
        for grp_name, grp_cols in DRIVER_GROUPS.items():
            idxs = [i for i, f in enumerate(feat_names) if f in grp_cols]
            if idxs:
                signed_share = np.sum(raw_shap[:, idxs], axis=1)
                driver_df[grp_name] = np.round(signed_share * unit_scale.ravel(), 2)
            else:
                driver_df[grp_name] = 0.0

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        driver_df.to_parquet(OUTPUTS_DIR / "forecast_drivers.parquet", index=False)

    out_preds = pd.DataFrame({
        "P10": pred_p10_enforced.round(),
        "P50": pred_p50.round(),
        "P90": pred_p90_enforced.round(),
        "Raw_P10": raw_p10_enforced.round(),
        "Raw_P90": raw_p90_enforced.round(),
        "Pred_Tweedie": pred_tw.round(),
        "Pred_Log1p": pred_log.round(),
    }, index=test_df.index)

    metadata = {
        "feature_count": len(X_train.columns),
        "ensemble_weights": {"tweedie": best_w[0], "log1p_l1": best_w[1], "seasonal_anchor": best_w[2]},
        "cqr_margins": {k: round(v, 2) for k, v in cqr_margins.items()},
        "top_features": importances.head(12).to_dict(orient="records"),
        "p50_mean": float(pred_p50.mean()),
        "p10_mean": float(pred_p10_enforced.mean()),
        "p90_mean": float(pred_p90_enforced.mean()),
    }

    return out_preds, metadata
