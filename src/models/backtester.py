"""
Rolling-Origin 19-Month Backtesting Engine (Workstream 4 Advanced & Expert Tier).
Strict zero-leakage evaluation over origins 2023-10, 2024-10, and 2025-03:
- Computes ALL level, seasonality, intermittency, and lag features strictly <= origin_date.
- Tracks WAPE, Bias, RMSE, plus Raw vs Conformal-Calibrated P10-P90 empirical coverage.
- Exports granular breakdowns by Product Family, Region, and Horizon Month (1..19).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from src.config import OUTPUTS_DIR
from src.features.feature_pipeline import attach_strict_causal_features, build_rolling_origin_training_set
from src.models.baselines import forecast_seasonal_naive, forecast_moving_average
from src.models.intermittent import forecast_intermittent_sba
from src.models.lightgbm_engine import train_and_forecast_lgb
from src.models.cold_start import apply_lifecycle_and_cold_start
from src.models.reconciliation import apply_macro_wls_reconciliation


def compute_wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(actual))
    return float(np.sum(np.abs(actual - pred)) / denom * 100.0) if denom > 0 else 0.0


def compute_bias(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(actual))
    return float(np.sum(pred - actual) / denom * 100.0) if denom > 0 else 0.0


def compute_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2))) if len(actual) > 0 else 0.0


def compute_coverage(actual: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    covered = (actual >= p10) & (actual <= p90)
    return float(np.mean(covered) * 100.0)


def compute_pinball_loss(actual: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    if len(actual) == 0:
        return 0.0
    diff = actual - q_pred
    loss = np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
    return float(np.mean(loss))


def compute_winkler_80(actual: np.ndarray, p10: np.ndarray, p90: np.ndarray, alpha: float = 0.20) -> float:
    if len(actual) == 0:
        return 0.0
    width = np.maximum(0.0, p90 - p10)
    penalty_lo = np.where(actual < p10, (2.0 / alpha) * (p10 - actual), 0.0)
    penalty_hi = np.where(actual > p90, (2.0 / alpha) * (actual - p90), 0.0)
    return float(np.mean(width + penalty_lo + penalty_hi))


def run_rolling_origin_backtest(
    panel_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    target_col: str = "Units_Sold_Unconstrained"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes strict leakage-free rolling-origin backtesting across origins 2023-10, 2024-10, and 2025-03.
    """
    all_metrics = []
    all_preds = []
    breakdowns = []

    origins_config = [
        ("2023-10", 19),
        ("2024-10", 19),
        ("2025-03", 19),
    ]

    snapshot_cache = {}
    for origin, horizon in origins_config:
        hist_base = panel_df[panel_df["Date"] <= origin].copy()
        avail_dates = sorted(panel_df[(panel_df["Date"] > origin) & (panel_df["Date"] <= "2026-10")]["Date"].unique())
        eval_dates = avail_dates[:horizon]
        test_base = panel_df[panel_df["Date"].isin(eval_dates)].copy()

        if test_base.empty or hist_base.empty:
            continue

        # 1. Leakage-free causal training snapshots + test features strictly frozen at `origin`
        train_snap = build_rolling_origin_training_set(
            panel_df, sku_master, max_origin_date=origin, target_col=target_col, snapshot_cache=snapshot_cache
        )
        test_slice = attach_strict_causal_features(hist_base, test_base, origin_date=origin, sku_master=sku_master, target_col=target_col)

        # 2. Baselines & Intermittent SBA (fit strictly on hist_base <= origin)
        test_slice["Pred_Seasonal_Naive"] = forecast_seasonal_naive(hist_base, test_slice, target_col=target_col).values
        test_slice["Pred_Moving_Average"] = forecast_moving_average(hist_base, test_slice, window=12, target_col=target_col).values
        test_slice["Pred_Intermittent_SBA"] = forecast_intermittent_sba(hist_base, test_slice, target_col=target_col).values

        # 3. Multi-Paradigm Hybrid Ensemble (Hurdle + LightGBM + HistGB + PyTorch ResNet) + CQR
        lgb_preds, lgb_meta = train_and_forecast_lgb(train_snap, test_slice, target_col=target_col, n_estimators=260, compute_shap=False)
        test_slice["Pred_LightGBM_P10"] = lgb_preds["P10"].values
        test_slice["Pred_LightGBM_P50"] = lgb_preds["P50"].values
        test_slice["Pred_LightGBM_P90"] = lgb_preds["P90"].values
        test_slice["Pred_LightGBM_Raw_P10"] = lgb_preds["Raw_P10"].values
        test_slice["Pred_LightGBM_Raw_P90"] = lgb_preds["Raw_P90"].values
        for col_src, col_dst in [
            ("Pred_Hurdle", "Pred_Hurdle"),
            ("Pred_HistGB", "Pred_HistGB"),
            ("Pred_PyTorch", "Pred_PyTorch"),
            ("Pred_Tweedie", "Pred_Tweedie"),
            ("Pred_LogRatio", "Pred_LogRatio"),
        ]:
            if col_src in lgb_preds.columns:
                test_slice[col_dst] = lgb_preds[col_src].values

        # 4. Enforce catalog lifecycle boundaries + Pair-Norm Top-Down Macro WLS Reconciliation (<= origin)
        test_slice = apply_lifecycle_and_cold_start(test_slice, sku_master, hist_base)
        test_slice = apply_macro_wls_reconciliation(test_slice, hist_base, alpha=0.30, origin_date=origin, target_col=target_col)
        test_slice["Backtest_Origin"] = origin
        all_preds.append(test_slice)

        # 5. Compute overall metrics + calibrated interval coverage + Pinball / Winkler scores
        y_true = pd.to_numeric(test_slice[target_col], errors="coerce").fillna(0.0).values
        cov_raw = compute_coverage(y_true, test_slice["Pred_LightGBM_Raw_P10"].values, test_slice["Pred_LightGBM_Raw_P90"].values)
        cov_cal = compute_coverage(y_true, test_slice["Pred_LightGBM_P10"].values, test_slice["Pred_LightGBM_P90"].values)
        pinball_avg = round((
            compute_pinball_loss(y_true, test_slice["Pred_LightGBM_P10"].values, 0.10)
            + compute_pinball_loss(y_true, test_slice["Pred_LightGBM_P50"].values, 0.50)
            + compute_pinball_loss(y_true, test_slice["Pred_LightGBM_P90"].values, 0.90)
        ) / 3.0, 2)
        winkler_80 = round(compute_winkler_80(y_true, test_slice["Pred_LightGBM_P10"].values, test_slice["Pred_LightGBM_P90"].values), 2)

        models = {
            "Seasonal Naive": test_slice["Pred_Seasonal_Naive"].values,
            "Moving Average": test_slice["Pred_Moving_Average"].values,
            "Intermittent SBA": test_slice["Pred_Intermittent_SBA"].values,
            "HistGB Poisson": test_slice["Pred_HistGB"].values,
            "PyTorch Entity-ResNet": test_slice["Pred_PyTorch"].values,
            "Hurdle Two-Stage": test_slice["Pred_Hurdle"].values,
            "LightGBM Tweedie": test_slice["Pred_Tweedie"].values,
            "LightGBM LogRatio": test_slice["Pred_LogRatio"].values,
            "Grandmaster Ensemble P50": test_slice["Pred_LightGBM_P50"].values,
        }

        for model_name, y_pred in models.items():
            is_hero = model_name == "Grandmaster Ensemble P50"
            all_metrics.append({
                "Origin": origin,
                "Model": model_name,
                "Horizon_Months": len(eval_dates),
                "WAPE_Pct": round(compute_wape(y_true, y_pred), 2),
                "Bias_Pct": round(compute_bias(y_true, y_pred), 2),
                "RMSE": round(compute_rmse(y_true, y_pred), 2),
                "Raw_Coverage_Pct": round(cov_raw, 1) if is_hero else np.nan,
                "Calibrated_Coverage_Pct": round(cov_cal, 1) if is_hero else np.nan,
                "Pinball_Loss_Avg": pinball_avg if is_hero else round(compute_pinball_loss(y_true, y_pred, 0.50), 2),
                "Winkler_80_Score": winkler_80 if is_hero else np.nan,
            })

        ens_w = compute_wape(y_true, test_slice["Pred_LightGBM_P50"].values)
        ens_b = compute_bias(y_true, test_slice["Pred_LightGBM_P50"].values)
        print(f"    Origin {origin}: Ensemble WAPE={ens_w:.2f}% | Bias={ens_b:+.2f}% | CQR Cov={cov_cal:.1f}% | Pinball={pinball_avg:.2f}", flush=True)

        # 6. Granular breakdowns (Family, Region, Horizon) for Dashboard Accuracy View
        for grp_type, grp_col in [("Product_Family", "Product_Family"), ("Region", "Region"), ("Horizon", "horizon_months")]:
            for grp_val, grp_df in test_slice.groupby(grp_col):
                ya = pd.to_numeric(grp_df[target_col], errors="coerce").fillna(0.0).values
                yp_lgb = grp_df["Pred_LightGBM_P50"].values
                yp_sn = grp_df["Pred_Seasonal_Naive"].values
                cov_g = compute_coverage(ya, grp_df["Pred_LightGBM_P10"].values, grp_df["Pred_LightGBM_P90"].values)
                breakdowns.append({
                    "Origin": origin,
                    "Breakdown_Type": grp_type,
                    "Group": str(grp_val),
                    "LightGBM_WAPE": round(compute_wape(ya, yp_lgb), 2),
                    "LightGBM_Bias": round(compute_bias(ya, yp_lgb), 2),
                    "Seasonal_Naive_WAPE": round(compute_wape(ya, yp_sn), 2),
                    "Coverage_Pct": round(cov_g, 1),
                    "Actual_Units": int(np.sum(ya)),
                    "Forecast_Units": int(np.sum(yp_lgb)),
                })

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    breakdowns_df = pd.DataFrame(breakdowns)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUTS_DIR / "backtest_metrics.csv", index=False)
    breakdowns_df.to_csv(OUTPUTS_DIR / "backtest_breakdowns.csv", index=False)
    preds_df.to_csv(OUTPUTS_DIR / "backtest_predictions.csv", index=False)

    return metrics_df, preds_df
