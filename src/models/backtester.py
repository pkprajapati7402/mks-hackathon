"""
Rolling-Origin 19-Month Backtesting Engine (Workstream 4 Advanced Tier).
Strict leakage-free time-series cross-validation over origins 2023-10, 2024-10, and 2025-03.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from src.config import BACKTEST_ORIGINS, OUTPUTS_DIR, REPORTS_DIR
from src.models.baselines import forecast_seasonal_naive, forecast_moving_average
from src.models.intermittent import forecast_intermittent_sba
from src.models.lightgbm_engine import train_and_forecast_lgb
from src.models.cold_start import apply_lifecycle_and_cold_start

def compute_wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Computes Weighted Absolute Percentage Error (WAPE)."""
    denom = np.sum(actual)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom * 100.0)

def compute_bias(actual: np.ndarray, pred: np.ndarray) -> float:
    """Computes Forecast Bias percentage (+ overforecast, - underforecast)."""
    denom = np.sum(actual)
    if denom == 0:
        return 0.0
    return float(np.sum(pred - actual) / denom * 100.0)

def compute_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((actual - pred) ** 2)))

def run_rolling_origin_backtest(
    panel_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    target_col: str = "Units_Sold_Unconstrained"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes rolling-origin backtesting across origins 2023-10, 2024-10, and 2025-03.
    Evaluates 19-month horizons strictly without future lookahead.
    """
    all_metrics = []
    all_preds = []

    origins_config = [
        ("2023-10", 19),  # 2023-11 to 2025-05
        ("2024-10", 19),  # 2024-11 to 2026-05
        ("2025-03", 19)   # 2025-04 to 2026-10 (holdout)
    ]

    for origin, horizon in origins_config:
        # Split train (<= origin) and test (> origin up to origin + 19 months)
        train_slice = panel_df[panel_df["Date"] <= origin].copy()
        
        # Determine evaluation dates
        avail_dates = sorted(panel_df[panel_df["Date"] > origin]["Date"].unique())
        eval_dates = avail_dates[:horizon]
        test_slice = panel_df[panel_df["Date"].isin(eval_dates)].copy()

        if test_slice.empty:
            continue

        # 1. Baseline 1: Seasonal Naive
        pred_naive = forecast_seasonal_naive(train_slice, test_slice, target_col=target_col)
        test_slice["Pred_Seasonal_Naive"] = pred_naive.values

        # 2. Baseline 2: Moving Average
        pred_ma = forecast_moving_average(train_slice, test_slice, window=12, target_col=target_col)
        test_slice["Pred_Moving_Average"] = pred_ma.values

        # 3. Intermittent: SBA
        pred_sba = forecast_intermittent_sba(train_slice, test_slice, target_col=target_col)
        test_slice["Pred_Intermittent_SBA"] = pred_sba.values

        # 4. Machine Learning: LightGBM (Tweedie P50, P10, P90)
        lgb_preds, _ = train_and_forecast_lgb(train_slice, test_slice, target_col=target_col)
        test_slice["Pred_LightGBM_P10"] = lgb_preds["P10"].values
        test_slice["Pred_LightGBM_P50"] = lgb_preds["P50"].values
        test_slice["Pred_LightGBM_P90"] = lgb_preds["P90"].values

        # Apply lifecycle boundaries
        test_slice = apply_lifecycle_and_cold_start(test_slice, sku_master, train_slice)

        test_slice["Backtest_Origin"] = origin
        all_preds.append(test_slice)

        # Compute metrics per model for this origin
        y_true = test_slice[target_col].fillna(0.0).values
        models = {
            "Seasonal Naive": test_slice["Pred_Seasonal_Naive"].values,
            "Moving Average": test_slice["Pred_Moving_Average"].values,
            "Intermittent SBA": test_slice["Pred_Intermittent_SBA"].values,
            "LightGBM P50": test_slice["Pred_LightGBM_P50"].values
        }

        for model_name, y_pred in models.items():
            wape = compute_wape(y_true, y_pred)
            bias = compute_bias(y_true, y_pred)
            rmse = compute_rmse(y_true, y_pred)

            all_metrics.append({
                "Origin": origin,
                "Model": model_name,
                "Horizon_Months": len(eval_dates),
                "WAPE_Pct": round(wape, 2),
                "Bias_Pct": round(bias, 2),
                "RMSE": round(rmse, 2)
            })

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUTS_DIR / "backtest_metrics.csv", index=False)
    preds_df.to_csv(OUTPUTS_DIR / "backtest_predictions.csv", index=False)

    return metrics_df, preds_df
