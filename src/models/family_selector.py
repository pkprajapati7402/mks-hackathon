"""
Per-Family Model Selection Engine (Workstream 4 Advanced Tier).
Selects champion model minimizing selection-origin WAPE per Product Family, verifies on holdout 2025-03,
and outputs actionable champion routing table consumed by Stage 5 forecasting.
"""

import pandas as pd
from src.config import OUTPUTS_DIR, CANONICAL_PRODUCT_FAMILIES
from src.models.backtester import compute_wape


def select_best_model_per_family(backtest_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates candidate models across selection origins (2023-10, 2024-10) per Product Family,
    checks generalization on holdout (2025-03), and writes family_model_selection.csv.
    """
    candidates = [
        ("Seasonal Naive", "Pred_Seasonal_Naive"),
        ("Moving Average", "Pred_Moving_Average"),
        ("Intermittent SBA", "Pred_Intermittent_SBA"),
        ("HistGB Poisson", "Pred_HistGB"),
        ("PyTorch Entity-ResNet", "Pred_PyTorch"),
        ("Hurdle Two-Stage", "Pred_Hurdle"),
        ("Grandmaster Ensemble P50", "Pred_LightGBM_P50"),
    ]

    selection_df = backtest_preds[backtest_preds["Backtest_Origin"].isin(["2023-10", "2024-10"])].copy()
    holdout_df = backtest_preds[backtest_preds["Backtest_Origin"] == "2025-03"].copy()

    results = []
    for fam in CANONICAL_PRODUCT_FAMILIES:
        fam_sel = selection_df[selection_df["Product_Family"] == fam]
        fam_hold = holdout_df[holdout_df["Product_Family"] == fam]
        if fam_sel.empty:
            continue

        y_sel = fam_sel["Units_Sold_Unconstrained"].values
        best_model = None
        best_wape = float("inf")
        scores = {}

        for m_name, pred_col in candidates:
            if pred_col in fam_sel.columns:
                w = compute_wape(y_sel, fam_sel[pred_col].values)
                scores[m_name] = round(w, 2)
                if w < best_wape:
                    best_wape = w
                    best_model = m_name

        hold_wape = 0.0
        if not fam_hold.empty and best_model:
            col_match = [c[1] for c in candidates if c[0] == best_model][0]
            hold_wape = compute_wape(fam_hold["Units_Sold_Unconstrained"].values, fam_hold[col_match].values)

        results.append({
            "Product_Family": fam,
            "Selected_Champion_Model": best_model,
            "Selection_WAPE_Pct": round(best_wape, 2),
            "Holdout_WAPE_Pct": round(hold_wape, 2),
            "Seasonal_Naive_WAPE": scores.get("Seasonal Naive", 0.0),
            "Moving_Average_WAPE": scores.get("Moving Average", 0.0),
            "Intermittent_SBA_WAPE": scores.get("Intermittent SBA", 0.0),
            "HistGB_WAPE": scores.get("HistGB Poisson", 0.0),
            "PyTorch_ResNet_WAPE": scores.get("PyTorch Entity-ResNet", 0.0),
            "Hurdle_WAPE": scores.get("Hurdle Two-Stage", 0.0),
            "Ensemble_P50_WAPE": scores.get("Grandmaster Ensemble P50", 0.0),
            "LightGBM_P50_WAPE": scores.get("Grandmaster Ensemble P50", 0.0),
        })

    selection_report = pd.DataFrame(results)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    selection_report.to_csv(OUTPUTS_DIR / "family_model_selection.csv", index=False)
    return selection_report
