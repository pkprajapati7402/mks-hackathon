"""
Demand Intermittency Classification Engine (Workstream 3).
Computes Syntetos-Boylan ADI and CV2 metrics and categorizes time series into:
Smooth, Intermittent, Erratic, or Lumpy.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.config import ADI_THRESHOLD, CV2_THRESHOLD, REPORTS_DIR

def classify_series_intermittency(series: pd.Series) -> Dict[str, Any]:
    """
    Computes ADI, CV2, and intermittency class for a demand series.
    - ADI (Average Demand Interval): Total non-empty periods / non-zero count
    - CV2 (Square of Coefficient of Variation): (std / mean)^2 for non-zero demand
    """
    s = series.dropna()
    total_periods = len(s)
    non_zeros = s[s > 0]
    n_nonzero = len(non_zeros)

    if n_nonzero == 0:
        return {
            "adi": 99.0,
            "cv2": 0.0,
            "category": "Intermittent",
            "croston_rate": 0.0,
            "sba_rate": 0.0
        }

    # ADI
    adi = float(total_periods) / float(n_nonzero)

    # CV2
    mean_val = float(non_zeros.mean())
    std_val = float(non_zeros.std()) if n_nonzero > 1 else 0.0
    cv2 = float((std_val / mean_val) ** 2) if mean_val > 0 else 0.0

    # Categorization per Syntetos-Boylan / ADI-CV2 quadrant
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        cat = "Smooth"
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        cat = "Intermittent"
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        cat = "Erratic"
    else:
        cat = "Lumpy"

    # Croston and SBA baseline demand rates
    croston_rate = mean_val / max(1.0, adi)
    sba_rate = croston_rate * (1.0 - cv2 / (2.0 * max(1.0, adi)))

    return {
        "adi": round(adi, 3),
        "cv2": round(cv2, 3),
        "category": cat,
        "croston_rate": round(max(0.0, croston_rate), 2),
        "sba_rate": round(max(0.0, sba_rate), 2)
    }

def compute_dataset_intermittency(df: pd.DataFrame, target_col: str = "Units_Sold_Unconstrained") -> pd.DataFrame:
    """
    Computes intermittency classification for every SKU-Region series in the portfolio.
    Saves summary to reports/intermittency_classification.csv.
    """
    results = []
    for (sku, reg), grp in df.groupby(["SKU_ID", "Region"]):
        fam = grp["Product_Family"].iloc[0]
        bu = grp["Business_Unit"].iloc[0]
        stats = classify_series_intermittency(grp[target_col])
        results.append({
            "SKU_ID": sku,
            "Region": reg,
            "Product_Family": fam,
            "Business_Unit": bu,
            "ADI": stats["adi"],
            "CV2": stats["cv2"],
            "Category": stats["category"],
            "Croston_Rate": stats["croston_rate"],
            "SBA_Rate": stats["sba_rate"]
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORTS_DIR / "intermittency_classification.csv", index=False)
    return res_df
