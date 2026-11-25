"""
Cold Start, Predecessor Transfer Learning, Smooth Launch Ramp, and Catalog Lifecycle Boundaries (Workstream 4 Expert Tier).
Enforces EOL cutoffs, Planned Launch 3-month S-curve ramp-ups, Replaces_SKU predecessor transfers,
and strict quantile monotonicity 0 <= P10 <= P50 <= P90.
"""

import numpy as np
import pandas as pd


def _ym_diff(date_a: pd.Series, date_b: pd.Series) -> np.ndarray:
    """Compute signed month difference date_a - date_b for YYYY-MM strings."""
    ya = pd.to_numeric(date_a.astype(str).str[:4], errors="coerce").fillna(2019).astype(int)
    ma = pd.to_numeric(date_a.astype(str).str[5:7], errors="coerce").fillna(1).astype(int)
    yb = pd.to_numeric(date_b.astype(str).str[:4], errors="coerce").fillna(2019).astype(int)
    mb = pd.to_numeric(date_b.astype(str).str[5:7], errors="coerce").fillna(1).astype(int)
    return ((ya - yb) * 12 + (ma - mb)).values


def apply_lifecycle_and_cold_start(
    forecast_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    history_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    Applies catalog lifecycle constraints and predecessor transfer learning:
    1. Zeroes forecasts after EOL_Date and prior to Launch_Date.
    2. For cold-start SKUs with zero/no training history, transfers demand from Replaces_SKU
       with a smooth 3-month ramp-up (0.45 -> 0.75 -> 1.00) from Launch_Date.
    3. Re-enforces strict 0 <= P10 <= P50 <= P90 monotonicity across all columns.
    """
    out = forecast_df.copy()

    sku_launch = dict(zip(sku_master["SKU_ID"], sku_master["Launch_Date"]))
    sku_eol = dict(zip(sku_master["SKU_ID"], sku_master["EOL_Date"]))
    sku_replaces = dict(zip(sku_master["SKU_ID"], sku_master["Replaces_SKU"]))

    launch_series = out["SKU_ID"].map(sku_launch).fillna("").astype(str).str[:7]
    eol_series = out["SKU_ID"].map(sku_eol).fillna("").astype(str).str[:7]

    cols_to_adjust = [
        c for c in [
            "P10", "P50", "P90", "Units_Sold_Forecast",
            "Pred_LightGBM_P10", "Pred_LightGBM_P50", "Pred_LightGBM_P90",
        ] if c in out.columns
    ]

    # 1. Enforce Launch & EOL Boundaries (vectorized)
    has_launch = (launch_series != "") & (launch_series != "nan")
    has_eol = (eol_series != "") & (eol_series != "nan")
    violates_launch = has_launch & (out["Date"] < launch_series)
    violates_eol = has_eol & (out["Date"] > eol_series)
    zero_mask = violates_launch | violates_eol

    for c in cols_to_adjust:
        out.loc[zero_mask, c] = 0.0

    # 2. Cold Start via Replaces_SKU with smooth 3-month launch ramp
    hist_valid = history_agg[history_agg["Units_Sold_Unconstrained"] > 0]
    hist_skus = set(hist_valid["SKU_ID"].unique())
    cold_skus = set(out["SKU_ID"].unique()) - hist_skus

    if cold_skus:
        pred_medians = hist_valid.groupby(["SKU_ID", "Region"])["Units_Sold_Unconstrained"].median().to_dict()
        fam_medians = (
            hist_valid.groupby("Product_Family")["Units_Sold_Unconstrained"].median().to_dict()
            if "Product_Family" in hist_valid.columns else {}
        )

        is_cold = out["SKU_ID"].isin(cold_skus)
        rep_skus = out["SKU_ID"].map(sku_replaces).fillna("").astype(str)

        # Ramp-up multiplier on launch month (m=0 -> 0.45, m=1 -> 0.75, m>=2 -> 1.0)
        months_live = np.where(has_launch, _ym_diff(out["Date"], launch_series), 99)
        ramp_mult = np.where(months_live == 0, 0.45, np.where(months_live == 1, 0.75, 1.0))

        rep_vals = np.array([
            pred_medians.get((rep.strip(), reg), fam_medians.get(fam, 15.0))
            if rep and rep != "nan" and rep.strip()
            else fam_medians.get(fam, 15.0)
            for rep, reg, fam in zip(rep_skus, out["Region"], out.get("Product_Family", [""] * len(out)))
        ], dtype=float) * ramp_mult

        cold_rep_mask = is_cold & (rep_vals > 0) & (~zero_mask)

        for p50_col, p10_col, p90_col in [
            ("P50", "P10", "P90"),
            ("Pred_LightGBM_P50", "Pred_LightGBM_P10", "Pred_LightGBM_P90"),
        ]:
            if p50_col in out.columns:
                needs_fill = cold_rep_mask & (out[p50_col] <= 1.0)
                out.loc[needs_fill, p50_col] = np.round(rep_vals[needs_fill])
                if p10_col in out.columns:
                    out.loc[needs_fill, p10_col] = np.round(rep_vals[needs_fill] * 0.65)
                if p90_col in out.columns:
                    out.loc[needs_fill, p90_col] = np.round(rep_vals[needs_fill] * 1.40)

        if "Units_Sold_Forecast" in out.columns:
            needs_fcst_fill = cold_rep_mask & (out["Units_Sold_Forecast"] <= 1.0)
            out.loc[needs_fcst_fill, "Units_Sold_Forecast"] = np.round(rep_vals[needs_fcst_fill])

    # 3. Enforce strict quantile monotonicity 0 <= P10 <= P50 <= P90
    for p50_col, p10_col, p90_col in [
        ("P50", "P10", "P90"),
        ("Pred_LightGBM_P50", "Pred_LightGBM_P10", "Pred_LightGBM_P90"),
    ]:
        if p50_col in out.columns:
            out[p50_col] = np.clip(out[p50_col], 0.0, None)
            if p10_col in out.columns:
                out[p10_col] = np.clip(np.minimum(out[p10_col], out[p50_col]), 0.0, None)
            if p90_col in out.columns:
                out[p90_col] = np.maximum(out[p90_col], out[p50_col])

    return out
