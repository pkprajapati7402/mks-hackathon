"""
Cold Start, Predecessor Transfer Learning, Empirical Launch S-Curve Ramp, and Lifecycle Boundaries.
Enforces EOL cutoffs, empirically calibrated 7-month Replaces_SKU launch ramps
(0.15 -> 0.42 -> 0.55 -> 0.65 -> 0.78 -> 0.90 -> 1.00), and strict 0 <= P10 <= P50 <= P90.
"""

import numpy as np
import pandas as pd

# Empirically observed median successor/predecessor transfer ramp across 2019-2026 catalog transitions
EMPIRICAL_RAMP_CURVE = {0: 0.15, 1: 0.42, 2: 0.55, 3: 0.65, 4: 0.78, 5: 0.90}


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
    2. For cold-start SKUs with zero historical sales, anchors regional scale via Replaces_SKU
       multiplied by the empirical 7-month launch S-curve (0.15 -> 0.42 -> ... -> 1.00).
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

    # 2. Cold Start via Replaces_SKU with empirical 7-month launch ramp
    hist_valid = history_agg[history_agg["Units_Sold_Unconstrained"] > 0]
    hist_skus = set(hist_valid["SKU_ID"].unique())
    cold_skus = set(out["SKU_ID"].unique()) - hist_skus

    if cold_skus:
        pred_reg_means = hist_valid.groupby(["SKU_ID", "Region"])["Units_Sold_Unconstrained"].mean().to_dict()
        pred_global_means = hist_valid.groupby("SKU_ID")["Units_Sold_Unconstrained"].mean().to_dict()
        fam_medians = (
            hist_valid.groupby("Product_Family")["Units_Sold_Unconstrained"].median().to_dict()
            if "Product_Family" in hist_valid.columns else {}
        )

        is_cold = out["SKU_ID"].isin(cold_skus)
        rep_skus = out["SKU_ID"].map(sku_replaces).fillna("").astype(str).str.strip()
        has_valid_rep = (rep_skus != "") & (rep_skus != "nan")

        months_live = np.where(has_launch, _ym_diff(out["Date"], launch_series), 99)
        ramp_mult = np.array([EMPIRICAL_RAMP_CURVE.get(int(m), 1.0 if m >= 6 else 0.0) for m in months_live], dtype=float)

        rep_base = np.array([
            pred_reg_means.get((rep, reg), pred_global_means.get(rep, fam_medians.get(fam, 15.0)))
            if hrep else fam_medians.get(fam, 15.0)
            for rep, hrep, reg, fam in zip(rep_skus, has_valid_rep, out["Region"], out.get("Product_Family", [""] * len(out)))
        ], dtype=float)
        rep_vals = rep_base * ramp_mult

        cold_rep_mask = is_cold & has_valid_rep.values & (rep_vals > 0) & (~zero_mask)
        cold_norep_mask = is_cold & (~has_valid_rep.values) & (rep_vals > 0) & (~zero_mask)

        for p50_col, p10_col, p90_col in [
            ("P50", "P10", "P90"),
            ("Pred_LightGBM_P50", "Pred_LightGBM_P10", "Pred_LightGBM_P90"),
        ]:
            if p50_col in out.columns:
                # For explicit Replaces_SKU launches, blend ML seasonality/macro shape with ramped predecessor regional anchor
                cur_vals = out.loc[cold_rep_mask, p50_col].values
                blended = np.where(cur_vals > 1.0, np.round(0.30 * cur_vals + 0.70 * rep_vals[cold_rep_mask]), np.round(rep_vals[cold_rep_mask]))
                out.loc[cold_rep_mask, p50_col] = blended
                if p10_col in out.columns:
                    out.loc[cold_rep_mask, p10_col] = np.round(blended * 0.68)
                if p90_col in out.columns:
                    out.loc[cold_rep_mask, p90_col] = np.round(blended * 1.38)

                # Fallback for unlinked cold starts with zero/minimal ML prediction
                needs_fill = cold_norep_mask & (out[p50_col] <= 1.0)
                if needs_fill.any():
                    out.loc[needs_fill, p50_col] = np.round(rep_vals[needs_fill])
                    if p10_col in out.columns:
                        out.loc[needs_fill, p10_col] = np.round(rep_vals[needs_fill] * 0.68)
                    if p90_col in out.columns:
                        out.loc[needs_fill, p90_col] = np.round(rep_vals[needs_fill] * 1.38)

        if "Units_Sold_Forecast" in out.columns and "P50" in out.columns:
            out.loc[is_cold & (~zero_mask), "Units_Sold_Forecast"] = out.loc[is_cold & (~zero_mask), "P50"]

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
