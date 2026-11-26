"""
Hierarchical Reconciliation, ABC-XYZ Inventory Optimizer & Submission Engine (Workstream 4 Expert Tier).
- Enforces exact bottom-up additivity across SKU -> Product Family -> Business Unit -> Region -> Global.
- Enforces strict 0 <= P10 <= P50 <= P90 integer bounds.
- Computes Portfolio ABC-XYZ Value x Variability Optimization & Working Capital table for the Control Tower.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.config import OUTPUTS_DIR, PROJECT_ROOT, CLEANED_DATA_DIR


def apply_macro_wls_reconciliation(
    forecast_df: pd.DataFrame,
    history_agg: pd.DataFrame,
    alpha: float = 0.30,
    origin_date: str = None,
    target_col: str = "Units_Sold_Unconstrained",
) -> pd.DataFrame:
    """
    Applies strictly causal Pair-Normalized Top-Down Family Macro WLS Reconciliation.
    Fits per-family regularized Ridge models on log1p(mean demand per active SKU-Region pair)
    strictly <= origin_date (excluding acute COVID 2020-03..2020-05), and pulls active positive
    leaf forecasts smoothly toward top-down macro totals while keeping zeros intact.
    """
    if alpha <= 0.0 or forecast_df.empty or history_agg.empty:
        return forecast_df

    out = forecast_df.copy()
    hist = history_agg.copy()
    if origin_date is not None:
        hist = hist[hist["Date"] <= origin_date]
    hist = hist[~hist["Date"].astype(str).between("2020-03", "2020-05")].copy()
    if hist.empty:
        return out

    macro_cols = [
        "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "New_Vehicle_Orders_Index",
        "Industrial_Util_Rate_Pct", "Raw_Material_Cost_Index", "EV_Adoption_Index",
    ]
    # Load cleaned macro if absent from history_agg or forecast_df
    macro_df = None
    if any(c not in hist.columns or c not in out.columns for c in macro_cols):
        m_path = CLEANED_DATA_DIR / "macro_indicators.csv"
        if m_path.exists():
            macro_df = pd.read_csv(m_path)

    hist["_y"] = pd.to_numeric(hist[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    fam_hist = hist.groupby(["Date", "Product_Family"]).agg(y=("_y", "sum"), cnt=("_y", "count")).reset_index()
    fam_hist["mean_per_pair"] = fam_hist["y"] / fam_hist["cnt"].clip(lower=1)

    avail_m = [c for c in macro_cols if c in hist.columns]
    if macro_df is not None:
        fam_hist = fam_hist.merge(macro_df[["Date"] + [c for c in macro_cols if c in macro_df.columns]], on="Date", how="left")
    elif avail_m:
        m_lut = hist[["Date"] + avail_m].drop_duplicates(subset=["Date"])
        fam_hist = fam_hist.merge(m_lut, on="Date", how="left")

    fam_hist["Month"] = fam_hist["Date"].astype(str).str[5:7].astype(int)
    fam_hist["t_idx"] = fam_hist["Date"].astype(str).str[:4].astype(int) * 12 + fam_hist["Month"] - (2019 * 12 + 1)

    fut_cnt = out.groupby(["Date", "Product_Family"])["SKU_ID"].count().reset_index(name="fut_cnt")
    if macro_df is not None:
        fh_fut = fut_cnt.merge(macro_df[["Date"] + [c for c in macro_cols if c in macro_df.columns]], on="Date", how="left")
    elif avail_m:
        m_out = out[["Date"] + [c for c in avail_m if c in out.columns]].drop_duplicates(subset=["Date"])
        fh_fut = fut_cnt.merge(m_out, on="Date", how="left")
    else:
        fh_fut = fut_cnt.copy()
    fh_fut["Month"] = fh_fut["Date"].astype(str).str[5:7].astype(int)

    x_macro = [c for c in macro_cols[:-1] if c in fam_hist.columns and c in fh_fut.columns]

    def _design(df_in: pd.DataFrame) -> np.ndarray:
        m_ohe = np.column_stack([(df_in["Month"].values == m).astype(float) for m in range(1, 13)])
        ev_raw = df_in["EV_Adoption_Index"].fillna(80.0).values if "EV_Adoption_Index" in df_in.columns else np.full(len(df_in), 80.0)
        ev_s = np.clip((ev_raw - 40.0) / 55.0, 0.0, 1.2)
        if x_macro:
            mac_x = df_in[x_macro].fillna(0.0).values / 100.0
            return np.column_stack([m_ohe, ev_s, mac_x])
        return np.column_stack([m_ohe, ev_s])

    top_preds = {}
    for fam in fh_fut["Product_Family"].unique():
        trf = fam_hist[fam_hist["Product_Family"] == fam].sort_values("Date").tail(48)
        fuf = fh_fut[fh_fut["Product_Family"] == fam]
        if len(trf) < 12 or fuf.empty:
            continue
        reg = Ridge(alpha=15.0)
        sw = np.exp(0.03 * (trf["t_idx"].values - trf["t_idx"].max()))
        reg.fit(_design(trf), np.log1p(trf["mean_per_pair"].values), sample_weight=sw)
        tp = np.clip(np.expm1(reg.predict(_design(fuf))), 1.0, None) * fuf["fut_cnt"].values
        for d, v in zip(fuf["Date"].values, tp):
            top_preds[(d, fam)] = float(v)

    for p50_col, p10_col, p90_col in [
        ("Pred_LightGBM_P50", "Pred_LightGBM_P10", "Pred_LightGBM_P90"),
        ("P50", "P10", "P90"),
    ]:
        if p50_col not in out.columns:
            continue
        bot_sum = out.groupby(["Date", "Product_Family"])[p50_col].transform("sum").clip(lower=1.0).values
        top_tgt = np.array([top_preds.get((d, f), bs) for d, f, bs in zip(out["Date"], out["Product_Family"], bot_sum)], dtype=float)
        scale = np.clip((1.0 - alpha) + alpha * (top_tgt / bot_sum), 0.78, 1.22)

        active = out[p50_col].values > 0
        p50_new = np.where(active, np.round(out[p50_col].values * scale), 0.0)
        out[p50_col] = p50_new
        if p10_col in out.columns:
            p10_new = np.where(active, np.round(out[p10_col].values * scale), 0.0)
            out[p10_col] = np.clip(np.minimum(p10_new, p50_new), 0.0, None)
        if p90_col in out.columns:
            p90_new = np.where(active, np.round(out[p90_col].values * scale), 0.0)
            out[p90_col] = np.maximum(p90_new, p50_new)

    if "Units_Sold_Forecast" in out.columns and "P50" in out.columns:
        out["Units_Sold_Forecast"] = out["P50"]

    return out


def compute_abc_xyz_matrix(
    forecast_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    unit_prices: pd.Series = None,
    output_dir: Path = OUTPUTS_DIR,
) -> pd.DataFrame:
    """
    Classifies every active SKU into the 9-cell ABC-XYZ matrix:
    - ABC: Forecast Revenue share (A=top 80%, B=next 15%, C=bottom 5%)
    - XYZ: Demand Interval Spread ratio (P90 - P10) / max(P50, 1) (X <= 0.65 stable, Y <= 1.25 variable, Z > 1.25 volatile)
    Assigns optimal target Customer Service Level (98% / 95% / 90%) & Newsvendor safety buffer USD.
    """
    agg = forecast_df.groupby(["SKU_ID", "Product_Family"]).agg(
        P10=("P10", "sum"),
        P50=("Units_Sold_Forecast", "sum"),
        P90=("P90", "sum"),
    ).reset_index()

    if unit_prices is not None and not unit_prices.empty:
        price_map = unit_prices.to_dict()
        agg["Unit_Price_USD"] = agg["SKU_ID"].map(price_map).fillna(120.0)
    else:
        agg["Unit_Price_USD"] = 120.0

    agg["Forecast_Revenue_USD"] = (agg["P50"] * agg["Unit_Price_USD"]).round(2)
    agg = agg.sort_values("Forecast_Revenue_USD", ascending=False).reset_index(drop=True)
    tot_rev = max(1.0, float(agg["Forecast_Revenue_USD"].sum()))
    agg["Cum_Rev_Pct"] = (agg["Forecast_Revenue_USD"].cumsum() / tot_rev) * 100.0

    agg["ABC_Class"] = np.where(agg["Cum_Rev_Pct"] <= 80.0, "A", np.where(agg["Cum_Rev_Pct"] <= 95.0, "B", "C"))
    spread = (agg["P90"] - agg["P10"]) / np.maximum(agg["P50"], 1.0)
    agg["Interval_Spread_Ratio"] = spread.round(3)
    # Self-calibrating portfolio variability tiers based on empirical P10-P90 spread quantiles (35% X, 40% Y, 25% Z)
    q_x = float(np.quantile(spread, 0.35)) if len(spread) > 0 else 0.70
    q_y = float(np.quantile(spread, 0.75)) if len(spread) > 0 else 0.80
    agg["XYZ_Class"] = np.where(spread <= q_x, "X", np.where(spread <= q_y, "Y", "Z"))
    agg["ABC_XYZ_Segment"] = agg["ABC_Class"] + agg["XYZ_Class"]

    # Policy Service Level & Z-score for empirical sigma = (P90 - P10) / 2.563
    csl_map = {"A": 0.98, "B": 0.95, "C": 0.90}
    z_map = {"A": 2.054, "B": 1.645, "C": 1.282}
    agg["Target_Service_Level"] = agg["ABC_Class"].map(csl_map)
    sigma_est = np.maximum(0.0, (agg["P90"] - agg["P10"]) / 2.563)
    z_vals = agg["ABC_Class"].map(z_map)
    # 19-month annualized monthly safety buffer
    agg["Recommended_Safety_Stock_Units"] = np.round((z_vals * sigma_est) / np.sqrt(19.0)).astype(int)
    agg["Cycle_Stock_Units"] = np.round(agg["P50"] / 19.0 * 0.5).astype(int)
    agg["Working_Capital_USD"] = np.round(
        (agg["Recommended_Safety_Stock_Units"] + agg["Cycle_Stock_Units"]) * agg["Unit_Price_USD"] * 0.65, 2
    )

    if sku_master is not None and not sku_master.empty and "Powertrain_Application" in sku_master.columns:
        pt_map = dict(zip(sku_master["SKU_ID"], sku_master["Powertrain_Application"]))
        agg["Powertrain_Application"] = agg["SKU_ID"].map(pt_map).fillna("Universal")

    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_dir / "abc_xyz_inventory.csv", index=False)
    return agg


def reconcile_and_generate_submission(
    test_features: pd.DataFrame,
    lgb_preds: pd.DataFrame,
    test_raw: pd.DataFrame,
    sku_master: pd.DataFrame,
    sku_prices: pd.Series = None,
    output_dir: Path = OUTPUTS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs hierarchical bottom-up reconciliation, writes submission.csv (58,219 rows),
    forecast_intervals.parquet (enriched for Dashboard), and ABC-XYZ working capital matrix.
    """
    out_forecast = test_features[["Date", "SKU_ID", "Region", "Product_Family", "Business_Unit"]].copy()

    out_forecast["P10"] = np.clip(np.round(lgb_preds["P10"].values), 0, None).astype(int)
    out_forecast["P50"] = np.clip(np.round(lgb_preds["P50"].values), 0, None).astype(int)
    out_forecast["P90"] = np.clip(np.round(lgb_preds["P90"].values), 0, None).astype(int)
    out_forecast["P10"] = np.minimum(out_forecast["P10"], out_forecast["P50"])
    out_forecast["P90"] = np.maximum(out_forecast["P90"], out_forecast["P50"])
    out_forecast["Units_Sold_Forecast"] = out_forecast["P50"]

    # 1. Exact Bottom-Up Reconciliation Check across BU, Family, Region, Portfolio
    bu_sum = out_forecast.groupby(["Date", "Business_Unit"])["Units_Sold_Forecast"].sum().reset_index(name="BU_Sum")
    fam_sum = out_forecast.groupby(["Date", "Product_Family"])["Units_Sold_Forecast"].sum().reset_index(name="Family_Sum")
    reg_sum = out_forecast.groupby(["Date", "Region"])["Units_Sold_Forecast"].sum().reset_index(name="Region_Sum")

    total_by_bu = bu_sum.groupby("Date")["BU_Sum"].sum().reset_index(name="Total_By_BU")
    total_by_fam = fam_sum.groupby("Date")["Family_Sum"].sum().reset_index(name="Total_By_Family")
    total_by_reg = reg_sum.groupby("Date")["Region_Sum"].sum().reset_index(name="Total_By_Region")

    recon_check = total_by_bu.merge(total_by_fam, on="Date").merge(total_by_reg, on="Date")
    recon_check["Discrepancy"] = (
        (recon_check["Total_By_BU"] - recon_check["Total_By_Region"]).abs()
        + (recon_check["Total_By_Family"] - recon_check["Total_By_Region"]).abs()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_check.to_csv(output_dir / "hierarchical_reconciliation.csv", index=False)

    # 2. Align predictions strictly to student_data/test.csv order and carry hierarchy/price/powertrain metadata
    meta_cols = [c for c in ["Date", "SKU_ID", "Region", "Product_Family", "Business_Unit", "Customer_Name", "Sales_Channel"] if c in test_raw.columns]
    merged_sub = test_raw[meta_cols].merge(
        out_forecast[["Date", "SKU_ID", "Region", "Units_Sold_Forecast", "P10", "P50", "P90"]],
        on=["Date", "SKU_ID", "Region"],
        how="left",
    )

    if sku_master is not None and not sku_master.empty and "Powertrain_Application" in sku_master.columns:
        sku_to_pt = dict(zip(sku_master["SKU_ID"], sku_master["Powertrain_Application"]))
        merged_sub["Powertrain_Application"] = merged_sub["SKU_ID"].map(sku_to_pt).fillna("Universal")

    if sku_prices is not None and not sku_prices.empty:
        price_map = sku_prices.to_dict()
        merged_sub["Unit_Price_USD"] = merged_sub["SKU_ID"].map(price_map).fillna(120.0).round(2)
    else:
        merged_sub["Unit_Price_USD"] = 120.0

    for c in ["Units_Sold_Forecast", "P10", "P50", "P90"]:
        merged_sub[c] = merged_sub[c].fillna(0).round().clip(lower=0).astype(int)
    merged_sub["P10"] = np.minimum(merged_sub["P10"], merged_sub["P50"])
    merged_sub["P90"] = np.maximum(merged_sub["P90"], merged_sub["P50"])
    merged_sub["Projected_Revenue_USD"] = (merged_sub["Units_Sold_Forecast"] * merged_sub["Unit_Price_USD"]).round(2)

    final_sub = merged_sub[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]].copy()
    final_sub.to_csv(output_dir / "submission.csv", index=False)
    final_sub.to_csv(PROJECT_ROOT / "submission.csv", index=False)

    merged_sub.to_parquet(output_dir / "forecast_intervals.parquet", index=False)

    # 3. Compute Portfolio ABC-XYZ Optimizer Output
    compute_abc_xyz_matrix(merged_sub, sku_master, sku_prices, output_dir=output_dir)

    return final_sub, merged_sub, recon_check
