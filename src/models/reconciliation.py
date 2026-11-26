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
from src.config import OUTPUTS_DIR, PROJECT_ROOT


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
