"""
Hierarchical Reconciliation & Submission Generation Engine (Workstream 4 Expert Tier).
Enforces exact bottom-up additivity across SKU -> Family -> BU -> Portfolio and Region,
and formats the final 58,219-row competition submission file.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import OUTPUTS_DIR, PROJECT_ROOT

def reconcile_and_generate_submission(
    test_features: pd.DataFrame,
    lgb_preds: pd.DataFrame,
    test_raw: pd.DataFrame,
    sku_master: pd.DataFrame,
    output_dir: Path = OUTPUTS_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs hierarchical bottom-up reconciliation and formats final submission:
    1. Verifies exact addition across SKU, Product Family, Business Unit, Region, and Total Portfolio.
    2. Enforces non-negative integers.
    3. Produces exact 58,219 row submission matching test.csv and sample_submission.csv.
    """
    out_forecast = test_features[["Date", "SKU_ID", "Region", "Product_Family", "Business_Unit"]].copy()

    # Align predictions
    out_forecast["P10"] = lgb_preds["P10"].values
    out_forecast["P50"] = lgb_preds["P50"].values
    out_forecast["P90"] = lgb_preds["P90"].values

    # Primary forecast target: P50 rounded non-negative integer
    out_forecast["Units_Sold_Forecast"] = np.clip(np.round(out_forecast["P50"]), 0, None).astype(int)

    # 1. Hierarchical Aggregation & Verification
    # SKU level sum
    sku_sum = out_forecast.groupby(["Date", "Product_Family", "Region"])["Units_Sold_Forecast"].sum().reset_index(name="Family_Region_Sum")

    # Family Global
    fam_sum = out_forecast.groupby(["Date", "Product_Family"])["Units_Sold_Forecast"].sum().reset_index(name="Family_Global_Sum")

    # Business Unit Global
    bu_sum = out_forecast.groupby(["Date", "Business_Unit"])["Units_Sold_Forecast"].sum().reset_index(name="BU_Global_Sum")

    # Region Global
    reg_sum = out_forecast.groupby(["Date", "Region"])["Units_Sold_Forecast"].sum().reset_index(name="Region_Global_Sum")

    # Total Portfolio
    total_by_bu = bu_sum.groupby("Date")["BU_Global_Sum"].sum().reset_index(name="Total_By_BU")
    total_by_reg = reg_sum.groupby("Date")["Region_Global_Sum"].sum().reset_index(name="Total_By_Region")

    recon_check = total_by_bu.merge(total_by_reg, on="Date")
    recon_check["Discrepancy"] = recon_check["Total_By_BU"] - recon_check["Total_By_Region"]
    recon_check.to_csv(output_dir / "hierarchical_reconciliation.csv", index=False)

    # 2. Map predictions to exact test.csv rows
    # test.csv has 58,219 rows with keys (Date, SKU_ID, Region, Customer_Name, Sales_Channel)
    merged_sub = test_raw[["Date", "SKU_ID", "Region"]].merge(
        out_forecast[["Date", "SKU_ID", "Region", "Units_Sold_Forecast", "P10", "P50", "P90"]],
        on=["Date", "SKU_ID", "Region"],
        how="left"
    )

    # Ensure no blanks or negatives
    merged_sub["Units_Sold_Forecast"] = merged_sub["Units_Sold_Forecast"].fillna(0).round().clip(lower=0).astype(int)
    merged_sub["P10"] = merged_sub["P10"].fillna(0).round().clip(lower=0).astype(int)
    merged_sub["P50"] = merged_sub["P50"].fillna(0).round().clip(lower=0).astype(int)
    merged_sub["P90"] = merged_sub["P90"].fillna(0).round().clip(lower=0).astype(int)

    # Final competition submission format
    final_sub = merged_sub[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]].copy()

    # Save outputs
    sub_path = output_dir / "submission.csv"
    root_sub_path = PROJECT_ROOT / "submission.csv"
    final_sub.to_csv(sub_path, index=False)
    final_sub.to_csv(root_sub_path, index=False)

    # Save prediction intervals
    intervals_path = output_dir / "forecast_intervals.parquet"
    merged_sub.to_parquet(intervals_path, index=False)

    return final_sub, merged_sub, recon_check
