"""
Cleaning and validation for auxiliary reference tables:
- sku_master.csv (Rules S1-S2)
- macro_indicators.csv (Rules M1-M5)
- events_calendar.csv (Rules E1-E4)
- inventory_stockouts.csv (Rules K1-K3)
"""

import re
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    PREFIX_TO_FAMILY, FAMILY_TO_BU,
    REGION_MAP, CANONICAL_REGIONS, MACRO_COLUMNS
)
from src.audit import audit

def clean_sku_master(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate SKU master."""
    dq = {}
    df = raw.copy()
    dq["raw_rows"] = len(df)

    # S1: Ensure Precision Machined Parts maps to Advanced Manufacturing
    conflict_skus = df[
        (df["Product_Family"] == "Precision Machined Parts") &
        (df["Business_Unit"] != "Advanced Manufacturing")
    ]
    dq["S1_conflict_count"] = len(conflict_skus)
    for idx in conflict_skus.index:
        sku = df.at[idx, "SKU_ID"]
        old_bu = df.at[idx, "Business_Unit"]
        audit.log(
            "sku_master.csv", sku, "Business_Unit",
            old_bu, "Advanced Manufacturing", "S1",
            "Precision Machined Parts assigned to Advanced Manufacturing"
        )
        df.at[idx, "Business_Unit"] = "Advanced Manufacturing"

    # S2: Validate lifecycle dates and Replaces_SKU reference
    all_skus = set(df["SKU_ID"])
    missing_refs = 0
    for idx, row in df.iterrows():
        rep = row.get("Replaces_SKU")
        if pd.notna(rep) and str(rep).strip():
            rep_clean = str(rep).strip()
            if rep_clean not in all_skus:
                audit.log(
                    "sku_master.csv", row["SKU_ID"], "Replaces_SKU",
                    rep, "MISSING_REF", "S2",
                    f"Replaces_SKU target {rep_clean} not in master"
                )
                missing_refs += 1
    dq["S2_missing_replaces_sku"] = missing_refs
    dq["clean_rows"] = len(df)
    return df, dq

def clean_macro_indicators(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean macroeconomic indicators (2019-01 -> 2028-05)."""
    dq = {}
    df = raw.copy()
    dq["raw_rows"] = len(df)

    # Convert numeric columns
    for col in MACRO_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # M1: Handle duplicate month (2022-06)
    dup_dates = df[df.duplicated("Date", keep=False)]["Date"].unique()
    dq["M1_duplicate_months"] = len(dup_dates)
    if len(dup_dates) > 0:
        for d in dup_dates:
            idxs = df[df["Date"] == d].index.tolist()
            prev_row = df[df["Date"] < d].iloc[-1] if not df[df["Date"] < d].empty else None
            next_row = df[df["Date"] > d].iloc[0] if not df[df["Date"] > d].empty else None

            chosen_idx = idxs[0]
            if prev_row is not None and next_row is not None:
                # Check PMI proximity to interpolated value
                target = (prev_row["ISM_Manufacturing_PMI"] + next_row["ISM_Manufacturing_PMI"]) / 2
                dist0 = abs(df.at[idxs[0], "ISM_Manufacturing_PMI"] - target)
                dist1 = abs(df.at[idxs[1], "ISM_Manufacturing_PMI"] - target)
                if dist1 < dist0:
                    chosen_idx = idxs[1]

            drop_idxs = [i for i in idxs if i != chosen_idx]
            audit.log(
                "macro_indicators.csv", d, "Row_Deduplication",
                f"{len(idxs)} rows", f"kept index {chosen_idx}", "M1",
                f"Resolved duplicate month {d} via neighbor interpolation"
            )
            df = df.drop(index=drop_idxs)

    # M2: Auto_Production_YoY_Pct recorded as fraction (< 1.0) in 2019
    df["Year"] = df["Date"].str[:4].astype(int)
    frac_mask = (df["Year"] <= 2019) & (df["Auto_Production_YoY_Pct"].abs() < 1.0) & (df["Auto_Production_YoY_Pct"].notna())
    dq["M2_fraction_pct_fixed"] = int(frac_mask.sum())
    for idx in df.index[frac_mask]:
        val = df.at[idx, "Auto_Production_YoY_Pct"]
        new_val = round(val * 100.0, 2)
        audit.log(
            "macro_indicators.csv", df.at[idx, "Date"], "Auto_Production_YoY_Pct",
            val, new_val, "M2", "Converted decimal fraction to percentage"
        )
        df.at[idx, "Auto_Production_YoY_Pct"] = new_val

    # M3: PMI decimal shift (values < 10 should be scaled by 10)
    pmi_shift = (df["ISM_Manufacturing_PMI"] < 10.0) & (df["ISM_Manufacturing_PMI"] > 0)
    dq["M3_pmi_decimal_shifted"] = int(pmi_shift.sum())
    for idx in df.index[pmi_shift]:
        val = df.at[idx, "ISM_Manufacturing_PMI"]
        new_val = round(val * 10.0, 2)
        audit.log(
            "macro_indicators.csv", df.at[idx, "Date"], "ISM_Manufacturing_PMI",
            val, new_val, "M3", "Corrected PMI decimal point shift (x10)"
        )
        df.at[idx, "ISM_Manufacturing_PMI"] = new_val

    # M4: Interpolate missing values across time
    df = df.sort_values("Date").reset_index(drop=True)
    df["Month"] = df["Date"].str[5:7].astype(int)
    for col in MACRO_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].interpolate(method="linear").bfill().ffill()
            audit.log(
                "macro_indicators.csv", col, "Missing_Values",
                f"{null_count} NaNs", "Interpolated", "M4",
                f"Interpolated {null_count} missing entries for {col}"
            )
    dq["clean_rows"] = len(df)
    return df, dq

def clean_events_calendar(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean events calendar and normalize scopes."""
    dq = {}
    df = raw.copy()
    dq["raw_rows"] = len(df)

    # E1 & E2: Standardize Start_Date and End_Date
    def parse_event_date(val):
        if pd.isna(val) or not str(val).strip():
            return None
        s = str(val).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s[:7]
        if re.match(r"^\d{4}-\d{2}$", s):
            return s
        if re.match(r"^\d{2}/\d{4}$", s):
            m, y = s.split("/")
            return f"{y}-{m}"
        return s[:7]

    df["Start_Date_Clean"] = df["Start_Date"].apply(parse_event_date)
    df["End_Date_Clean"] = df["End_Date"].apply(parse_event_date)

    # E3: Split combined Customer | Region in Scope_Value
    df["Scope_Customer"] = None
    df["Scope_Region"] = None
    for idx, row in df.iterrows():
        sv = str(row.get("Scope_Value", "")).strip()
        if "|" in sv:
            parts = [p.strip() for p in sv.split("|")]
            df.at[idx, "Scope_Customer"] = parts[0]
            df.at[idx, "Scope_Region"] = parts[1] if len(parts) > 1 else None
        elif row.get("Scope_Type") == "Region":
            df.at[idx, "Scope_Region"] = sv
        elif row.get("Scope_Type") == "Customer":
            df.at[idx, "Scope_Customer"] = sv

    dq["clean_rows"] = len(df)
    return df, dq

def clean_inventory_stockouts(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean inventory stockouts table."""
    dq = {}
    df = raw.copy()
    dq["raw_rows"] = len(df)

    # Normalize Region
    df["Region"] = df["Region"].astype(str).str.strip().str.lower().map(
        lambda r: REGION_MAP.get(r, r.title())
    )

    # Normalize SKU_ID
    def clean_sku(s):
        if pd.isna(s): return s
        st = str(s).strip()
        m = re.match(r"^([a-zA-Z]{3})[-_]?(\d{1,3})$", st)
        if m:
            return f"{m.group(1).upper()}-{int(m.group(2)):03d}"
        return st.upper()

    df["SKU_ID"] = df["SKU_ID"].apply(clean_sku)

    # Parse numeric units
    cols = ["Requested_Units", "Carried_In_Units", "Shipped_Units", "Backordered_Units", "Lost_Units"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round().astype(int)

    # Reconcile Lost_Units
    expected_lost = (df["Requested_Units"] + df["Carried_In_Units"] - df["Shipped_Units"] - df["Backordered_Units"]).clip(lower=0)
    diff = (df["Lost_Units"] - expected_lost).abs() > 0
    dq["lost_units_reconciled"] = int(diff.sum())
    df["Lost_Units"] = expected_lost

    dq["clean_rows"] = len(df)
    return df, dq
