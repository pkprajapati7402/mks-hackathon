"""
EV Transition Dynamics & Powertrain Application Analysis (Workstream 3 Expert Tier).
Analyzes the structural industry shift from ICE to EV and creates powertrain interaction features.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.config import REPORTS_DIR

def add_ev_powertrain_features(
    df: pd.DataFrame,
    sku_master: pd.DataFrame
) -> pd.DataFrame:
    """
    Enriches dataset with Powertrain Application classifications (ICE, EV, Universal)
    and interacts them with EV_Adoption_Index to model the technology transition.
    """
    out = df.copy()

    # Map Powertrain_Application from sku_master
    sku_to_pt = dict(zip(sku_master["SKU_ID"], sku_master["Powertrain_Application"]))
    out["Powertrain_Application"] = out["SKU_ID"].map(sku_to_pt).fillna("Universal")

    # One-hot flags
    out["is_powertrain_ev"] = (out["Powertrain_Application"] == "EV").astype(int)
    out["is_powertrain_ice"] = (out["Powertrain_Application"] == "ICE").astype(int)
    out["is_powertrain_universal"] = (out["Powertrain_Application"] == "Universal").astype(int)

    # Interaction terms with EV_Adoption_Index
    if "EV_Adoption_Index" in out.columns:
        out["ev_index_x_is_ev"] = out["EV_Adoption_Index"] * out["is_powertrain_ev"]
        out["ev_index_x_is_ice"] = out["EV_Adoption_Index"] * out["is_powertrain_ice"]
    else:
        out["ev_index_x_is_ev"] = 0.0
        out["ev_index_x_is_ice"] = 0.0

    return out

def analyze_ev_shift(
    train_df: pd.DataFrame,
    sku_master: pd.DataFrame
) -> pd.DataFrame:
    """
    Quantifies the structural shift across ICE, EV, and Universal component lines over 2019-2026.
    Computes annual volume, growth rates, and portfolio share.
    """
    df = train_df.copy()
    if "Year" not in df.columns:
        df["Year"] = df["Date"].str[:4].astype(int)

    sku_to_pt = dict(zip(sku_master["SKU_ID"], sku_master["Powertrain_Application"]))
    df["Powertrain_Application"] = df["SKU_ID"].map(sku_to_pt).fillna("Universal")

    # Group by Year and Powertrain_Application
    annual_pt = df.groupby(["Year", "Powertrain_Application"])["Units_Sold_Unconstrained"].sum().unstack(fill_value=0)
    
    records = []
    for yr in annual_pt.index:
        tot = annual_pt.loc[yr].sum()
        ice_vol = annual_pt.loc[yr].get("ICE", 0)
        ev_vol = annual_pt.loc[yr].get("EV", 0)
        uni_vol = annual_pt.loc[yr].get("Universal", 0)

        records.append({
            "Year": yr,
            "Total_Volume": tot,
            "ICE_Volume": ice_vol,
            "EV_Volume": ev_vol,
            "Universal_Volume": uni_vol,
            "ICE_Share_Pct": round(ice_vol / tot * 100.0, 2) if tot > 0 else 0.0,
            "EV_Share_Pct": round(ev_vol / tot * 100.0, 2) if tot > 0 else 0.0,
            "Universal_Share_Pct": round(uni_vol / tot * 100.0, 2) if tot > 0 else 0.0
        })

    shift_df = pd.DataFrame(records)
    shift_df.to_csv(REPORTS_DIR / "ev_shift_analysis.csv", index=False)
    return shift_df
