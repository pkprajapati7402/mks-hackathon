"""
Master Feature Pipeline Engine (Workstream 3).
Orchestrates calendar, macro, lifecycle, intermittency, and time-series lag feature generation.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from src.config import FEATURES_DATA_DIR, REPORTS_DIR
from src.features.intermittency import compute_dataset_intermittency
from src.features.calendar_events import add_calendar_and_event_features
from src.features.macro_drivers import add_macro_features, analyze_macro_lead_lags, test_lag_stability
from src.features.ev_transition import add_ev_powertrain_features, analyze_ev_shift

def build_feature_store(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    sku_master: pd.DataFrame,
    macro_clean: pd.DataFrame,
    events_clean: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds continuous panel feature store covering 2019-01 through 2028-05 across all SKUs and Regions.
    Returns:
    - train_features: Features with historical target (Units_Sold_Unconstrained).
    - test_features: Features for forecast window (2026-11 to 2028-05) matching test.csv keys.
    """
    # 1. Aggregate training data to Date-SKU-Region level (authoritative forecasting unit)
    train_agg = train_clean.groupby(["Date", "SKU_ID", "Region"]).agg({
        "Units_Sold": "sum",
        "Units_Sold_Unconstrained": "sum",
        "Revenue_USD": "sum",
        "Unit_Price_USD": "median",
        "Product_Family": "first",
        "Business_Unit": "first",
        "Stockout_Flag": "max",
        "Lost_Units": "sum"
    }).reset_index()

    # 2. Extract unique test keys
    test_keys = test_clean[["Date", "SKU_ID", "Region", "Product_Family", "Business_Unit"]].drop_duplicates().copy()
    test_keys["Units_Sold"] = np.nan
    test_keys["Units_Sold_Unconstrained"] = np.nan
    test_keys["Revenue_USD"] = np.nan
    test_keys["Unit_Price_USD"] = np.nan
    test_keys["Stockout_Flag"] = 0
    test_keys["Lost_Units"] = 0

    # 3. Concatenate into continuous panel (2019-01 to 2028-05)
    panel = pd.concat([train_agg, test_keys], ignore_index=True)
    panel = panel.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)

    # 4. Add Calendar and Event features
    panel = add_calendar_and_event_features(panel, events_clean)

    # 5. Add Macroeconomic features and lags
    panel = add_macro_features(panel, macro_clean)

    # 6. Add EV and Powertrain transition features
    panel = add_ev_powertrain_features(panel, sku_master)

    # 7. Add Lifecycle features
    sku_launch = dict(zip(sku_master["SKU_ID"], sku_master["Launch_Date"]))
    sku_eol = dict(zip(sku_master["SKU_ID"], sku_master["EOL_Date"]))
    sku_replaces = dict(zip(sku_master["SKU_ID"], sku_master["Replaces_SKU"]))

    panel["Launch_Date"] = panel["SKU_ID"].map(sku_launch).fillna("")
    panel["EOL_Date"] = panel["SKU_ID"].map(sku_eol).fillna("")
    panel["Replaces_SKU"] = panel["SKU_ID"].map(sku_replaces).fillna("")
    panel["has_replacement_sku"] = (panel["Replaces_SKU"] != "").astype(int)

    # Date diff calculations
    cur_year = panel["Date"].str[:4].astype(int)
    cur_month = panel["Date"].str[5:7].astype(int)

    # Months since launch
    launch_yr = pd.to_numeric(panel["Launch_Date"].str[:4], errors="coerce").fillna(2019).astype(int)
    launch_mo = pd.to_numeric(panel["Launch_Date"].str[5:7], errors="coerce").fillna(1).astype(int)
    panel["months_since_launch"] = (cur_year - launch_yr) * 12 + (cur_month - launch_mo)
    panel["is_pre_launch"] = (panel["Launch_Date"] != "") & (panel["Date"] < panel["Launch_Date"].str[:7])
    panel["is_pre_launch"] = panel["is_pre_launch"].astype(int)

    # Months until EOL
    eol_yr = pd.to_numeric(panel["EOL_Date"].str[:4], errors="coerce").fillna(2099).astype(int)
    eol_mo = pd.to_numeric(panel["EOL_Date"].str[5:7], errors="coerce").fillna(12).astype(int)
    panel["months_until_eol"] = (eol_yr - cur_year) * 12 + (eol_mo - cur_month)
    panel["is_post_eol"] = (panel["EOL_Date"] != "") & (panel["Date"] > panel["EOL_Date"].str[:7])
    panel["is_post_eol"] = panel["is_post_eol"].astype(int)

    # 8. Compute Intermittency and Baseline rates on training history
    interm_df = compute_dataset_intermittency(train_agg, target_col="Units_Sold_Unconstrained")
    interm_map = interm_df.set_index(["SKU_ID", "Region"]).to_dict(orient="index")

    panel["adi"] = [interm_map.get((s, r), {}).get("ADI", 2.0) for s, r in zip(panel["SKU_ID"], panel["Region"])]
    panel["cv2"] = [interm_map.get((s, r), {}).get("CV2", 0.5) for s, r in zip(panel["SKU_ID"], panel["Region"])]
    panel["intermittency_category"] = [interm_map.get((s, r), {}).get("Category", "Intermittent") for s, r in zip(panel["SKU_ID"], panel["Region"])]
    panel["croston_baseline_rate"] = [interm_map.get((s, r), {}).get("Croston_Rate", 0.0) for s, r in zip(panel["SKU_ID"], panel["Region"])]
    panel["sba_baseline_rate"] = [interm_map.get((s, r), {}).get("SBA_Rate", 0.0) for s, r in zip(panel["SKU_ID"], panel["Region"])]

    # 9. Time Series Lags and Rolling Windows (using training unconstrained demand)
    # Lags: 12, 13, 14, 15, 16, 17, 18, 19
    # (Note: For 19-month multi-step direct forecasting without leakage, lags >= 19 or seasonal lag 12 are safest)
    grouped = panel.groupby(["SKU_ID", "Region"])["Units_Sold_Unconstrained"]

    panel["lag_12"] = grouped.shift(12).fillna(0.0)
    panel["lag_13"] = grouped.shift(13).fillna(0.0)
    panel["lag_14"] = grouped.shift(14).fillna(0.0)
    panel["lag_15"] = grouped.shift(15).fillna(0.0)
    panel["lag_16"] = grouped.shift(16).fillna(0.0)
    panel["lag_17"] = grouped.shift(17).fillna(0.0)
    panel["lag_18"] = grouped.shift(18).fillna(0.0)
    panel["lag_19"] = grouped.shift(19).fillna(0.0)

    # Rolling seasonal aggregations on lag 12
    panel["rolling_mean_12"] = grouped.transform(lambda s: s.shift(12).rolling(12, min_periods=1).mean()).fillna(0.0)
    panel["rolling_std_12"] = grouped.transform(lambda s: s.shift(12).rolling(12, min_periods=1).std()).fillna(0.0)

    # 10. Run Macro Lead/Lag and EV Shift Analyses
    analyze_macro_lead_lags(train_agg, macro_clean)
    test_lag_stability(train_agg, macro_clean)
    analyze_ev_shift(train_agg, sku_master)

    # 11. Split back into train and test feature sets
    train_features = panel[panel["Date"] <= "2026-10"].copy().reset_index(drop=True)
    test_features = panel[panel["Date"] >= "2026-11"].copy().reset_index(drop=True)

    # Save to parquet
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_features.to_parquet(FEATURES_DATA_DIR / "train_features.parquet", index=False)
    test_features.to_parquet(FEATURES_DATA_DIR / "test_features.parquet", index=False)
    panel.to_parquet(FEATURES_DATA_DIR / "full_panel_features.parquet", index=False)

    return train_features, test_features
