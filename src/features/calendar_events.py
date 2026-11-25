"""
Calendar, Seasonality, Regional Holiday, and Scoped Events Engineering Engine (Workstream 3 & 4 Expert Tier).
Properly resolves Scope_Type (Global, Region, Product_Family, Channel, Customer|Region),
Start_Date..End_Date spans, open-ended expansions (EV-052), and forecast-window events (EV-059..EV-068).
"""

import re
from typing import Dict
import numpy as np
import pandas as pd

# Fallback LNY / Ramadan month maps (2019-2028) when events_df is omitted in unit tests
LNY_MONTHS: Dict[str, int] = {
    "2019": 2, "2020": 1, "2021": 2, "2022": 2, "2023": 1,
    "2024": 2, "2025": 1, "2026": 2, "2027": 2, "2028": 1,
}

RAMADAN_MONTHS: Dict[str, tuple] = {
    "2019": (5, 6), "2020": (4, 5), "2021": (4, 5), "2022": (4, 5), "2023": (3, 4),
    "2024": (3, 4), "2025": (3,), "2026": (2, 3), "2027": (2, 3), "2028": (1, 2),
}

PRICE_HIKE_PCT_BY_JAN_YEAR: Dict[int, float] = {
    2019: 2.0, 2020: 2.0, 2021: 2.5, 2022: 5.0, 2023: 9.0,
    2024: 4.5, 2025: 3.7, 2026: 3.7, 2027: 4.5, 2028: 4.0,
}


def add_calendar_and_event_features(
    df: pd.DataFrame,
    events_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Enriches dataset with cyclical calendar features, scoped regional holidays (EU Summer, NA Retool,
    APAC LNY/Golden Week, MEA Ramadan), price hike anticipation, and exact scoped calendar events
    across both history (2019-2026) and forecast window (2026-11..2028-05).
    """
    out = df.copy()

    # 1. Basic Calendar
    month = out["Date"].str[5:7].astype(int)
    year = out["Date"].str[:4].astype(int)
    quarter = ((month - 1) // 3 + 1).astype(int)

    out["Month"] = month
    out["Year"] = year
    out["Quarter"] = quarter

    # 2. Cyclical Encodings (1st, 2nd & 3rd Fourier seasonal harmonics)
    out["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0).round(4)
    out["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0).round(4)
    out["month_sin2"] = np.sin(4.0 * np.pi * (month - 1) / 12.0).round(4)
    out["month_cos2"] = np.cos(4.0 * np.pi * (month - 1) / 12.0).round(4)
    out["month_sin3"] = np.sin(6.0 * np.pi * (month - 1) / 12.0).round(4)
    out["month_cos3"] = np.cos(6.0 * np.pi * (month - 1) / 12.0).round(4)
    out["quarter_sin"] = np.sin(2.0 * np.pi * (quarter - 1) / 4.0).round(4)
    out["quarter_cos"] = np.cos(2.0 * np.pi * (quarter - 1) / 4.0).round(4)

    # 3. Scoped Industry Holidays & Plant Schedules
    out["is_europe_summer_shutdown"] = ((out["Region"] == "Europe") & (out["Month"] == 8)).astype(int)
    out["is_na_retool_month"] = ((out["Region"] == "North America") & (out["Month"] == 7)).astype(int)
    out["is_golden_week_month"] = ((out["Region"] == "Asia Pacific") & (out["Month"] == 10)).astype(int)
    out["is_year_end_holiday"] = (out["Month"] == 12).astype(int)

    out["is_cny_month"] = [
        int(r == "Asia Pacific" and LNY_MONTHS.get(str(y)) == m)
        for r, y, m in zip(out["Region"], out["Year"], out["Month"])
    ]
    out["is_ramadan_month"] = [
        int(r in ("Middle East & Africa", "India") and m in RAMADAN_MONTHS.get(str(y), ()))
        for r, y, m in zip(out["Region"], out["Year"], out["Month"])
    ]

    # 4. Price-Hike Anticipation (Nov/Dec pull-forward + Jan post-hike dip)
    out["is_pre_price_hike_month"] = out["Month"].isin([11, 12]).astype(int)
    out["is_post_price_hike_month"] = (out["Month"] == 1).astype(int)
    out["announced_price_hike_pct"] = [
        PRICE_HIKE_PCT_BY_JAN_YEAR.get(y + 1 if m in (11, 12) else y, 3.5)
        if m in (11, 12, 1) else 0.0
        for y, m in zip(out["Year"], out["Month"])
    ]

    # 5. Scoped Events Integration from events_calendar.csv
    out["is_calendar_price_change"] = 0
    out["is_calendar_disruption"] = 0
    out["is_supply_constraint_event"] = 0
    out["is_orion_retool_event"] = 0
    out["is_helios_expansion"] = 0
    out["is_erp_migration_gap"] = 0
    out["is_marketing_event"] = 0
    out["is_mea_pre_entry"] = ((out["Region"] == "Middle East & Africa") & (out["Date"] < "2022-01")).astype(int)

    if events_df is not None and not events_df.empty:
        dates_s = out["Date"].astype(str)
        has_fam = "Product_Family" in out.columns
        has_chan = "Sales_Channel" in out.columns
        has_cust = "Customer_Name" in out.columns

        for _, ev in events_df.iterrows():
            ev_id = str(ev.get("Event_ID", ""))
            start_d = str(ev.get("Start_Date_Clean") or ev.get("Start_Date") or "")[:7]
            if not re.match(r"^\d{4}-\d{2}$", start_d):
                continue
            raw_end = ev.get("End_Date_Clean") if pd.notna(ev.get("End_Date_Clean")) else ev.get("End_Date")
            if pd.notna(raw_end) and re.match(r"^\d{4}-\d{2}", str(raw_end).strip()):
                end_d = str(raw_end).strip()[:7]
            elif ev_id == "EV-052" or "permanent" in str(ev.get("Description", "")).lower():
                end_d = "2028-12"
            else:
                end_d = start_d

            date_mask = (dates_s >= start_d) & (dates_s <= end_d)
            if not date_mask.any():
                continue

            scope_type = str(ev.get("Scope_Type", "Global")).strip()
            scope_val = str(ev.get("Scope_Value", "All")).strip()
            ev_type = str(ev.get("Event_Type", "")).strip()

            scope_mask = pd.Series(True, index=out.index)
            if scope_type == "Region":
                scope_mask = out["Region"] == scope_val
            elif scope_type == "Product_Family" and has_fam:
                scope_mask = out["Product_Family"] == scope_val
            elif scope_type == "Channel" and has_chan:
                scope_mask = out["Sales_Channel"] == scope_val
            elif scope_type == "Customer" and has_cust:
                cust_part = str(ev.get("Scope_Customer") or scope_val.split("|")[0]).strip()
                reg_part = ev.get("Scope_Region")
                if pd.isna(reg_part) and "|" in scope_val:
                    reg_part = scope_val.split("|")[1].strip()
                scope_mask = out["Customer_Name"].astype(str) == cust_part
                if pd.notna(reg_part) and str(reg_part).strip():
                    scope_mask = scope_mask & (out["Region"] == str(reg_part).strip())

            hit = date_mask & scope_mask
            if not hit.any():
                continue

            if ev_type == "Price Change":
                out.loc[hit, "is_calendar_price_change"] = 1
            elif ev_type == "Supply Constraint":
                out.loc[hit, "is_supply_constraint_event"] = 1
                out.loc[hit, "is_calendar_disruption"] = 1
            elif ev_type == "Disruption":
                out.loc[hit, "is_calendar_disruption"] = 1
            elif ev_type == "System" and "ERP migration" in str(ev.get("Description", "")):
                out.loc[hit, "is_erp_migration_gap"] = 1
            elif ev_type == "Marketing":
                out.loc[hit, "is_marketing_event"] = 1

            # Explicit high-impact customer events in history & forecast window
            if ev_id == "EV-062":
                out.loc[hit, "is_orion_retool_event"] = 1
                out.loc[hit, "is_calendar_disruption"] = 1
            elif ev_id == "EV-052":
                out.loc[hit, "is_helios_expansion"] = 1
            elif ev_id == "EV-038":
                out.loc[hit, "is_calendar_disruption"] = 1

    return out
