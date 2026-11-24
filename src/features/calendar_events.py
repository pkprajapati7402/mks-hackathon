"""
Calendar, Seasonality, Regional Holiday, and Event Engineering Engine (Workstream 3).
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Lunar New Year Months (APAC)
LNY_MONTHS = {
    "2019": 2, "2020": 1, "2021": 2, "2022": 2, "2023": 1,
    "2024": 2, "2025": 1, "2026": 2, "2027": 2, "2028": 1
}

def add_calendar_and_event_features(
    df: pd.DataFrame,
    events_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Enriches dataset with cyclical calendar features, regional automotive holiday schedules,
    and business calendar event indicators.
    """
    out = df.copy()

    # 1. Basic Calendar
    month = out["Date"].str[5:7].astype(int)
    year = out["Date"].str[:4].astype(int)
    quarter = ((month - 1) // 3 + 1).astype(int)

    out["Month"] = month
    out["Year"] = year
    out["Quarter"] = quarter

    # 2. Cyclical Encodings (smooth circular representations)
    out["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0).round(4)
    out["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0).round(4)
    out["quarter_sin"] = np.sin(2.0 * np.pi * (quarter - 1) / 4.0).round(4)
    out["quarter_cos"] = np.cos(2.0 * np.pi * (quarter - 1) / 4.0).round(4)

    # 3. Automotive Industry Schedules
    # Europe August 3-4 week OEM summer plant shutdown
    out["is_europe_summer_shutdown"] = (
        (out["Region"] == "Europe") & (out["Month"] == 8)
    ).astype(int)

    # Lunar New Year shutdown in APAC
    out["is_cny_month"] = [
        int(r == "Asia Pacific" and LNY_MONTHS.get(str(y)) == m)
        for r, y, m in zip(out["Region"], out["Year"], out["Month"])
    ]

    # Golden Week in APAC (October)
    out["is_golden_week_month"] = (
        (out["Region"] == "Asia Pacific") & (out["Month"] == 10)
    ).astype(int)

    # Year-end shutdowns across global plants
    out["is_year_end_holiday"] = (out["Month"] == 12).astype(int)

    # Pre-price hike buying rush (November & December)
    out["is_pre_price_hike_month"] = out["Month"].isin([11, 12]).astype(int)

    # Post-price hike order lull (January)
    out["is_post_price_hike_month"] = (out["Month"] == 1).astype(int)

    # 4. Integrate Events Calendar
    if events_df is not None and not events_df.empty:
        # Check active disruptions and price change events
        price_events = events_df[events_df["Event_Type"] == "Price Change"]
        disruption_events = events_df[events_df["Event_Type"].isin(["Disruption", "Supply Constraint"])]
        
        # Build date sets for global/regional events
        price_dates = set(price_events["Start_Date_Clean"].dropna())
        disruption_dates = set(disruption_events["Start_Date_Clean"].dropna())
        
        out["is_calendar_price_change"] = out["Date"].isin(price_dates).astype(int)
        out["is_calendar_disruption"] = out["Date"].isin(disruption_dates).astype(int)
    else:
        out["is_calendar_price_change"] = 0
        out["is_calendar_disruption"] = 0

    return out
