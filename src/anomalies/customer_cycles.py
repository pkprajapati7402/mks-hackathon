"""
Customer Ordering Cycle & Account Dynamics Engine (Workstream 2).
High-performance vectorized cadence detection, customer switches, and lost accounts.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from src.config import REPORTS_DIR

def analyze_customer_cycles(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes ordering frequency and flags churn/switches.
    Returns:
    - cycle_stats: Per Customer-SKU ordering interval statistics.
    - customer_switches: Instances where a customer ceased one SKU and started another.
    - lost_accounts: Inactive accounts in recent 6 months.
    """
    valid_orders = df[df["Units_Sold"] > 0].copy()
    
    # 1. Vectorized customer cadence statistics
    cust_stats = valid_orders.groupby(["Customer_Name", "SKU_ID", "Region"]).agg(
        Order_Count=("Date", "count"),
        First_Date=("Date", "min"),
        Last_Date=("Date", "max"),
        Total_Units=("Units_Sold", "sum")
    ).reset_index()

    # Calculate span in months
    first_y = cust_stats["First_Date"].str[:4].astype(int)
    first_m = cust_stats["First_Date"].str[5:7].astype(int)
    last_y = cust_stats["Last_Date"].str[:4].astype(int)
    last_m = cust_stats["Last_Date"].str[5:7].astype(int)
    month_span = (last_y - first_y) * 12 + (last_m - first_m) + 1

    cust_stats["Mean_Interval_Months"] = np.where(
        cust_stats["Order_Count"] > 1,
        np.round(month_span / np.maximum(cust_stats["Order_Count"] - 1, 1), 2),
        0.0
    )

    # Classify Cadence
    intervals = cust_stats["Mean_Interval_Months"].values
    counts = cust_stats["Order_Count"].values
    cadence = []
    for cnt, iv in zip(counts, intervals):
        if cnt <= 1:
            cadence.append("Ad-hoc / Single Order")
        elif iv <= 1.4:
            cadence.append("Monthly")
        elif 1.7 <= iv <= 2.3:
            cadence.append("Bi-Monthly")
        elif 2.7 <= iv <= 3.4:
            cadence.append("Quarterly")
        elif 5.5 <= iv <= 6.5:
            cadence.append("Semi-Annual")
        else:
            cadence.append("Irregular")

    cust_stats["Cadence"] = cadence
    cycle_df = cust_stats[["Customer_Name", "SKU_ID", "Region", "Order_Count", "Mean_Interval_Months", "Cadence"]].copy()

    # 2. Lost Accounts (Active prior to 2026-05, but zero sales between 2026-05 and 2026-10)
    recent_active = df[df["Date"] >= "2026-05"].groupby(["Customer_Name", "SKU_ID", "Region"])["Units_Sold"].sum()
    prior_active = df[df["Date"] < "2026-05"].groupby(["Customer_Name", "SKU_ID", "Region"])["Units_Sold"].sum()
    
    lost_keys = set(prior_active[prior_active > 0].index) - set(recent_active[recent_active > 0].index)
    lost_records = []
    for (cust, sku, reg) in list(lost_keys)[:1000]:
        tot_prior = prior_active.get((cust, sku, reg), 0)
        lost_records.append({
            "Customer_Name": cust,
            "SKU_ID": sku,
            "Region": reg,
            "Historical_Units": int(tot_prior),
            "Status": "Dormant / Lost Account (> 6 months inactive)"
        })
    lost_df = pd.DataFrame(lost_records)

    # 3. Customer Switches
    family_orders = valid_orders.groupby(["Customer_Name", "Product_Family", "SKU_ID"]).agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        total_units=("Units_Sold", "sum")
    ).reset_index()

    switch_records = []
    for (cust, fam), grp in family_orders.groupby(["Customer_Name", "Product_Family"]):
        if len(grp) >= 2:
            grp_sorted = grp.sort_values("last_date")
            for i in range(len(grp_sorted) - 1):
                curr = grp_sorted.iloc[i]
                nxt = grp_sorted.iloc[i+1]
                if curr["last_date"] <= nxt["first_date"] and curr["last_date"] < "2026-06":
                    switch_records.append({
                        "Customer_Name": cust,
                        "Product_Family": fam,
                        "Replaced_SKU": curr["SKU_ID"],
                        "Successor_SKU": nxt["SKU_ID"],
                        "Transition_Date": curr["last_date"],
                        "Prior_SKU_Volume": int(curr["total_units"]),
                        "New_SKU_Volume": int(nxt["total_units"])
                    })
    switch_df = pd.DataFrame(switch_records)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cycle_df.to_csv(REPORTS_DIR / "customer_ordering_cycles.csv", index=False)
    lost_df.to_csv(REPORTS_DIR / "lost_accounts.csv", index=False)
    switch_df.to_csv(REPORTS_DIR / "customer_switches.csv", index=False)

    return cycle_df, switch_df, lost_df
