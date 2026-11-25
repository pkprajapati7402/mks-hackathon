"""
Customer Ordering Cycle & Account Dynamics Engine (Workstream 2 Expert Tier).
Detects per Customer-SKU ordering cadences, temporal ordering-cycle shifts (Rule 6),
customer-SKU handoff switches, and lost/dormant accounts.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from src.config import REPORTS_DIR


def _classify_cadence(count: int, interval: float) -> str:
    if count <= 1:
        return "Ad-hoc / Single Order"
    if interval <= 1.4:
        return "Monthly"
    if 1.4 < interval <= 2.4:
        return "Bi-Monthly"
    if 2.4 < interval <= 3.6:
        return "Quarterly"
    if 5.2 <= interval <= 6.8:
        return "Semi-Annual"
    return "Irregular"


def analyze_customer_cycles(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes ordering frequency, temporal cadence changes (Rule 6), switches, and lost accounts.
    Saves deterministic CSV reports to reports/.
    """
    valid_orders = df[df["Units_Sold"] > 0].copy()

    # 1. Overall Customer-SKU-Region cadence statistics
    cust_stats = valid_orders.groupby(["Customer_Name", "SKU_ID", "Region"]).agg(
        Order_Count=("Date", "count"),
        First_Date=("Date", "min"),
        Last_Date=("Date", "max"),
        Total_Units=("Units_Sold", "sum"),
    ).reset_index()

    first_y = cust_stats["First_Date"].str[:4].astype(int)
    first_m = cust_stats["First_Date"].str[5:7].astype(int)
    last_y = cust_stats["Last_Date"].str[:4].astype(int)
    last_m = cust_stats["Last_Date"].str[5:7].astype(int)
    month_span = (last_y - first_y) * 12 + (last_m - first_m) + 1

    cust_stats["Mean_Interval_Months"] = np.where(
        cust_stats["Order_Count"] > 1,
        np.round(month_span / np.maximum(cust_stats["Order_Count"] - 1, 1), 2),
        0.0,
    )
    cust_stats["Cadence"] = [
        _classify_cadence(cnt, iv)
        for cnt, iv in zip(cust_stats["Order_Count"].values, cust_stats["Mean_Interval_Months"].values)
    ]

    cycle_df = cust_stats[
        ["Customer_Name", "SKU_ID", "Region", "Order_Count", "First_Date", "Last_Date", "Mean_Interval_Months", "Cadence"]
    ].sort_values(["Customer_Name", "SKU_ID", "Region"]).reset_index(drop=True)

    # 2. Temporal Ordering-Cycle Changes (Early Period <= 2022-12 vs Recent Period >= 2023-01)
    def _period_cadence(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        g = sub.groupby(["Customer_Name", "SKU_ID", "Region"]).agg(
            cnt=("Date", "count"),
            dmin=("Date", "min"),
            dmax=("Date", "max"),
        ).reset_index()
        span = (
            (g["dmax"].str[:4].astype(int) - g["dmin"].str[:4].astype(int)) * 12
            + (g["dmax"].str[5:7].astype(int) - g["dmin"].str[5:7].astype(int))
            + 1
        )
        iv = np.where(g["cnt"] > 1, span / np.maximum(g["cnt"] - 1, 1), 0.0)
        g[f"{label}_Interval"] = np.round(iv, 2)
        g[f"{label}_Cadence"] = [_classify_cadence(c, x) for c, x in zip(g["cnt"].values, iv)]
        return g[["Customer_Name", "SKU_ID", "Region", f"{label}_Cadence", f"{label}_Interval"]]

    early_cad = _period_cadence(valid_orders[valid_orders["Date"] <= "2022-12"], "Early")
    recent_cad = _period_cadence(valid_orders[valid_orders["Date"] >= "2023-01"], "Recent")
    cad_compare = early_cad.merge(recent_cad, on=["Customer_Name", "SKU_ID", "Region"], how="inner")
    cad_changes = cad_compare[cad_compare["Early_Cadence"] != cad_compare["Recent_Cadence"]].copy()
    cad_changes["Shift_Pattern"] = cad_changes["Early_Cadence"] + " -> " + cad_changes["Recent_Cadence"]
    cad_changes = cad_changes.sort_values(["Customer_Name", "SKU_ID", "Region"]).reset_index(drop=True)

    # 3. Lost Accounts (Active prior to 2026-05, zero sales between 2026-05 and 2026-10)
    recent_active = df[df["Date"] >= "2026-05"].groupby(["Customer_Name", "SKU_ID", "Region"])["Units_Sold"].sum()
    prior_active = df[df["Date"] < "2026-05"].groupby(["Customer_Name", "SKU_ID", "Region"])["Units_Sold"].sum()

    lost_keys = sorted(list(set(prior_active[prior_active > 0].index) - set(recent_active[recent_active > 0].index)))
    lost_records = []
    for cust, sku, reg in lost_keys[:1000]:
        tot_prior = prior_active.get((cust, sku, reg), 0)
        lost_records.append({
            "Customer_Name": cust,
            "SKU_ID": sku,
            "Region": reg,
            "Historical_Units": int(tot_prior),
            "Status": "Dormant / Lost Account (> 6 months inactive)",
        })
    lost_df = pd.DataFrame(lost_records, columns=["Customer_Name", "SKU_ID", "Region", "Historical_Units", "Status"])

    # 4. Customer Switches
    family_orders = valid_orders.groupby(["Customer_Name", "Product_Family", "SKU_ID"]).agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        total_units=("Units_Sold", "sum"),
    ).reset_index()

    switch_records = []
    for (cust, fam), grp in family_orders.groupby(["Customer_Name", "Product_Family"]):
        if len(grp) >= 2:
            grp_sorted = grp.sort_values(["last_date", "SKU_ID"])
            for i in range(len(grp_sorted) - 1):
                curr = grp_sorted.iloc[i]
                nxt = grp_sorted.iloc[i + 1]
                if curr["last_date"] <= nxt["first_date"] and curr["last_date"] < "2026-06":
                    switch_records.append({
                        "Customer_Name": cust,
                        "Product_Family": fam,
                        "Replaced_SKU": curr["SKU_ID"],
                        "Successor_SKU": nxt["SKU_ID"],
                        "Transition_Date": curr["last_date"],
                        "Prior_SKU_Volume": int(curr["total_units"]),
                        "New_SKU_Volume": int(nxt["total_units"]),
                    })
    switch_df = pd.DataFrame(switch_records)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cycle_df.to_csv(REPORTS_DIR / "customer_ordering_cycles.csv", index=False)
    cad_changes.to_csv(REPORTS_DIR / "cadence_changes.csv", index=False)
    lost_df.to_csv(REPORTS_DIR / "lost_accounts.csv", index=False)
    switch_df.to_csv(REPORTS_DIR / "customer_switches.csv", index=False)

    return cycle_df, switch_df, lost_df
