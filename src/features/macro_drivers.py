"""
Macroeconomic Drivers, Cross-Correlation, and Lag Stability Engine (Workstream 3).
Analyzes which macro indicators lead which product families and tests temporal lag stability.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from src.config import MACRO_COLUMNS, CANONICAL_PRODUCT_FAMILIES, REPORTS_DIR

def add_macro_features(
    df: pd.DataFrame,
    macro_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges cleaned macroeconomic indicators and creates strategic lag features (lag 1, 2, 3, 6).
    """
    out = df.copy()
    m_df = macro_clean.sort_values("Date").copy()

    # Create lag features in macro table first
    for col in MACRO_COLUMNS:
        if col in m_df.columns:
            m_df[f"{col}_lag1"] = m_df[col].shift(1)
            m_df[f"{col}_lag2"] = m_df[col].shift(2)
            m_df[f"{col}_lag3"] = m_df[col].shift(3)
            m_df[f"{col}_lag6"] = m_df[col].shift(6)

    # Forward fill early lags
    m_df = m_df.bfill()

    # Merge on Date
    macro_cols_to_merge = ["Date"] + [c for c in m_df.columns if c in MACRO_COLUMNS or "_lag" in c]
    out = out.merge(m_df[macro_cols_to_merge], on="Date", how="left")
    return out

def analyze_macro_lead_lags(
    train_df: pd.DataFrame,
    macro_clean: pd.DataFrame,
    max_lag: int = 12
) -> pd.DataFrame:
    """
    Performs cross-correlation analysis between all 8 macro indicators and 7 Product Families
    across lags 0 to 12. Identifies the leading indicator and optimal lead time per family.
    """
    # 1. Aggregate monthly demand per product family
    monthly_family = train_df.groupby(["Date", "Product_Family"])["Units_Sold_Unconstrained"].sum().unstack(fill_value=0)
    
    # 2. Merge with macro indicators
    m_indexed = macro_clean.set_index("Date").sort_index()
    common_dates = sorted(list(set(monthly_family.index) & set(m_indexed.index)))
    
    results = []
    for fam in CANONICAL_PRODUCT_FAMILIES:
        if fam not in monthly_family.columns:
            continue
        y = monthly_family.loc[common_dates, fam].values
        
        for col in MACRO_COLUMNS:
            if col not in m_indexed.columns:
                continue
            x = m_indexed.loc[common_dates, col].values
            
            best_lag = 0
            best_corr = 0.0
            lag_corrs = {}
            
            for lag in range(0, max_lag + 1):
                if lag == 0:
                    corr = np.corrcoef(x, y)[0, 1]
                else:
                    corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                
                lag_corrs[f"lag_{lag}"] = round(corr, 3)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
                    
            results.append({
                "Product_Family": fam,
                "Macro_Indicator": col,
                "Optimal_Lag_Months": best_lag,
                "Max_Correlation": round(best_corr, 3),
                "Relationship": "Leading Indicator" if best_lag > 0 else "Coincident",
                **lag_corrs
            })

    lead_lag_df = pd.DataFrame(results)
    lead_lag_df.to_csv(REPORTS_DIR / "macro_lead_lag_analysis.csv", index=False)
    return lead_lag_df

def test_lag_stability(
    train_df: pd.DataFrame,
    macro_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluates whether macroeconomic lead/lag relationships remained stable over time
    by comparing Pre-2022 (2019-2021) vs Post-2022 (2022-2026).
    """
    monthly_family = train_df.groupby(["Date", "Product_Family"])["Units_Sold_Unconstrained"].sum().unstack(fill_value=0)
    m_indexed = macro_clean.set_index("Date").sort_index()

    dates_pre = sorted([d for d in monthly_family.index if d < "2022-01"])
    dates_post = sorted([d for d in monthly_family.index if d >= "2022-01"])

    stability_records = []
    key_pairs = [
        ("Powertrain Components", "New_Vehicle_Orders_Index", 3),
        ("Hydraulic Systems", "Industrial_Util_Rate_Pct", 1),
        ("Electrical Assemblies", "EV_Adoption_Index", 0),
        ("Brake & Safety Systems", "Auto_Production_YoY_Pct", 1),
        ("Chassis & Structural", "ISM_Manufacturing_PMI", 2)
    ]

    for fam, indicator, lag in key_pairs:
        if fam in monthly_family.columns and indicator in m_indexed.columns:
            # Pre-2022 correlation
            x_pre = m_indexed.loc[dates_pre, indicator].values
            y_pre = monthly_family.loc[dates_pre, fam].values
            corr_pre = np.corrcoef(x_pre[:-lag], y_pre[lag:])[0, 1] if lag > 0 else np.corrcoef(x_pre, y_pre)[0, 1]

            # Post-2022 correlation
            x_post = m_indexed.loc[dates_post, indicator].values
            y_post = monthly_family.loc[dates_post, fam].values
            corr_post = np.corrcoef(x_post[:-lag], y_post[lag:])[0, 1] if lag > 0 else np.corrcoef(x_post, y_post)[0, 1]

            drift = abs(corr_post - corr_pre)
            stability = "Stable" if drift < 0.25 else "Structural Regime Shift"

            stability_records.append({
                "Product_Family": fam,
                "Macro_Indicator": indicator,
                "Lag_Months": lag,
                "Corr_Pre_2022": round(corr_pre, 3),
                "Corr_Post_2022": round(corr_post, 3),
                "Drift_Magnitude": round(drift, 3),
                "Stability_Assessment": stability
            })

    stab_df = pd.DataFrame(stability_records)
    stab_df.to_csv(REPORTS_DIR / "lag_stability_analysis.csv", index=False)
    return stab_df
