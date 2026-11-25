"""
Master Feature Pipeline & Leakage-Free Multi-Horizon Snapshot Engine (Workstream 3).
Orchestrates calendar/scoped-events, macro drivers, EV transitions, intermittency,
and strictly causal origin-anchored feature generation across 19-month horizons.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from src.config import FEATURES_DATA_DIR, REPORTS_DIR
from src.features.intermittency import compute_dataset_intermittency
from src.features.calendar_events import add_calendar_and_event_features
from src.features.macro_drivers import add_macro_features, analyze_macro_lead_lags, test_lag_stability
from src.features.ev_transition import add_ev_powertrain_features, analyze_ev_shift


def _ym_to_int(ym_series: pd.Series) -> np.ndarray:
    """Vectorized YYYY-MM -> total months integer for exact horizon arithmetic."""
    s = ym_series.astype(str)
    return s.str[:4].astype(int).values * 12 + s.str[5:7].astype(int).values


def attach_strict_causal_features(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    origin_date: str,
    sku_master: pd.DataFrame,
    target_col: str = "Units_Sold_Unconstrained",
) -> pd.DataFrame:
    """
    Computes SKU-Region level, trend, seasonality, intermittency, customer cadence,
    and Replaces_SKU predecessor transfer features strictly from history_df (Date <= origin_date),
    guaranteeing ZERO lookahead leakage and ZERO Month 13-19 lag collapse.
    """
    hist = history_df[history_df["Date"] <= origin_date].sort_values(["SKU_ID", "Region", "Date"]).copy()
    out = target_df.copy()

    origin_ym = int(origin_date[:4]) * 12 + int(origin_date[5:7])
    target_ym = _ym_to_int(out["Date"])
    horizon = np.clip(target_ym - origin_ym, 1, 24)
    out["horizon_months"] = horizon
    out["horizon_log"] = np.log1p(horizon)

    if hist.empty:
        return out

    hist["_ym"] = _ym_to_int(hist["Date"])
    hist["_val"] = pd.to_numeric(hist[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    hist["_nz"] = (hist["_val"] > 0).astype(float)
    hist["_months_ago"] = origin_ym - hist["_ym"]

    # 1. Trailing window levels strictly relative to origin_date (100% vectorized, zero Python lambdas)
    w3 = hist[hist["_months_ago"] < 3].groupby(["SKU_ID", "Region"])["_val"].mean()
    w6 = hist[hist["_months_ago"] < 6].groupby(["SKU_ID", "Region"])["_val"].mean()
    w6_prev = hist[(hist["_months_ago"] >= 6) & (hist["_months_ago"] < 12)].groupby(["SKU_ID", "Region"])["_val"].mean()
    w12 = hist[hist["_months_ago"] < 12].groupby(["SKU_ID", "Region"])["_val"].agg(["mean", "median", "std", "max"])
    w24 = hist[hist["_months_ago"] < 24].groupby(["SKU_ID", "Region"])["_val"].mean()
    agg_spec = {"mean_all": ("_val", "mean"), "nonzero_rate": ("_nz", "mean"), "last_ym": ("_ym", "max")}
    if "Stockout_Flag" in hist.columns:
        agg_spec["stockout_rate"] = ("Stockout_Flag", "mean")
    all_stats = hist.groupby(["SKU_ID", "Region"]).agg(**agg_spec)
    if "stockout_rate" not in all_stats.columns:
        all_stats["stockout_rate"] = 0.0

    # Exponentially weighted trailing level (recent observations weighted heavier)
    hist["_decay_w"] = np.power(0.82, np.clip(hist["_months_ago"], 0, 60))
    hist["_w_val"] = hist["_val"] * hist["_decay_w"]
    ewm_num = hist.groupby(["SKU_ID", "Region"])["_w_val"].sum()
    ewm_den = hist.groupby(["SKU_ID", "Region"])["_decay_w"].sum().clip(lower=1e-6)
    ewm_series = ewm_num / ewm_den

    # Customer ordering interval (Rule 6) strictly up to origin
    nz = hist[hist["_val"] > 0]
    nz_stats = nz.groupby(["SKU_ID", "Region"]).agg(
        nz_cnt=("_ym", "count"),
        nz_min=("_ym", "min"),
        nz_max=("_ym", "max"),
    )
    nz_iv = np.where(
        nz_stats["nz_cnt"] > 1,
        (nz_stats["nz_max"] - nz_stats["nz_min"] + 1) / np.maximum(nz_stats["nz_cnt"] - 1, 1),
        1.0,
    )
    cycle_iv_map = dict(zip(nz_stats.index, np.round(nz_iv, 2)))
    nz_last_map = dict(zip(nz_stats.index, nz_stats["nz_max"]))

    # Zero streak / intermittency & SBA strictly up to origin
    total_cnt = hist.groupby(["SKU_ID", "Region"])["_val"].count()
    nz_cnt_full = nz.groupby(["SKU_ID", "Region"])["_val"].count()
    nz_mean_full = nz.groupby(["SKU_ID", "Region"])["_val"].mean()
    nz_std_full = nz.groupby(["SKU_ID", "Region"])["_val"].std().fillna(0.0)

    adi_s = (total_cnt / nz_cnt_full.reindex(total_cnt.index).fillna(0.0).replace(0.0, np.nan)).fillna(99.0)
    cv2_s = ((nz_std_full / nz_mean_full.replace(0.0, np.nan)) ** 2).fillna(0.0)
    sba_s = (nz_mean_full / adi_s.clip(lower=1.0)) * (1.0 - cv2_s / (2.0 * adi_s.clip(lower=1.0))).clip(lower=0.0)

    # Hierarchical priors (SKU global, Product_Family x Region, Customer x Region)
    recent_h = hist[hist["_months_ago"] < 12]
    if recent_h.empty:
        recent_h = hist
    sku_global_map = recent_h.groupby("SKU_ID")["_val"].mean().to_dict()
    fam_reg_map = recent_h.groupby(["Product_Family", "Region"])["_val"].mean().to_dict()
    cust_reg_map = (
        recent_h.groupby(["Customer_Name", "Region"])["_val"].mean().to_dict()
        if "Customer_Name" in recent_h.columns else {}
    )

    # Same-calendar-month seasonality table strictly <= origin_date
    same_mo = hist.groupby(["SKU_ID", "Region", "Month"])["_val"].mean().to_dict()
    # Latest and 2-yr prior exact calendar month values
    hist_yr = hist["Year"].values
    origin_year = int(origin_date[:4])
    last_year_exact = hist.sort_values("Date").groupby(["SKU_ID", "Region", "Month"])["_val"].last().to_dict()
    prior2_exact = (
        hist[hist["_months_ago"] >= 12]
        .sort_values("Date")
        .groupby(["SKU_ID", "Region", "Month"])["_val"]
        .last()
        .to_dict()
    )

    # Family-level monthly seasonality index (helpful fallback for cold starts / MEA new SKUs)
    fam_mo_mean = recent_h.groupby(["Product_Family", "Month"])["_val"].mean()
    fam_overall_mean = recent_h.groupby("Product_Family")["_val"].mean()
    fam_seas_idx = {}
    for (fam, m), v in fam_mo_mean.items():
        base = fam_overall_mean.get(fam, 1.0)
        fam_seas_idx[(fam, m)] = float(np.clip((v + 1.0) / (base + 1.0), 0.4, 2.2))

    # Build predecessor lookup via Replaces_SKU
    sku_replaces = dict(zip(sku_master["SKU_ID"], sku_master["Replaces_SKU"]))

    # Map series-level dicts cleanly via tuples
    keys = list(zip(out["SKU_ID"], out["Region"]))
    m_all_d = all_stats["mean_all"].to_dict()
    nz_rate_d = all_stats["nonzero_rate"].to_dict()
    so_rate_d = all_stats["stockout_rate"].to_dict()
    w3_d = w3.to_dict()
    w6_d = w6.to_dict()
    w6p_d = w6_prev.to_dict()
    w12_m_d = w12["mean"].to_dict()
    w12_med_d = w12["median"].to_dict()
    w12_std_d = w12["std"].fillna(0.0).to_dict()
    w24_d = w24.to_dict()
    ewm_d = ewm_series.to_dict()
    adi_d = adi_s.to_dict()
    cv2_d = cv2_s.to_dict()
    sba_d = sba_s.fillna(0.0).to_dict()

    fam_arr = out["Product_Family"].values
    reg_arr = out["Region"].values
    sku_arr = out["SKU_ID"].values
    cust_arr = out["Customer_Name"].values if "Customer_Name" in out.columns else [""] * len(out)
    mo_arr = out["Month"].values

    fam_prior = np.array([fam_reg_map.get((f, r), 15.0) for f, r in zip(fam_arr, reg_arr)], dtype=float)
    sku_prior = np.array([sku_global_map.get(s, fp) for s, fp in zip(sku_arr, fam_prior)], dtype=float)
    cust_prior = np.array([cust_reg_map.get((c, r), fp) for c, r, fp in zip(cust_arr, reg_arr, fam_prior)], dtype=float)

    # Direct series stats with hierarchical shrinkage when short history
    mean12 = np.array([w12_m_d.get(k, w24_d.get(k, m_all_d.get(k, np.nan))) for k in keys], dtype=float)
    has_hist = ~np.isnan(mean12)
    mean12_filled = np.where(has_hist, mean12, 0.5 * sku_prior + 0.5 * fam_prior)

    mean3 = np.array([w3_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    mean6 = np.array([w6_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    prev6 = np.array([w6p_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    mean24 = np.array([w24_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    ewm_val = np.array([ewm_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    med12 = np.array([w12_med_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    std12 = np.array([w12_std_d.get(k, 0.0) for k in keys], dtype=float)
    mean_all = np.array([m_all_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)

    out["origin_mean_3m"] = mean3
    out["origin_mean_6m"] = mean6
    out["origin_mean_12m"] = mean12_filled
    out["origin_mean_24m"] = mean24
    out["origin_median_12m"] = med12
    out["origin_std_12m"] = std12
    out["origin_ewm"] = ewm_val
    out["origin_trend_ratio"] = np.clip((mean6 + 1.0) / (prev6 + 1.0), 0.5, 2.0)
    out["origin_nonzero_rate"] = np.array([nz_rate_d.get(k, 0.7) for k in keys], dtype=float)
    out["historical_stockout_rate"] = np.array([so_rate_d.get(k, 0.0) for k in keys], dtype=float)
    out["sku_global_prior"] = sku_prior
    out["family_region_prior"] = fam_prior
    out["customer_region_prior"] = cust_prior

    # Intermittency (strictly <= origin)
    adi_vals = np.array([adi_d.get(k, 1.5) for k in keys], dtype=float)
    cv2_vals = np.array([cv2_d.get(k, 0.35) for k in keys], dtype=float)
    sba_vals = np.array([sba_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    out["adi"] = adi_vals
    out["cv2"] = cv2_vals
    out["sba_baseline_rate"] = sba_vals
    out["croston_baseline_rate"] = sba_vals

    interm_cat = []
    for a, c in zip(adi_vals, cv2_vals):
        if a < 1.32 and c < 0.49:
            interm_cat.append("Smooth")
        elif a >= 1.32 and c < 0.49:
            interm_cat.append("Intermittent")
        elif a < 1.32 and c >= 0.49:
            interm_cat.append("Erratic")
        else:
            interm_cat.append("Lumpy")
    out["intermittency_category"] = interm_cat

    # Ordering cycle timing relative to target month (Rule 6)
    cycle_iv = np.array([cycle_iv_map.get(k, 1.0) for k in keys], dtype=float)
    last_nz = np.array([nz_last_map.get(k, origin_ym - 1) for k in keys], dtype=float)
    months_since_nz = np.maximum(0.0, target_ym - last_nz)
    out["customer_cycle_months"] = cycle_iv
    out["months_since_last_order"] = months_since_nz
    # Phase match with multi-month ordering cadence (1 when expected cycle hits, else decaying)
    mod_dist = np.abs(np.mod(months_since_nz, np.maximum(cycle_iv, 1.0)))
    out["is_expected_cycle_hit"] = ((cycle_iv <= 1.4) | (mod_dist < 0.6) | (np.abs(cycle_iv - mod_dist) < 0.6)).astype(int)

    # Predecessor Replaces_SKU transfer prior strictly <= origin
    pred_level = []
    for s, r, f, h_ok, m12_v in zip(sku_arr, reg_arr, fam_arr, has_hist, mean12_filled):
        rep = sku_replaces.get(s, "")
        if pd.notna(rep) and str(rep).strip() and str(rep).strip() != "nan":
            rep_k = (str(rep).strip(), r)
            pred_v = w12_m_d.get(rep_k, m_all_d.get(rep_k, fam_reg_map.get((f, r), 15.0)))
        else:
            pred_v = m12_v if h_ok else fam_reg_map.get((f, r), 15.0)
        pred_level.append(float(pred_v))
    out["predecessor_prior"] = np.array(pred_level, dtype=float)

    # Same-calendar-month seasonal lookup (zero Month 13-19 drop!)
    # For h <= 12: exact lag_12 is same_month_last_obs; for h > 12: same calendar month 2 yrs prior (or latest observed)!
    sm_hist_mean = []
    sm_last_obs = []
    seas_idx = []
    for s, r, f, m, h, ma, f_pr in zip(sku_arr, reg_arr, fam_arr, mo_arr, horizon, mean_all, fam_prior):
        k3 = (s, r, int(m))
        hm = same_mo.get(k3, np.nan)
        if np.isnan(hm):
            f_idx = fam_seas_idx.get((f, int(m)), 1.0)
            hm = ma * f_idx
            lo = hm
            idx_v = f_idx
        else:
            idx_v = float(np.clip((hm + 1.0) / (ma + 1.0), 0.3, 2.8))
            lo = last_year_exact.get(k3, hm) if h <= 12 else prior2_exact.get(k3, last_year_exact.get(k3, hm))
        sm_hist_mean.append(float(hm))
        sm_last_obs.append(float(lo))
        seas_idx.append(idx_v)

    sm_hist_arr = np.array(sm_hist_mean, dtype=float)
    sm_last_arr = np.array(sm_last_obs, dtype=float)
    seas_idx_arr = np.array(seas_idx, dtype=float)

    out["same_month_hist_mean"] = sm_hist_arr
    out["same_month_last_obs"] = sm_last_arr
    out["seasonal_index"] = seas_idx_arr
    # Anchor combining recent exponentially weighted level with calendar month seasonality
    out["seasonal_ewm_anchor"] = (0.60 * ewm_val + 0.40 * mean12_filled) * seas_idx_arr
    # Backwards compatibility aliases (populated cleanly across all 19 horizons!)
    out["lag_12"] = sm_last_arr
    out["rolling_mean_12"] = mean12_filled
    out["rolling_std_12"] = std12

    return out


def build_rolling_origin_training_set(
    panel_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    max_origin_date: str,
    target_col: str = "Units_Sold_Unconstrained",
    origin_step_months: int = 16,
    snapshot_cache: dict = None,
) -> pd.DataFrame:
    """
    Constructs multi-origin horizon-consistent training samples (origin, target_date)
    strictly using dates <= max_origin_date so train features match 19-month test inference identically.
    Reuses cached origin snapshots safely without ever exposing future dates > max_origin_date.
    """
    hist_panel = panel_df[panel_df["Date"] <= max_origin_date].copy()
    dates = sorted(hist_panel["Date"].unique())
    if len(dates) < 20:
        return attach_strict_causal_features(hist_panel, hist_panel, max_origin_date, sku_master, target_col)

    candidate_origins = []
    start_idx = 14
    for i in range(start_idx, len(dates) - 3, origin_step_months):
        candidate_origins.append(dates[i])
    if len(dates) - 1 - 12 >= start_idx:
        candidate_origins.append(dates[len(dates) - 1 - 12])
    candidate_origins = sorted(set(candidate_origins))

    slices = []
    for orig in candidate_origins:
        avail_future = [d for d in dates if d > orig][:19]
        if not avail_future:
            continue
        cache_key = (orig, avail_future[-1])
        if snapshot_cache is not None and cache_key in snapshot_cache:
            slices.append(snapshot_cache[cache_key])
            continue
        tgt = hist_panel[hist_panel["Date"].isin(avail_future)].copy()
        enriched = attach_strict_causal_features(hist_panel, tgt, orig, sku_master, target_col)
        enriched["_origin"] = orig
        if snapshot_cache is not None:
            snapshot_cache[cache_key] = enriched
        slices.append(enriched)

    if not slices:
        return attach_strict_causal_features(hist_panel, hist_panel, max_origin_date, sku_master, target_col)

    return pd.concat(slices, ignore_index=True)


def build_feature_store(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    sku_master: pd.DataFrame,
    macro_clean: pd.DataFrame,
    events_clean: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds unified continuous panel feature store covering 2019-01 through 2028-05 across all SKUs and Regions.
    Exports train_features, test_features, and full_panel_features for strictly causal modeling.
    """
    # 1. Aggregate training data to Date-SKU-Region level keeping latest active Customer/Channel
    train_sorted = train_clean.sort_values(["Date", "SKU_ID", "Region", "Units_Sold"])
    train_agg = train_sorted.groupby(["Date", "SKU_ID", "Region"]).agg({
        "Units_Sold": "sum",
        "Units_Sold_Unconstrained": "sum",
        "Revenue_USD": "sum",
        "Unit_Price_USD": "median",
        "Product_Family": "first",
        "Business_Unit": "first",
        "Customer_Name": "last",
        "Sales_Channel": "last",
        "Stockout_Flag": "max",
        "Lost_Units": "sum",
    }).reset_index()

    # 2. Extract test keys (preserving Customer_Name & Sales_Channel for scoped forecast events like EV-062!)
    test_meta_cols = [c for c in ["Date", "SKU_ID", "Region", "Product_Family", "Business_Unit", "Customer_Name", "Sales_Channel"] if c in test_clean.columns]
    test_keys = test_clean[test_meta_cols].drop_duplicates(subset=["Date", "SKU_ID", "Region"]).copy()
    test_keys["Units_Sold"] = np.nan
    test_keys["Units_Sold_Unconstrained"] = np.nan
    test_keys["Revenue_USD"] = np.nan
    test_keys["Unit_Price_USD"] = np.nan
    test_keys["Stockout_Flag"] = 0
    test_keys["Lost_Units"] = 0

    # 3. Concatenate into continuous panel (2019-01 to 2028-05)
    panel = pd.concat([train_agg, test_keys], ignore_index=True)
    panel = panel.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)

    # 4. Add Calendar and Scoped Event features (history + forecast window EV-059..EV-068!)
    panel = add_calendar_and_event_features(panel, events_clean)

    # 5. Add Macroeconomic features and lags
    panel = add_macro_features(panel, macro_clean)

    # 6. Add EV and Powertrain transition features
    panel = add_ev_powertrain_features(panel, sku_master)

    # 7. Add Catalog Lifecycle features
    sku_launch = dict(zip(sku_master["SKU_ID"], sku_master["Launch_Date"]))
    sku_eol = dict(zip(sku_master["SKU_ID"], sku_master["EOL_Date"]))
    sku_replaces = dict(zip(sku_master["SKU_ID"], sku_master["Replaces_SKU"]))

    panel["Launch_Date"] = panel["SKU_ID"].map(sku_launch).fillna("")
    panel["EOL_Date"] = panel["SKU_ID"].map(sku_eol).fillna("")
    panel["Replaces_SKU"] = panel["SKU_ID"].map(sku_replaces).fillna("")
    panel["has_replacement_sku"] = ((panel["Replaces_SKU"] != "") & (panel["Replaces_SKU"] != "nan")).astype(int)

    cur_year = panel["Date"].str[:4].astype(int)
    cur_month = panel["Date"].str[5:7].astype(int)

    launch_yr = pd.to_numeric(panel["Launch_Date"].str[:4], errors="coerce").fillna(2019).astype(int)
    launch_mo = pd.to_numeric(panel["Launch_Date"].str[5:7], errors="coerce").fillna(1).astype(int)
    panel["months_since_launch"] = (cur_year - launch_yr) * 12 + (cur_month - launch_mo)
    panel["is_pre_launch"] = ((panel["Launch_Date"] != "") & (panel["Date"] < panel["Launch_Date"].str[:7])).astype(int)

    eol_yr = pd.to_numeric(panel["EOL_Date"].str[:4], errors="coerce").fillna(2099).astype(int)
    eol_mo = pd.to_numeric(panel["EOL_Date"].str[5:7], errors="coerce").fillna(12).astype(int)
    panel["months_until_eol"] = (eol_yr - cur_year) * 12 + (eol_mo - cur_month)
    panel["is_post_eol"] = ((panel["EOL_Date"] != "") & (panel["Date"] > panel["EOL_Date"].str[:7])).astype(int)

    # 8. Intermittency Report + Driver/Shift Analyses
    compute_dataset_intermittency(train_agg, target_col="Units_Sold_Unconstrained")
    analyze_macro_lead_lags(train_agg, macro_clean)
    test_lag_stability(train_agg, macro_clean)
    analyze_ev_shift(train_agg, sku_master)

    # Save continuous base panel (strict causal snapshots constructed per origin during backtest/forecast!)
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURES_DATA_DIR / "full_panel_features.parquet", index=False)

    # Construct final forecast snapshot strictly from origin 2026-10
    raw_train_hist = panel[panel["Date"] <= "2026-10"].copy()
    raw_test_fut = panel[panel["Date"] >= "2026-11"].copy()
    train_features = build_rolling_origin_training_set(panel, sku_master, max_origin_date="2026-10")
    test_features = attach_strict_causal_features(raw_train_hist, raw_test_fut, origin_date="2026-10", sku_master=sku_master)

    train_features.to_parquet(FEATURES_DATA_DIR / "train_features.parquet", index=False)
    test_features.to_parquet(FEATURES_DATA_DIR / "test_features.parquet", index=False)

    return train_features, test_features
