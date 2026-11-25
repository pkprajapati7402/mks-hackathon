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
    Computes strictly causal (Date <= origin_date) multi-horizon features:
    - Anomaly-despiked trailing levels & exponential smoothing (resistant to 1-month spikes/ERP gaps)
    - Customer ordering-cycle phases (mod-2 bi-monthly, mod-3 quarterly, exact month shutdown rates)
    - Empirical Bayes shrunk calendar seasonality normalized by SKU-Region & Family-Region priors
    - Origin-anchored EV saturation & macroeconomic elasticity ratios
    - Launch maturity ramp normalization & Replaces_SKU predecessor transfer priors
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

    # 1. Causal Despiked & Shock-Healed History (strictly <= origin_date)
    pos_h = hist[hist["_val"] > 0]
    pos_stats = pos_h.groupby(["SKU_ID", "Region"])["_val"].agg(
        pos_med="median",
        pos_q80=lambda s: float(np.quantile(s, 0.80)),
    )
    med_map = pos_stats["pos_med"].to_dict()
    q80_map = pos_stats["pos_q80"].to_dict()

    hist_keys = list(zip(hist["SKU_ID"], hist["Region"]))
    hist_med = np.array([med_map.get(k, 0.0) for k in hist_keys], dtype=float)
    hist_q80 = np.array([q80_map.get(k, 0.0) for k in hist_keys], dtype=float)
    spike_cap = np.maximum(5.0, np.maximum(2.5 * hist_med, 1.75 * hist_q80))

    clean_v = np.minimum(hist["_val"].values, spike_cap)
    covid_mask = hist["Date"].astype(str).between("2020-03", "2020-05").values
    erp_mask = (hist["is_erp_migration_gap"].values == 1) if "is_erp_migration_gap" in hist.columns else np.zeros(len(hist), dtype=bool)
    heal_mask = (covid_mask | erp_mask) & (clean_v < 0.45 * hist_med) & (hist_med > 0)
    clean_v = np.where(heal_mask, hist_med, clean_v)
    hist["_clean_val"] = clean_v

    # 2. Trailing Window Stats (both Raw Unconstrained & Despiked Clean)
    w3 = hist[hist["_months_ago"] < 3].groupby(["SKU_ID", "Region"])["_val"].mean()
    w6 = hist[hist["_months_ago"] < 6].groupby(["SKU_ID", "Region"])["_val"].mean()
    w6_prev = hist[(hist["_months_ago"] >= 6) & (hist["_months_ago"] < 12)].groupby(["SKU_ID", "Region"])["_val"].mean()
    w12 = hist[hist["_months_ago"] < 12].groupby(["SKU_ID", "Region"])["_val"].agg(["mean", "median", "std", "max"])
    w24 = hist[hist["_months_ago"] < 24].groupby(["SKU_ID", "Region"])["_val"].mean()

    cw6 = hist[hist["_months_ago"] < 6].groupby(["SKU_ID", "Region"])["_clean_val"].mean()
    cw12 = hist[hist["_months_ago"] < 12].groupby(["SKU_ID", "Region"])["_clean_val"].agg(["mean", "median", "std"])
    cw24 = hist[hist["_months_ago"] < 24].groupby(["SKU_ID", "Region"])["_clean_val"].mean()
    c_all = hist.groupby(["SKU_ID", "Region"])["_clean_val"].mean()

    agg_spec = {"mean_all": ("_val", "mean"), "nonzero_rate": ("_nz", "mean"), "last_ym": ("_ym", "max")}
    if "Stockout_Flag" in hist.columns:
        agg_spec["stockout_rate"] = ("Stockout_Flag", "mean")
    all_stats = hist.groupby(["SKU_ID", "Region"]).agg(**agg_spec)
    if "stockout_rate" not in all_stats.columns:
        all_stats["stockout_rate"] = 0.0

    # Exponentially weighted clean & raw levels (decay 0.84 -> responsive without 1-month overreaction)
    decay_w = np.power(0.84, np.clip(hist["_months_ago"].values, 0, 60))
    hist["_decay_w"] = decay_w
    hist["_w_val"] = hist["_val"].values * decay_w
    hist["_w_clean"] = hist["_clean_val"].values * decay_w
    grp_ewm = hist.groupby(["SKU_ID", "Region"])[["_w_val", "_w_clean", "_decay_w"]].sum()
    ewm_den = grp_ewm["_decay_w"].clip(lower=1e-6)
    ewm_series = grp_ewm["_w_val"] / ewm_den
    clean_ewm_series = grp_ewm["_w_clean"] / ewm_den

    # 3. Recent-Regime Intermittency, SBA & Ordering-Phase Matrix (Rule 6)
    nz_recent = hist[(hist["_val"] > 0) & (hist["_months_ago"] < 42)].copy()
    if nz_recent.empty:
        nz_recent = pos_h.copy()
    nz_recent["_gap"] = nz_recent.groupby(["SKU_ID", "Region"])["_ym"].diff()
    med_gap_s = nz_recent.groupby(["SKU_ID", "Region"])["_gap"].median().clip(lower=1.0, upper=12.0)

    nz = pos_h
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
    rec_cycle_map = med_gap_s.to_dict()
    nz_last_map = dict(zip(nz_stats.index, nz_stats["nz_max"]))

    # Recent periodicity phase probabilities (Bi-Monthly mod-2, Quarterly mod-3, Exact Calendar Month)
    hist_phase = hist[hist["_months_ago"] < 42].copy()
    if hist_phase.empty:
        hist_phase = hist.copy()
    hist_phase["_m2"] = hist_phase["_ym"] % 2
    hist_phase["_m3"] = hist_phase["_ym"] % 3
    mod2_map = hist_phase.groupby(["SKU_ID", "Region", "_m2"])["_nz"].mean().to_dict()
    mod3_map = hist_phase.groupby(["SKU_ID", "Region", "_m3"])["_nz"].mean().to_dict()

    hist_48 = hist[hist["_months_ago"] < 48]
    if hist_48.empty:
        hist_48 = hist
    sm_nz_stats = hist_48.groupby(["SKU_ID", "Region", "Month"])["_nz"].agg(["mean", "count"])
    sm_nz_map = sm_nz_stats["mean"].to_dict()
    sm_cnt_map = sm_nz_stats["count"].to_dict()

    nz_r6_map = hist[hist["_months_ago"] < 6].groupby(["SKU_ID", "Region"])["_nz"].mean().to_dict()
    nz_r12_map = hist[hist["_months_ago"] < 12].groupby(["SKU_ID", "Region"])["_nz"].mean().to_dict()
    nz_r24_map = hist[hist["_months_ago"] < 24].groupby(["SKU_ID", "Region"])["_nz"].mean().to_dict()

    # Zero streak / intermittency & SBA strictly <= origin
    total_cnt = hist.groupby(["SKU_ID", "Region"])["_val"].count()
    nz_cnt_full = nz.groupby(["SKU_ID", "Region"])["_val"].count()
    nz_mean_full = nz.groupby(["SKU_ID", "Region"])["_val"].mean()
    nz_std_full = nz.groupby(["SKU_ID", "Region"])["_val"].std().fillna(0.0)

    adi_s = (total_cnt / nz_cnt_full.reindex(total_cnt.index).fillna(0.0).replace(0.0, np.nan)).fillna(99.0)
    cv2_s = ((nz_std_full / nz_mean_full.replace(0.0, np.nan)) ** 2).fillna(0.0)
    sba_s = (nz_mean_full / adi_s.clip(lower=1.0)) * (1.0 - cv2_s / (2.0 * adi_s.clip(lower=1.0))).clip(lower=0.0)

    # 4. Hierarchical Priors & SKU Price (strictly <= origin)
    recent_h = hist[hist["_months_ago"] < 12]
    if recent_h.empty:
        recent_h = hist
    recent_24 = hist[hist["_months_ago"] < 24]
    if recent_24.empty:
        recent_24 = hist

    sku_global_map = recent_h.groupby("SKU_ID")["_clean_val"].mean().to_dict()
    fam_reg_map = recent_h.groupby(["Product_Family", "Region"])["_clean_val"].mean().to_dict()
    cust_reg_map = (
        recent_h.groupby(["Customer_Name", "Region"])["_clean_val"].mean().to_dict()
        if "Customer_Name" in recent_h.columns else {}
    )
    sku_px_map = (
        recent_h.groupby("SKU_ID")["Unit_Price_USD"].median().to_dict()
        if "Unit_Price_USD" in recent_h.columns else {}
    )
    med_px_global = float(recent_h["Unit_Price_USD"].median()) if "Unit_Price_USD" in recent_h.columns and recent_h["Unit_Price_USD"].notna().any() else 120.0

    # Same-calendar-month clean averages strictly <= origin_date
    same_mo_clean = hist_48.groupby(["SKU_ID", "Region", "Month"])["_clean_val"].mean().to_dict()
    same_mo_raw = hist.groupby(["SKU_ID", "Region", "Month"])["_val"].mean().to_dict()
    last_year_exact = hist.sort_values("Date").groupby(["SKU_ID", "Region", "Month"])["_clean_val"].last().to_dict()
    prior2_exact = (
        hist[hist["_months_ago"] >= 12]
        .sort_values("Date")
        .groupby(["SKU_ID", "Region", "Month"])["_clean_val"]
        .last()
        .to_dict()
    )

    # Family-Region-Month seasonal index (robust hierarchical EB prior)
    fam_reg_mo_mean = recent_24.groupby(["Product_Family", "Region", "Month"])["_clean_val"].mean()
    fam_reg_base = recent_24.groupby(["Product_Family", "Region"])["_clean_val"].mean()
    fam_reg_seas_idx = {}
    for (fam, reg, m), v in fam_reg_mo_mean.items():
        base = fam_reg_base.get((fam, reg), 1.0)
        fam_reg_seas_idx[(fam, reg, int(m))] = float(np.clip((v + 1.0) / (base + 1.0), 0.35, 2.5))

    fam_mo_mean = recent_24.groupby(["Product_Family", "Month"])["_clean_val"].mean()
    fam_overall_mean = recent_24.groupby("Product_Family")["_clean_val"].mean()
    fam_seas_idx = {}
    for (fam, m), v in fam_mo_mean.items():
        base = fam_overall_mean.get(fam, 1.0)
        fam_seas_idx[(fam, int(m))] = float(np.clip((v + 1.0) / (base + 1.0), 0.35, 2.5))

    sku_replaces = dict(zip(sku_master["SKU_ID"], sku_master["Replaces_SKU"]))
    sku_launch_map = dict(zip(sku_master["SKU_ID"], sku_master["Launch_Date"]))

    # Map series-level dicts cleanly via tuples
    keys = list(zip(out["SKU_ID"], out["Region"]))
    m_all_d = all_stats["mean_all"].to_dict()
    c_all_d = c_all.to_dict()
    nz_rate_d = all_stats["nonzero_rate"].to_dict()
    so_rate_d = all_stats["stockout_rate"].to_dict()
    w3_d = w3.to_dict()
    w6_d = w6.to_dict()
    w6p_d = w6_prev.to_dict()
    w12_m_d = w12["mean"].to_dict()
    w12_med_d = w12["median"].to_dict()
    w12_std_d = w12["std"].fillna(0.0).to_dict()
    w24_d = w24.to_dict()
    cw6_d = cw6.to_dict()
    cw12_m_d = cw12["mean"].to_dict()
    cw12_med_d = cw12["median"].to_dict()
    cw12_std_d = cw12["std"].fillna(0.0).to_dict()
    cw24_d = cw24.to_dict()
    ewm_d = ewm_series.to_dict()
    cewm_d = clean_ewm_series.to_dict()
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
    px_arr = np.array([sku_px_map.get(s, med_px_global) for s in sku_arr], dtype=float)
    out["sku_price_log"] = np.log1p(np.clip(px_arr, 1.0, None))

    mean12 = np.array([w12_m_d.get(k, w24_d.get(k, m_all_d.get(k, np.nan))) for k in keys], dtype=float)
    has_hist = ~np.isnan(mean12)
    prior_blend = 0.5 * sku_prior + 0.5 * fam_prior
    mean12_filled = np.where(has_hist, mean12, prior_blend)

    clean12 = np.array([cw12_m_d.get(k, cw24_d.get(k, c_all_d.get(k, np.nan))) for k in keys], dtype=float)
    clean12_filled = np.where(~np.isnan(clean12), clean12, prior_blend)

    mean3 = np.array([w3_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    mean6 = np.array([w6_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    prev6 = np.array([w6p_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    mean24 = np.array([w24_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    ewm_val = np.array([ewm_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    med12 = np.array([w12_med_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)
    std12 = np.array([w12_std_d.get(k, 0.0) for k in keys], dtype=float)
    mean_all = np.array([m_all_d.get(k, m12) for k, m12 in zip(keys, mean12_filled)], dtype=float)

    clean6 = np.array([cw6_d.get(k, c12) for k, c12 in zip(keys, clean12_filled)], dtype=float)
    clean24 = np.array([cw24_d.get(k, c12) for k, c12 in zip(keys, clean12_filled)], dtype=float)
    clean_ewm = np.array([cewm_d.get(k, c12) for k, c12 in zip(keys, clean12_filled)], dtype=float)
    clean_med12 = np.array([cw12_med_d.get(k, c12) for k, c12 in zip(keys, clean12_filled)], dtype=float)
    clean_std12 = np.array([cw12_std_d.get(k, 0.0) for k in keys], dtype=float)
    clean_all = np.array([c_all_d.get(k, c12) for k, c12 in zip(keys, clean12_filled)], dtype=float)

    out["origin_mean_3m"] = mean3
    out["origin_mean_6m"] = mean6
    out["origin_mean_12m"] = mean12_filled
    out["origin_mean_24m"] = mean24
    out["origin_median_12m"] = med12
    out["origin_std_12m"] = std12
    out["origin_ewm"] = ewm_val
    out["origin_trend_ratio"] = np.clip((clean6 + 1.0) / (prev6 + 1.0), 0.55, 1.85)
    out["clean_mean_6m"] = clean6
    out["clean_mean_12m"] = clean12_filled
    out["clean_mean_24m"] = clean24
    out["clean_ewm"] = clean_ewm
    out["clean_median_12m"] = clean_med12
    out["clean_std_12m"] = clean_std12
    out["clean_cv"] = np.clip(clean_std12 / np.maximum(clean12_filled, 1.0), 0.0, 4.0)

    nz_all = np.array([nz_rate_d.get(k, 0.7) for k in keys], dtype=float)
    out["origin_nonzero_rate"] = nz_all
    out["recent_6m_nz_rate"] = np.array([nz_r6_map.get(k, nz_a) for k, nz_a in zip(keys, nz_all)], dtype=float)
    out["recent_12m_nz_rate"] = np.array([nz_r12_map.get(k, nz_a) for k, nz_a in zip(keys, nz_all)], dtype=float)
    out["recent_24m_nz_rate"] = np.array([nz_r24_map.get(k, nz_a) for k, nz_a in zip(keys, nz_all)], dtype=float)
    out["historical_stockout_rate"] = np.array([so_rate_d.get(k, 0.0) for k in keys], dtype=float)
    out["sku_global_prior"] = sku_prior
    out["family_region_prior"] = fam_prior
    out["customer_region_prior"] = cust_prior

    # Intermittency (strictly <= origin)
    adi_vals = np.array([adi_d.get(k, 1.5) for k in keys], dtype=float)
    cv2_vals = np.array([cv2_d.get(k, 0.35) for k in keys], dtype=float)
    sba_vals = np.array([sba_d.get(k, c12) for k, c12 in zip(keys, clean12_filled)], dtype=float)
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

    # Ordering cycle timing & periodic phase alignment (Rule 6: catches mod-2, mod-3 & shutdown months)
    cycle_iv = np.array([cycle_iv_map.get(k, 1.0) for k in keys], dtype=float)
    rec_cycle = np.array([rec_cycle_map.get(k, ci) for k, ci in zip(keys, cycle_iv)], dtype=float)
    last_nz = np.array([nz_last_map.get(k, origin_ym - 1) for k in keys], dtype=float)
    months_since_nz = np.maximum(0.0, target_ym - last_nz)
    out["customer_cycle_months"] = rec_cycle
    out["months_since_last_order"] = months_since_nz
    out["months_since_last_nz_at_origin"] = np.maximum(0.0, origin_ym - last_nz)

    mod_dist = np.abs(np.mod(months_since_nz, np.maximum(rec_cycle, 1.0)))
    out["is_expected_cycle_hit"] = ((rec_cycle <= 1.35) | (mod_dist < 0.55) | (np.abs(rec_cycle - mod_dist) < 0.55)).astype(int)

    t_m2 = target_ym % 2
    t_m3 = target_ym % 3
    m2_same = np.array([mod2_map.get((s, r, int(p2)), nz_a) for (s, r), p2, nz_a in zip(keys, t_m2, out["recent_24m_nz_rate"])], dtype=float)
    m2_opp = np.array([mod2_map.get((s, r, int((p2 + 1) % 2)), nz_a) for (s, r), p2, nz_a in zip(keys, t_m2, out["recent_24m_nz_rate"])], dtype=float)
    m3_same = np.array([mod3_map.get((s, r, int(p3)), nz_a) for (s, r), p3, nz_a in zip(keys, t_m3, out["recent_24m_nz_rate"])], dtype=float)
    m3_opp = np.array([
        max(mod3_map.get((s, r, int((p3 + 1) % 3)), nz_a), mod3_map.get((s, r, int((p3 + 2) % 3)), nz_a))
        for (s, r), p3, nz_a in zip(keys, t_m3, out["recent_24m_nz_rate"])
    ], dtype=float)

    out["ym_mod2_nz_rate"] = m2_same
    out["mod2_phase_advantage"] = m2_same - m2_opp
    out["ym_mod3_nz_rate"] = m3_same
    out["mod3_phase_advantage"] = m3_same - m3_opp
    sm_nz_rate = np.array([sm_nz_map.get((s, r, int(m)), nz_a) for (s, r), m, nz_a in zip(keys, mo_arr, nz_all)], dtype=float)
    sm_obs_cnt = np.array([sm_cnt_map.get((s, r, int(m)), 0) for (s, r), m in zip(keys, mo_arr)], dtype=float)
    out["same_month_nz_rate"] = sm_nz_rate
    out["same_month_obs_cnt"] = sm_obs_cnt

    # Predecessor Replaces_SKU transfer prior strictly <= origin
    pred_level = []
    for s, r, f, h_ok, c12_v in zip(sku_arr, reg_arr, fam_arr, has_hist, clean12_filled):
        rep = sku_replaces.get(s, "")
        if pd.notna(rep) and str(rep).strip() and str(rep).strip() != "nan":
            rep_k = (str(rep).strip(), r)
            pred_v = cw12_m_d.get(rep_k, c_all_d.get(rep_k, fam_reg_map.get((f, r), 15.0)))
        else:
            pred_v = c12_v if h_ok else fam_reg_map.get((f, r), 15.0)
        pred_level.append(float(pred_v))
    out["predecessor_prior"] = np.array(pred_level, dtype=float)

    # Empirical Bayes Shrunk Same-Calendar-Month Seasonality (zero Month 13-19 drop!)
    sm_clean_mean = []
    sm_last_obs = []
    seas_idx = []
    fam_mo_idx_arr = []
    for s, r, f, m, h, ca, f_pr, nz_sm, n_sm in zip(
        sku_arr, reg_arr, fam_arr, mo_arr, horizon, clean_all, fam_prior, sm_nz_rate, sm_obs_cnt
    ):
        k3 = (s, r, int(m))
        f_reg_idx = fam_reg_seas_idx.get((f, r, int(m)), fam_seas_idx.get((f, int(m)), 1.0))
        fam_mo_idx_arr.append(f_reg_idx)

        hm = same_mo_clean.get(k3, np.nan)
        if np.isnan(hm):
            hm = ca * f_reg_idx
            lo = hm
            idx_v = f_reg_idx
        else:
            raw_idx = float((hm + 0.5) / (ca + 0.5))
            idx_shrunk = float((n_sm * raw_idx + 2.0 * f_reg_idx) / (n_sm + 2.0))
            low_floor = 0.0 if (n_sm >= 3 and nz_sm == 0.0) else 0.25
            idx_v = float(np.clip(idx_shrunk, low_floor, 2.8))
            lo = last_year_exact.get(k3, hm) if h <= 12 else prior2_exact.get(k3, last_year_exact.get(k3, hm))
        sm_clean_mean.append(float(hm))
        sm_last_obs.append(float(lo))
        seas_idx.append(idx_v)

    sm_hist_arr = np.array(sm_clean_mean, dtype=float)
    sm_last_arr = np.array(sm_last_obs, dtype=float)
    seas_idx_arr = np.array(seas_idx, dtype=float)

    out["same_month_hist_mean"] = sm_hist_arr
    out["same_month_last_obs"] = sm_last_arr
    out["seasonal_index"] = seas_idx_arr
    out["fam_reg_month_index"] = np.array(fam_mo_idx_arr, dtype=float)

    # 5. Origin-Anchored Relative Macro & EV-Saturation Multipliers (fixes EV plateau overshoot!)
    orig_rows = hist[hist["Date"] == origin_date]
    if orig_rows.empty:
        orig_rows = hist.sort_values("Date").tail(1)
    orig_row = orig_rows.iloc[0]

    def _ratio(col, default_val=100.0):
        raw_base = pd.to_numeric(orig_row.get(col, default_val), errors="coerce")
        base = float(default_val if pd.isna(raw_base) else raw_base)
        cur = pd.to_numeric(out[col], errors="coerce").fillna(base).values if col in out.columns else np.full(len(out), base)
        return np.clip(cur / max(abs(base), 1.0), 0.65, 2.20), cur - base

    ev_ratio, ev_delta = _ratio("EV_Adoption_Index", 80.0)
    pmi_ratio, pmi_delta = _ratio("ISM_Manufacturing_PMI", 50.0)
    ord_ratio, _ = _ratio("New_Vehicle_Orders_Index_lag3", 100.0)
    rm_ratio, _ = _ratio("Raw_Material_Cost_Index", 100.0)
    raw_auto = pd.to_numeric(orig_row.get("Auto_Production_YoY_Pct_lag1", 0.0), errors="coerce")
    base_auto = float(0.0 if pd.isna(raw_auto) else raw_auto)
    auto_delta = (
        pd.to_numeric(out["Auto_Production_YoY_Pct_lag1"], errors="coerce").fillna(base_auto).values - base_auto
        if "Auto_Production_YoY_Pct_lag1" in out.columns else np.zeros(len(out))
    )

    is_ev = (out["Powertrain_Application"].values == "EV") if "Powertrain_Application" in out.columns else np.zeros(len(out), dtype=bool)
    is_ice = (out["Powertrain_Application"].values == "ICE") if "Powertrain_Application" in out.columns else np.zeros(len(out), dtype=bool)
    ev_mult = np.where(is_ev, np.power(ev_ratio, 1.15), np.where(is_ice, np.power(ev_ratio, -0.25), 1.0))

    raw_orig_ev = pd.to_numeric(orig_row.get("EV_Adoption_Index", 80.0), errors="coerce")
    out["ev_adoption_at_origin"] = float(80.0 if pd.isna(raw_orig_ev) else raw_orig_ev)
    out["ev_adoption_ratio"] = ev_ratio
    out["ev_adoption_delta"] = ev_delta
    out["ev_demand_multiplier"] = ev_mult
    out["pmi_delta_origin"] = pmi_delta
    out["orders_ratio_origin"] = ord_ratio
    out["raw_mat_ratio_origin"] = rm_ratio
    out["auto_yoy_delta_origin"] = auto_delta

    # Launch maturity ramp gain for recently launched SKUs
    launch_s = out["SKU_ID"].map(sku_launch_map).fillna("").astype(str).str[:7]
    has_l = (launch_s != "") & (launch_s != "nan")
    ly = pd.to_numeric(launch_s.str[:4], errors="coerce").fillna(2019).astype(int).values
    lm = pd.to_numeric(launch_s.str[5:7], errors="coerce").fillna(1).astype(int).values
    launch_ym = ly * 12 + lm
    m_live_orig = np.where(has_l, origin_ym - launch_ym, 99)
    m_live_tgt = np.where(has_l, target_ym - launch_ym, 99)

    def _ramp(m):
        return 1.0 / (1.0 + np.exp(-0.65 * np.clip(m - 2.5, -10.0, 30.0)))

    ramp_gain = np.where(
        (m_live_orig >= 0) & (m_live_orig < 10) & (m_live_tgt > m_live_orig),
        np.clip(_ramp(m_live_tgt) / np.maximum(_ramp(m_live_orig), 0.30), 1.0, 2.0),
        1.0,
    )
    out["launch_ramp_gain"] = ramp_gain

    # Structural Analytical Anchor combining despiked EWM level, shrunk seasonality, EV curve & phase probability
    clean_level = 0.55 * clean_ewm + 0.45 * clean12_filled
    phase_gate = np.where(
        (sm_obs_cnt >= 3) & (sm_nz_rate == 0.0),
        0.0,
        np.where((out["mod2_phase_advantage"].values < -0.55) | (out["mod3_phase_advantage"].values < -0.55), 0.15, 1.0),
    )
    anchor = np.clip(clean_level * seas_idx_arr * ev_mult * ramp_gain * phase_gate, 0.0, None)
    out["analytical_anchor"] = anchor
    out["seasonal_ewm_anchor"] = anchor

    # Backwards compatibility aliases (populated cleanly across all 19 horizons!)
    out["lag_12"] = sm_last_arr
    out["rolling_mean_12"] = clean12_filled
    out["rolling_std_12"] = clean_std12

    return out


def build_rolling_origin_training_set(
    panel_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    max_origin_date: str,
    target_col: str = "Units_Sold_Unconstrained",
    origin_step_months: int = 12,
    snapshot_cache: dict = None,
) -> pd.DataFrame:
    """
    Constructs multi-origin horizon-consistent training samples (origin, target_date)
    strictly using dates <= max_origin_date so train features match 19-month test inference identically.
    Includes max_origin - 19 and max_origin - 12 so validation covers all 1..19 horizons.
    """
    hist_panel = panel_df[panel_df["Date"] <= max_origin_date].copy()
    dates = sorted(hist_panel["Date"].unique())
    if len(dates) < 20:
        return attach_strict_causal_features(hist_panel, hist_panel, max_origin_date, sku_master, target_col)

    candidate_origins = []
    start_idx = 14
    for i in range(start_idx, len(dates) - 4, origin_step_months):
        candidate_origins.append(dates[i])
    for back in (12, 19):
        if len(dates) - 1 - back >= start_idx:
            candidate_origins.append(dates[len(dates) - 1 - back])
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
