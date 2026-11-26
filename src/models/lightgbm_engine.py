"""
Multi-Paradigm Hybrid Ensemble, Conformal Quantiles & Tree-SHAP Engine (Workstream 4 Expert+ Tier).
Combines genuinely diverse ML & Deep Learning architectures:
  1. Two-Stage Zero-Inflated Hurdle (LGBMClassifier P(Y>0) * Positive-Only LGBM Tweedie)
  2. Global LightGBM Tweedie Regressor (volume-unbiased with early stopping)
  3. Global LightGBM Multiplicative Log-Ratio Residual (scale-invariant extrapolator around analytical anchor)
  4. Scikit-Learn HistGradientBoostingRegressor (Poisson Deviance orthogonal histogram learner)
  5. PyTorch Deep Tabular Entity-Embedding Residual Network (gpu/cpu adaptive, Smooth-L1 around anchor)
  6. Empirical Bayes Structural Analytical Anchor (despiked EWM * EB seasonality * EV saturation)
Blends via constrained SLSQP out-of-sample WAPE + Bias optimizer + Hurdle zero-gating,
calibrates P10/P90 with Scaled Horizon-Stratified CQR, and exports Tree-SHAP driver attributions.
"""

from typing import Dict, Any, List, Tuple
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from src.config import RANDOM_SEED, REPORTS_DIR, OUTPUTS_DIR

FEATURE_COLUMNS = [
    # Origin-Anchored Raw & Despiked Clean Levels & Momentum (strictly causal <= origin)
    "origin_mean_3m", "origin_mean_6m", "origin_mean_12m", "origin_mean_24m",
    "origin_median_12m", "origin_std_12m", "origin_ewm", "origin_trend_ratio",
    "clean_mean_6m", "clean_mean_12m", "clean_mean_24m", "clean_ewm",
    "clean_median_12m", "clean_std_12m", "clean_cv",
    "origin_nonzero_rate", "recent_6m_nz_rate", "recent_12m_nz_rate", "recent_24m_nz_rate",
    "historical_stockout_rate",
    # Seasonality, EB Priors & Analytical Anchor
    "same_month_hist_mean", "same_month_last_obs", "seasonal_index",
    "fam_reg_month_index", "analytical_anchor", "seasonal_ewm_anchor",
    "sku_global_prior", "family_region_prior", "customer_region_prior", "sku_price_log",
    "predecessor_prior", "horizon_months", "horizon_log",
    # Customer Ordering Cycle & Periodicity Phase Matrix (Rule 6)
    "customer_cycle_months", "months_since_last_order", "months_since_last_nz_at_origin",
    "is_expected_cycle_hit", "ym_mod2_nz_rate", "mod2_phase_advantage",
    "ym_mod3_nz_rate", "mod3_phase_advantage", "same_month_nz_rate", "same_month_obs_cnt",
    # Cyclical Harmonics, Holidays & Scoped Calendar Events
    "Month", "Quarter", "month_sin", "month_cos", "month_sin2", "month_cos2",
    "month_sin3", "month_cos3", "quarter_sin", "quarter_cos",
    "is_europe_summer_shutdown", "is_na_retool_month", "is_golden_week_month",
    "is_cny_month", "is_ramadan_month", "is_year_end_holiday",
    "is_pre_price_hike_month", "is_post_price_hike_month", "announced_price_hike_pct",
    "is_calendar_price_change", "is_calendar_disruption", "is_supply_constraint_event",
    "is_orion_retool_event", "is_helios_expansion", "is_erp_migration_gap",
    "is_marketing_event", "is_mea_pre_entry",
    # Macroeconomic Absolute & Origin-Relative Elasticities
    "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index", "EV_Adoption_Index",
    "ISM_Manufacturing_PMI_lag1", "Auto_Production_YoY_Pct_lag1",
    "New_Vehicle_Orders_Index_lag1", "New_Vehicle_Orders_Index_lag3",
    "Industrial_Util_Rate_Pct_lag1",
    "ev_adoption_at_origin", "ev_adoption_ratio", "ev_adoption_delta",
    "ev_velocity_12m_origin", "ev_saturation_headroom", "ev_demand_multiplier",
    "pmi_delta_origin", "orders_ratio_origin", "raw_mat_ratio_origin", "auto_yoy_delta_origin",
    # Intermittency & Baselines
    "adi", "cv2", "croston_baseline_rate", "sba_baseline_rate",
    # Lifecycle & Catalog
    "months_since_launch", "months_until_eol", "is_pre_launch", "is_post_eol",
    "has_replacement_sku", "launch_ramp_gain",
    # EV Powertrain Interaction
    "is_powertrain_ev", "is_powertrain_ice", "ev_index_x_is_ev", "ev_index_x_is_ice",
    # Compatibility anchors
    "lag_12", "rolling_mean_12", "rolling_std_12",
]

CATEGORICAL_COLUMNS = [
    "Product_Family", "Region", "Business_Unit",
    "Customer_Name", "Sales_Channel", "intermittency_category", "Powertrain_Application"
]
NN_EMBEDDING_COLUMNS = ["SKU_ID"] + CATEGORICAL_COLUMNS

DRIVER_GROUPS = {
    "Baseline & Seasonal Level": {
        "origin_mean_3m", "origin_mean_6m", "origin_mean_12m", "origin_mean_24m",
        "origin_median_12m", "origin_std_12m", "origin_ewm", "origin_trend_ratio",
        "clean_mean_6m", "clean_mean_12m", "clean_mean_24m", "clean_ewm",
        "clean_median_12m", "clean_std_12m", "clean_cv",
        "origin_nonzero_rate", "recent_6m_nz_rate", "recent_12m_nz_rate", "recent_24m_nz_rate",
        "historical_stockout_rate", "same_month_hist_mean", "same_month_last_obs",
        "seasonal_index", "fam_reg_month_index", "analytical_anchor", "seasonal_ewm_anchor",
        "sku_global_prior", "family_region_prior", "customer_region_prior", "sku_price_log",
        "horizon_months", "horizon_log", "adi", "cv2", "croston_baseline_rate",
        "sba_baseline_rate", "lag_12", "rolling_mean_12", "rolling_std_12",
        "SKU_ID", "Product_Family", "Region", "Business_Unit", "intermittency_category",
    },
    "Macro Economy": {
        "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
        "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
        "Fuel_Price_Index", "ISM_Manufacturing_PMI_lag1", "Auto_Production_YoY_Pct_lag1",
        "New_Vehicle_Orders_Index_lag1", "New_Vehicle_Orders_Index_lag3",
        "Industrial_Util_Rate_Pct_lag1", "pmi_delta_origin", "orders_ratio_origin",
        "raw_mat_ratio_origin", "auto_yoy_delta_origin",
    },
    "EV Transition": {
        "EV_Adoption_Index", "ev_adoption_at_origin", "ev_adoption_ratio", "ev_adoption_delta",
        "ev_velocity_12m_origin", "ev_saturation_headroom",
        "ev_demand_multiplier", "is_powertrain_ev", "is_powertrain_ice",
        "ev_index_x_is_ev", "ev_index_x_is_ice", "Powertrain_Application",
    },
    "Calendar & Price Hikes": {
        "Month", "Quarter", "month_sin", "month_cos", "month_sin2", "month_cos2",
        "month_sin3", "month_cos3", "quarter_sin", "quarter_cos",
        "is_europe_summer_shutdown", "is_na_retool_month", "is_golden_week_month",
        "is_cny_month", "is_ramadan_month", "is_year_end_holiday",
        "is_pre_price_hike_month", "is_post_price_hike_month", "announced_price_hike_pct",
        "is_calendar_price_change",
    },
    "Customer & Scoped Events": {
        "Customer_Name", "Sales_Channel", "customer_cycle_months",
        "months_since_last_order", "months_since_last_nz_at_origin",
        "is_expected_cycle_hit", "ym_mod2_nz_rate", "mod2_phase_advantage",
        "ym_mod3_nz_rate", "mod3_phase_advantage", "same_month_nz_rate", "same_month_obs_cnt",
        "is_calendar_disruption", "is_supply_constraint_event",
        "is_orion_retool_event", "is_helios_expansion", "is_erp_migration_gap",
        "is_marketing_event", "is_mea_pre_entry",
    },
    "Catalog & Lifecycle": {
        "months_since_launch", "months_until_eol", "is_pre_launch",
        "is_post_eol", "has_replacement_sku", "predecessor_prior", "launch_ramp_gain",
    },
}


class TabularDemandResNet(nn.Module):
    """PyTorch Entity-Embedding + Gated Residual Network predicting log1p demand residual."""

    def __init__(self, num_features: int, cat_dims: List[Tuple[int, int]]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n_cls + 1, emb_dim) for n_cls, emb_dim in cat_dims])
        in_dim = num_features + sum(d for _, d in cat_dims)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.res_block = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 128),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, log_anchor: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        h = self.in_proj(torch.cat([x_num, emb], dim=1))
        h = h + self.res_block(h)
        delta = self.head(h).squeeze(-1)
        return log_anchor + delta


def _fit_predict_pytorch_resnet(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    log_anchor_tr: np.ndarray,
    X_eval_list: List[pd.DataFrame],
    log_anchor_eval_list: List[np.ndarray],
    num_cols: List[str],
    cat_cols: List[str],
    epochs: int = 6,
) -> List[np.ndarray]:
    """Trains PyTorch Entity-Embedding ResNet deterministically on CUDA or CPU."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    x_num_tr = np.nan_to_num(scaler.fit_transform(X_tr[num_cols].values), nan=0.0, posinf=6.0, neginf=-6.0)
    x_num_tr = np.clip(x_num_tr, -8.0, 8.0).astype(np.float32)

    cat_maps = {}
    cat_dims = []
    x_cat_tr = np.zeros((len(X_tr), len(cat_cols)), dtype=np.int64)
    for idx, c in enumerate(cat_cols):
        cats = list(X_tr[c].cat.categories) if hasattr(X_tr[c], "cat") else sorted(X_tr[c].astype(str).unique())
        cmap = {v: i + 1 for i, v in enumerate(cats)}
        cat_maps[c] = cmap
        n_cls = len(cats) + 1
        emb_dim = min(10, max(3, int(round(np.log2(max(n_cls, 2)) + 1))))
        cat_dims.append((n_cls, emb_dim))
        x_cat_tr[:, idx] = X_tr[c].astype(str).map(cmap).fillna(0).astype(np.int64).values

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_num_tr),
        torch.from_numpy(x_cat_tr),
        torch.from_numpy(np.log1p(y_tr).astype(np.float32)),
        torch.from_numpy(w_tr.astype(np.float32)),
        torch.from_numpy(log_anchor_tr.astype(np.float32)),
    )
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    loader = torch.utils.data.DataLoader(ds, batch_size=8192, shuffle=True, generator=gen)

    model = TabularDemandResNet(len(num_cols), cat_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(loader)))

    model.train()
    for _ in range(epochs):
        for bn, bc, by, bw, ba in loader:
            bn, bc, by, bw, ba = bn.to(device), bc.to(device), by.to(device), bw.to(device), ba.to(device)
            optimizer.zero_grad()
            pred_log = model(bn, bc, ba)
            loss_vec = nn.functional.smooth_l1_loss(pred_log, by, beta=0.25, reduction="none")
            # Mildly regularize neural residual delta toward analytical anchor to curb regime drift
            loss = (loss_vec * bw).mean() + 0.015 * ((pred_log - ba) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    preds_out = []
    with torch.no_grad():
        for X_ev, la_ev in zip(X_eval_list, log_anchor_eval_list):
            if len(X_ev) == 0:
                preds_out.append(np.array([], dtype=float))
                continue
            xn = np.clip(
                np.nan_to_num(scaler.transform(X_ev[num_cols].values), nan=0.0, posinf=6.0, neginf=-6.0),
                -8.0, 8.0,
            ).astype(np.float32)
            xc = np.zeros((len(X_ev), len(cat_cols)), dtype=np.int64)
            for idx, c in enumerate(cat_cols):
                xc[:, idx] = X_ev[c].astype(str).map(cat_maps[c]).fillna(0).astype(np.int64).values
            bs = 16384
            buf = []
            for start in range(0, len(X_ev), bs):
                end = min(start + bs, len(X_ev))
                p = model(
                    torch.from_numpy(xn[start:end]).to(device),
                    torch.from_numpy(xc[start:end]).to(device),
                    torch.from_numpy(la_ev[start:end].astype(np.float32)).to(device),
                )
                buf.append(np.clip(np.expm1(np.clip(p.cpu().numpy(), 0.0, 11.5)), 0.0, None))
            preds_out.append(np.concatenate(buf))
    return preds_out


def prepare_lgb_data(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    cat_cols: List[str] = None,
    category_levels: Dict[str, pd.Index] = None,
) -> pd.DataFrame:
    """Prepares and aligns numeric/categorical columns identically between train, val, and test."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    X = df[feature_cols + cat_cols].copy()
    for col in feature_cols:
        X[col] = np.nan_to_num(pd.to_numeric(X[col], errors="coerce").values, nan=0.0, posinf=0.0, neginf=0.0)

    for c in cat_cols:
        vals = X[c].astype(str).fillna("Unknown")
        if category_levels is not None and c in category_levels:
            X[c] = pd.Categorical(vals, categories=category_levels[c])
        else:
            X[c] = vals.astype("category")

    return X


def _compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Downweights structural anomaly regimes & early EV hypergrowth noise; emphasizes recent cycles."""
    w = np.ones(len(df), dtype=float)
    if "Date" in df.columns:
        dates = df["Date"].astype(str)
        w = np.where(dates.between("2020-03", "2020-05"), 0.25, w)
        w = np.where(dates >= "2024-01", 1.30, w)
        w = np.where(dates <= "2020-12", np.minimum(w, 0.70), w)
        if "Powertrain_Application" in df.columns:
            early_ev = (df["Powertrain_Application"].astype(str).values == "EV") & (dates.values <= "2022-12")
            w = np.where(early_ev, w * 0.45, w)
    if "is_erp_migration_gap" in df.columns:
        w = np.where(df["is_erp_migration_gap"].values == 1, 0.10, w)
    return w


def _optimize_simplex_blend(preds_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Finds diverse convex meta-weights minimizing validation WAPE + bias + prior L2 shrinkage."""
    k = preds_matrix.shape[1]
    denom = max(1.0, float(np.sum(y_true)))
    prior = np.array([0.08, 0.28, 0.28, 0.16, 0.12, 0.08], dtype=float)
    if k != len(prior):
        prior = np.full(k, 1.0 / k)

    def _loss(w):
        p = preds_matrix @ w
        wape = float(np.sum(np.abs(y_true - p)) / denom)
        bias = float(abs(np.sum(p - y_true)) / denom)
        div_reg = 0.018 * float(np.sum((w - prior) ** 2))
        return wape + 0.15 * bias + div_reg

    w0 = prior.copy()

    res = minimize(
        _loss,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 200, "ftol": 1e-6},
    )
    w_opt = np.clip(res.x, 0.0, 1.0) if res.success else w0
    w_opt = w_opt / max(1e-9, float(np.sum(w_opt)))
    return w_opt


def _tune_zero_gate(prob_nz: np.ndarray, base_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Tunes conservative out-of-sample Hurdle cutoff tau when it strictly reduces WAPE."""
    denom = max(1.0, float(np.sum(y_true)))
    best_tau = 0.0
    best_wape = float(np.sum(np.abs(y_true - base_pred)) / denom)
    base_wape = best_wape
    for tau in np.arange(0.10, 0.44, 0.02):
        gated = np.where(prob_nz < tau, 0.0, base_pred)
        w = float(np.sum(np.abs(y_true - gated)) / denom)
        if w < best_wape - 0.0005 and w < base_wape - 0.0005:
            best_wape = w
            best_tau = float(round(tau, 2))
    return best_tau


def _fit_all_learners(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    log_anchor_tr: np.ndarray,
    X_eval: pd.DataFrame,
    log_anchor_eval: np.ndarray,
    anchor_eval: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    max_trees: int,
    nn_epochs: int,
    X_tr_nn: pd.DataFrame = None,
    X_eval_nn: pd.DataFrame = None,
    X_es_val: pd.DataFrame = None,
    y_es_val: np.ndarray = None,
    w_es_val: np.ndarray = None,
    log_anchor_es: np.ndarray = None,
    fixed_trees: Dict[str, int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Fits all 5 ML/DL learners consistently (with optional early stopping and full refit)."""
    use_es = X_es_val is not None and len(X_es_val) >= 50
    es_cb = [lgb.early_stopping(25, verbose=False)] if use_es else None
    fixed_trees = fixed_trees or {}

    base_params = {
        "boosting_type": "gbdt",
        "learning_rate": 0.055,
        "num_leaves": 47,
        "max_depth": 7,
        "max_bin": 127,
        "min_child_samples": 30,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 4.0,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    # 1. LightGBM Tweedie (Volume-Unbiased Level)
    nt_tw = fixed_trees.get("tw", max_trees)
    m_tw = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.25, n_estimators=nt_tw, **base_params)
    if use_es:
        m_tw.fit(X_tr, y_tr, sample_weight=w_tr, categorical_feature=cat_cols, eval_set=[(X_es_val, y_es_val)], eval_sample_weight=[w_es_val], callbacks=es_cb)
        best_tw = int(m_tw.best_iteration_ or max_trees)
    else:
        m_tw.fit(X_tr, y_tr, sample_weight=w_tr, categorical_feature=cat_cols)
        best_tw = nt_tw
    pred_tw = np.clip(m_tw.predict(X_eval), 0.0, None)

    # 2. LightGBM Multiplicative Log-Ratio Residual (scale-invariant extrapolator)
    res_target_tr = np.log1p(y_tr) - log_anchor_tr
    nt_lr = fixed_trees.get("lr", max_trees)
    m_lr = lgb.LGBMRegressor(objective="huber", alpha=0.5, n_estimators=nt_lr, **base_params)
    if use_es:
        res_target_es = np.log1p(y_es_val) - log_anchor_es
        m_lr.fit(X_tr, res_target_tr, sample_weight=w_tr, categorical_feature=cat_cols, eval_set=[(X_es_val, res_target_es)], eval_sample_weight=[w_es_val], callbacks=es_cb)
        best_lr = int(m_lr.best_iteration_ or max_trees)
    else:
        m_lr.fit(X_tr, res_target_tr, sample_weight=w_tr, categorical_feature=cat_cols)
        best_lr = nt_lr
    pred_lr = np.clip(np.expm1(np.clip(log_anchor_eval + m_lr.predict(X_eval), 0.0, 11.5)), 0.0, None)

    # 3. Two-Stage Zero-Inflated Hurdle (Classifier P(Y>0) + Positive-Only Tweedie E[Y|Y>0])
    y_bin = (y_tr > 0).astype(int)
    nt_cls = fixed_trees.get("cls", min(350, max_trees))
    m_cls = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=47, max_depth=7, min_child_samples=25,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=3.0,
        n_estimators=nt_cls, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
    )
    if use_es:
        m_cls.fit(X_tr, y_bin, sample_weight=w_tr, categorical_feature=cat_cols, eval_set=[(X_es_val, (y_es_val > 0).astype(int))], eval_sample_weight=[w_es_val], callbacks=es_cb)
        best_cls = int(m_cls.best_iteration_ or nt_cls)
    else:
        m_cls.fit(X_tr, y_bin, sample_weight=w_tr, categorical_feature=cat_cols)
        best_cls = nt_cls
    prob_nz = m_cls.predict_proba(X_eval)[:, 1]

    pos_mask = y_tr > 0
    nt_pos = fixed_trees.get("pos", max_trees)
    m_pos = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.20, n_estimators=nt_pos, **base_params)
    if use_es and (y_es_val > 0).sum() >= 50:
        es_pos = y_es_val > 0
        m_pos.fit(
            X_tr.iloc[pos_mask], y_tr[pos_mask], sample_weight=w_tr[pos_mask], categorical_feature=cat_cols,
            eval_set=[(X_es_val.iloc[es_pos], y_es_val[es_pos])], eval_sample_weight=[w_es_val[es_pos]], callbacks=es_cb,
        )
        best_pos = int(m_pos.best_iteration_ or nt_pos)
    else:
        m_pos.fit(X_tr.iloc[pos_mask], y_tr[pos_mask], sample_weight=w_tr[pos_mask], categorical_feature=cat_cols)
        best_pos = nt_pos
    mu_pos = np.clip(m_pos.predict(X_eval), 0.0, None)
    pred_hurdle = prob_nz * mu_pos

    # 4. Scikit-Learn HistGradientBoostingRegressor (Poisson Deviance)
    hgb_cols = [c for c in X_tr.columns if c != "SKU_ID"]
    m_hgb = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.06, max_iter=180, max_leaf_nodes=47,
        min_samples_leaf=30, l2_regularization=3.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.10, random_state=RANDOM_SEED,
    )
    m_hgb.fit(X_tr[hgb_cols], np.clip(y_tr, 0.0, None), sample_weight=w_tr)
    pred_hgb = np.clip(m_hgb.predict(X_eval[hgb_cols]), 0.0, None)

    # 5. PyTorch Deep Tabular Entity-Embedding ResNet (includes SKU_ID embedding via nn_X dataframes)
    nn_tr_df = X_tr_nn if X_tr_nn is not None else X_tr
    nn_ev_df = X_eval_nn if X_eval_nn is not None else X_eval
    nn_cat_cols = [c for c in NN_EMBEDDING_COLUMNS if c in nn_tr_df.columns]
    pred_nn = _fit_predict_pytorch_resnet(
        nn_tr_df, y_tr, w_tr, log_anchor_tr,
        [nn_ev_df], [log_anchor_eval],
        num_cols=num_cols, cat_cols=nn_cat_cols, epochs=nn_epochs,
    )[0]

    preds = {
        "tw": pred_tw,
        "lr": pred_lr,
        "hurdle": pred_hurdle,
        "prob_nz": prob_nz,
        "hgb": pred_hgb,
        "pytorch": pred_nn,
        "anchor": anchor_eval,
    }
    artifacts = {
        "model_tweedie": m_tw,
        "best_iters": {"tw": best_tw, "lr": best_lr, "cls": best_cls, "pos": best_pos},
    }
    return preds, artifacts


def train_and_forecast_lgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Units_Sold_Unconstrained",
    n_estimators: int = 420,
    compute_shap: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains the Multi-Paradigm Ensemble (Hurdle + LightGBM Tweedie + LogRatio + HistGB + PyTorch ResNet + Anchor),
    solves out-of-sample simplex meta-weights + zero-gate threshold, calibrates CQR P10/P90,
    and optionally exports Tree-SHAP driver attributions.
    """
    # Fit LightGBM categorical_feature using low-cardinality CATEGORICAL_COLUMNS, while passing SKU_ID through X_train for PyTorch embeddings
    t_start = time.time()
    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns and c in test_df.columns]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in train_df.columns and c in test_df.columns]
    nn_all_cats = [c for c in NN_EMBEDDING_COLUMNS if c in train_df.columns and c in test_df.columns]

    X_train_full = prepare_lgb_data(train_df, feature_cols=feature_cols, cat_cols=nn_all_cats)
    cat_levels = {c: X_train_full[c].cat.categories for c in nn_all_cats}
    X_test_full = prepare_lgb_data(test_df, feature_cols=feature_cols, cat_cols=nn_all_cats, category_levels=cat_levels)

    # Separate clean LightGBM matrix (without 601-level SKU_ID) + PyTorch entity-embedding view
    tree_cols = feature_cols + cat_cols
    X_train = X_train_full[tree_cols]
    X_test = X_test_full[tree_cols]

    y_train = np.nan_to_num(pd.to_numeric(train_df[target_col], errors="coerce").values, nan=0.0).clip(min=0.0)
    weights = _compute_sample_weights(train_df)

    anchor_col = "analytical_anchor" if "analytical_anchor" in train_df.columns else "seasonal_ewm_anchor"
    anchor_train = (
        np.nan_to_num(pd.to_numeric(train_df[anchor_col], errors="coerce").values, nan=0.0).clip(min=0.0)
        if anchor_col in train_df.columns else np.zeros(len(train_df), dtype=float)
    )
    anchor_test = (
        np.nan_to_num(pd.to_numeric(test_df[anchor_col], errors="coerce").values, nan=0.0).clip(min=0.0)
        if anchor_col in test_df.columns else np.zeros(len(test_df), dtype=float)
    )
    log_anchor_train = np.log1p(anchor_train)
    log_anchor_test = np.log1p(anchor_test)

    # Strict Zero-Overlap Temporal Validation Split:
    # Choose the latest full-horizon origin (second-to-last if >=3 origins, since -1 is 12m and -2 is 19m)
    # and restrict training rows strictly to target Date <= val_origin so ZERO target dates overlap!
    val_mask = np.zeros(len(train_df), dtype=bool)
    tr_mask = np.ones(len(train_df), dtype=bool)
    if "_origin" in train_df.columns and train_df["_origin"].nunique() >= 3 and "Date" in train_df.columns:
        sorted_origs = sorted(train_df["_origin"].unique())
        # Pick latest origin that has >= 15 months of future evaluation
        orig_spans = train_df.groupby("_origin")["horizon_months"].max() if "horizon_months" in train_df.columns else pd.Series(12, index=sorted_origs)
        full_origs = [o for o in sorted_origs if orig_spans.get(o, 0) >= 15]
        val_orig = full_origs[-1] if full_origs else sorted_origs[-1]
        val_mask = (train_df["_origin"] == val_orig).values
        tr_mask = (train_df["Date"].astype(str) <= str(val_orig)).values & (~val_mask)

    if tr_mask.sum() < 500 or val_mask.sum() < 200:
        cutoff = max(1, int(len(train_df) * 0.82))
        tr_mask = np.zeros(len(train_df), dtype=bool)
        tr_mask[:cutoff] = True
        val_mask = ~tr_mask

    nn_epochs_val = 4 if not compute_shap else 5
    nn_epochs_full = 4 if not compute_shap else 6

    # --- 1. Out-of-Sample Temporal Validation Fit for Meta-Weights, Hurdle Gate & CQR ---
    val_preds, val_art = _fit_all_learners(
        X_train.iloc[tr_mask], y_train[tr_mask], weights[tr_mask], log_anchor_train[tr_mask],
        X_train.iloc[val_mask], log_anchor_train[val_mask], anchor_train[val_mask],
        cat_cols=cat_cols, num_cols=feature_cols,
        max_trees=n_estimators, nn_epochs=nn_epochs_val,
        X_tr_nn=X_train_full.iloc[tr_mask], X_eval_nn=X_train_full.iloc[val_mask],
        X_es_val=X_train.iloc[val_mask], y_es_val=y_train[val_mask], w_es_val=weights[val_mask], log_anchor_es=log_anchor_train[val_mask],
    )

    model_keys = ["tw", "lr", "hurdle", "hgb", "pytorch", "anchor"]
    val_mat = np.column_stack([val_preds[k] for k in model_keys])
    y_val = y_train[val_mask]
    w_opt = _optimize_simplex_blend(val_mat, y_val)
    val_blend = np.clip(val_mat @ w_opt, 0.0, None)
    best_tau = _tune_zero_gate(val_preds["prob_nz"], val_blend, y_val)

    # Validation Quantile Models for Horizon- & Scale-Normalized CQR (target 80.5% coverage)
    q_params = {
        "boosting_type": "gbdt", "learning_rate": 0.06, "num_leaves": 31, "max_depth": 6,
        "max_bin": 127, "min_child_samples": 30, "subsample": 0.85, "subsample_freq": 1,
        "colsample_bytree": 0.80, "reg_alpha": 0.5, "reg_lambda": 3.0,
        "n_estimators": max(100, int(n_estimators * 0.45)),
        "random_state": RANDOM_SEED, "verbose": -1, "n_jobs": -1,
    }
    m_p10_val = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **q_params)
    m_p10_val.fit(X_train.iloc[tr_mask], y_train[tr_mask], sample_weight=weights[tr_mask], categorical_feature=cat_cols)
    v_p10 = np.clip(m_p10_val.predict(X_train.iloc[val_mask]), 0.0, None)

    m_p90_val = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **q_params)
    m_p90_val.fit(X_train.iloc[tr_mask], y_train[tr_mask], sample_weight=weights[tr_mask], categorical_feature=cat_cols)
    v_p90 = np.clip(m_p90_val.predict(X_train.iloc[val_mask]), 0.0, None)

    # Compute Signed Multiplicative CQR Scale Factor targeting exact ~80.5% empirical coverage per horizon band
    v_p10_b = np.minimum(v_p10, val_blend)
    v_p90_b = np.maximum(v_p90, val_blend)
    half_lo = np.maximum(val_blend - v_p10_b, 1.0)
    half_hi = np.maximum(v_p90_b - val_blend, 1.0)
    norm_score = np.maximum((val_blend - y_val) / half_lo, (y_val - val_blend) / half_hi)

    global_scale = float(np.clip(np.quantile(norm_score, 0.745), 0.55, 1.25))
    cqr_scales = {"short": global_scale, "mid": global_scale, "long": global_scale}
    if "horizon_months" in train_df.columns:
        vh = train_df.loc[val_mask, "horizon_months"].values
        for band, mask_b in [("short", vh <= 6), ("mid", (vh >= 7) & (vh <= 12)), ("long", vh >= 13)]:
            if mask_b.sum() >= 100:
                cqr_scales[band] = float(np.clip(np.quantile(norm_score[mask_b], 0.745), 0.55, 1.25))

    # --- 2. Full-History Refit Across All 5 ML/DL Learners ---
    refit_trees = {k: max(60, int(round(v * 1.12))) for k, v in val_art["best_iters"].items()}
    test_preds, full_art = _fit_all_learners(
        X_train, y_train, weights, log_anchor_train,
        X_test, log_anchor_test, anchor_test,
        cat_cols=cat_cols, num_cols=feature_cols,
        max_trees=n_estimators, nn_epochs=nn_epochs_full,
        X_tr_nn=X_train_full, X_eval_nn=X_test_full,
        fixed_trees=refit_trees,
    )

    test_mat = np.column_stack([test_preds[k] for k in model_keys])
    pred_p50_raw = np.clip(test_mat @ w_opt, 0.0, None)
    pred_p50 = np.where(test_preds["prob_nz"] < best_tau, 0.0, pred_p50_raw) if best_tau > 0.0 else pred_p50_raw

    model_p10 = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **q_params)
    model_p10.fit(X_train, y_train, sample_weight=weights, categorical_feature=cat_cols)
    raw_p10 = np.clip(model_p10.predict(X_test), 0.0, None)

    model_p90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **q_params)
    model_p90.fit(X_train, y_train, sample_weight=weights, categorical_feature=cat_cols)
    raw_p90 = np.clip(model_p90.predict(X_test), 0.0, None)

    raw_p10_b = np.minimum(raw_p10, pred_p50)
    raw_p90_b = np.maximum(raw_p90, pred_p50)
    if "horizon_months" in test_df.columns:
        th = test_df["horizon_months"].values
        scale_arr = np.where(
            th <= 6, cqr_scales["short"],
            np.where(th <= 12, cqr_scales["mid"], cqr_scales["long"]),
        )
    else:
        scale_arr = np.full(len(test_df), global_scale)

    t_half_lo = np.maximum(pred_p50 - raw_p10_b, 0.0)
    t_half_hi = np.maximum(raw_p90_b - pred_p50, 0.0)
    cal_p10 = np.clip(pred_p50 - scale_arr * t_half_lo, 0.0, None)
    cal_p90 = np.clip(pred_p50 + scale_arr * t_half_hi, 0.0, None)

    pred_p10_enforced = np.minimum(cal_p10, pred_p50)
    pred_p90_enforced = np.maximum(cal_p90, pred_p50)

    # Extract Feature Importances from LightGBM Tweedie backbone
    model_tweedie = full_art["model_tweedie"]
    importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model_tweedie.feature_importances_,
    }).sort_values("Importance", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importances.to_csv(REPORTS_DIR / "feature_importances.csv", index=False)

    # 3. Native Tree-SHAP Driver Attribution (pred_contrib=True scaled cleanly to final Ensemble P50)
    if compute_shap and len(X_test) > 0:
        contrib = model_tweedie.predict(X_test, pred_contrib=True)
        feat_names = list(X_train.columns)
        raw_shap = contrib[:, :-1]
        abs_sum = np.maximum(np.sum(np.abs(raw_shap), axis=1, keepdims=True), 1e-6)
        unit_scale = pred_p50[:, None] / abs_sum

        driver_df = pd.DataFrame({
            "Date": test_df["Date"].values,
            "SKU_ID": test_df["SKU_ID"].values,
            "Region": test_df["Region"].values,
            "Product_Family": test_df["Product_Family"].values if "Product_Family" in test_df.columns else "",
            "P50_Forecast": np.round(pred_p50, 2),
        })
        for grp_name, grp_cols in DRIVER_GROUPS.items():
            idxs = [i for i, f in enumerate(feat_names) if f in grp_cols]
            if idxs:
                signed_share = np.sum(raw_shap[:, idxs], axis=1)
                driver_df[grp_name] = np.round(signed_share * unit_scale.ravel(), 2)
            else:
                driver_df[grp_name] = 0.0

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        driver_df.to_parquet(OUTPUTS_DIR / "forecast_drivers.parquet", index=False)

    out_preds = pd.DataFrame({
        "P10": pred_p10_enforced.round(),
        "P50": pred_p50.round(),
        "P90": pred_p90_enforced.round(),
        "Raw_P10": raw_p10_b.round(),
        "Raw_P90": raw_p90_b.round(),
        "Pred_Tweedie": test_preds["tw"].round(),
        "Pred_LogRatio": test_preds["lr"].round(),
        "Pred_Log1p": test_preds["lr"].round(),
        "Pred_Hurdle": (np.where(test_preds["prob_nz"] < best_tau, 0.0, test_preds["hurdle"]) if best_tau > 0 else test_preds["hurdle"]).round(),
        "Pred_HistGB": test_preds["hgb"].round(),
        "Pred_PyTorch": test_preds["pytorch"].round(),
        "Pred_Anchor": test_preds["anchor"].round(),
        "Prob_NonZero": np.round(test_preds["prob_nz"], 4),
    }, index=test_df.index)

    weights_dict = {k: round(float(w), 3) for k, w in zip(model_keys, w_opt)}
    print(
        f"    [Ensemble Fit {time.time() - t_start:.1f}s] weights={weights_dict}, hurdle_tau={best_tau}",
        flush=True,
    )
    metadata = {
        "feature_count": len(X_train.columns),
        "ensemble_weights": {
            "tweedie": weights_dict["tw"],
            "log1p_l1": weights_dict["lr"],
            "hurdle": weights_dict["hurdle"],
            "histgb": weights_dict["hgb"],
            "pytorch_resnet": weights_dict["pytorch"],
            "seasonal_anchor": weights_dict["anchor"],
        },
        "hurdle_zero_gate": best_tau,
        "cqr_scales": {k: round(v, 3) for k, v in cqr_scales.items()},
        "top_features": importances.head(12).to_dict(orient="records"),
        "p50_mean": float(pred_p50.mean()),
        "p10_mean": float(pred_p10_enforced.mean()),
        "p90_mean": float(pred_p90_enforced.mean()),
    }

    return out_preds, metadata
