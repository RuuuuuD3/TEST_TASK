import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACTS,
    BLEND_PAIRS,
    FIGURES_ROOT,
    HGB_PARAMS,
    PROCESSED_DIR,
    RF_PARAMS,
    RANDOM_STATE,
    RIDGE_ALPHA,
    TRAIN_FRAC,
    VAL_FRAC,
    WF_MAX_FOLDS,
    WF_MIN_FOLDS,
    WF_MIN_TRAIN_FRAC,
    WF_VAL_FRAC,
)
from .features import build_feature_table, feature_target_split
from .plotting import plot_column_splits
from .preprocess import (
    clip_bounds_from_train,
    clip_series,
    global_minute_index,
    load_raw_sorted,
    minute_series_for_column,
    save_minute_csv,
)


def _lowest_mae_key(maes: Dict[str, float], order_if_same: Dict[str, int]) -> str:
    best_k = None
    for k in maes:
        if best_k is None:
            best_k = k
            continue
        m = maes[k]
        mb = maes[best_k]
        if m < mb:
            best_k = k
        elif m == mb and order_if_same[k] < order_if_same[best_k]:
            best_k = k
    if best_k is None:
        raise RuntimeError("empty maes")
    return best_k


def _split_train_val_test(n: int) -> Tuple[slice, slice, slice]:
    n_tr = int(np.floor(TRAIN_FRAC * n))
    n_va = int(np.floor((TRAIN_FRAC + VAL_FRAC) * n))
    n_tr = max(1, min(n - 2, n_tr))
    n_va = max(n_tr + 1, min(n - 1, n_va))
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)


def _lag1_baseline_mae(y_va: np.ndarray, X_va: pd.DataFrame) -> float:
    if "lag_1" not in X_va.columns:
        return float("nan")
    pred = X_va["lag_1"].to_numpy(dtype=float)
    m = np.isfinite(pred) & np.isfinite(y_va)
    if not np.any(m):
        return float("nan")
    return float(mean_absolute_error(y_va[m], pred[m]))


def _three_model_preds(pred_r_tr, pred_r_va, pred_r_te, pred_h_tr, pred_h_va, pred_h_te, pred_rf_tr, pred_rf_va, pred_rf_te):
    return {
        "ridge": (pred_r_tr, pred_r_va, pred_r_te),
        "hgb": (pred_h_tr, pred_h_va, pred_h_te),
        "rf": (pred_rf_tr, pred_rf_va, pred_rf_te),
    }


def _fit_three_val_preds(
    X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
        ]
    )
    ridge.fit(X_tr, y_tr)
    pr_va = ridge.predict(X_va)

    hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
    hgb.fit(X_tr, y_tr)
    ph_va = hgb.predict(X_va)

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    prf_va = rf.predict(X_va)
    return pr_va, ph_va, prf_va


def _best_two_model_blend(y_va: np.ndarray, pack) -> Tuple[str, str, float, float]:
    best_mae = 1e18
    best_a, best_b = "ridge", "hgb"
    best_w = 0.5
    for a, b in BLEND_PAIRS:
        pa, pva, _ = pack[a]
        _, pvb, _ = pack[b]
        for w in np.linspace(0.0, 1.0, 41):
            pred = w * pva + (1.0 - w) * pvb
            m = float(mean_absolute_error(y_va, pred))
            if m < best_mae:
                best_mae = m
                best_a, best_b = a, b
                best_w = float(w)
    return best_a, best_b, best_w, best_mae


def _mean_val_mae_many_train_sizes(
    X_dev: pd.DataFrame, y_dev: pd.Series
) -> Optional[Tuple[Dict[str, float], int]]:
    n_d = len(X_dev)
    val_len = max(400, int(WF_VAL_FRAC * n_d))
    min_train = max(2000, int(WF_MIN_TRAIN_FRAC * n_d))
    last_start = n_d - val_len
    if last_start <= min_train or val_len < 50:
        return None
    starts = np.linspace(min_train, last_start, num=WF_MAX_FOLDS, dtype=int)
    starts = np.unique(starts)
    if len(starts) < WF_MIN_FOLDS:
        return None

    sums = {"ridge": 0.0, "hgb": 0.0, "rf": 0.0, "blend": 0.0}
    n_used = 0
    for s in starts:
        X_trf = X_dev.iloc[: int(s)]
        y_trf = y_dev.iloc[: int(s)]
        end = int(s) + val_len
        X_vaf = X_dev.iloc[int(s) : end]
        y_vaf = y_dev.iloc[int(s) : end]
        if len(X_trf) < 100 or len(X_vaf) < 50:
            continue
        pr_v, ph_v, prf_v = _fit_three_val_preds(X_trf, y_trf, X_vaf)
        yv = y_vaf.to_numpy()
        pack_fold = {
            "ridge": (np.array([]), pr_v, np.array([])),
            "hgb": (np.array([]), ph_v, np.array([])),
            "rf": (np.array([]), prf_v, np.array([])),
        }
        _, _, _, mae_blend = _best_two_model_blend(yv, pack_fold)
        sums["ridge"] += float(mean_absolute_error(yv, pr_v))
        sums["hgb"] += float(mean_absolute_error(yv, ph_v))
        sums["rf"] += float(mean_absolute_error(yv, prf_v))
        sums["blend"] += mae_blend
        n_used += 1
    if n_used < WF_MIN_FOLDS:
        return None
    means: Dict[str, float] = {}
    for k in sums:
        means[k] = sums[k] / float(n_used)
    return means, n_used


def train_one_column(
    col: int,
    horizon_minutes: int = 0,
    artifacts_base: Optional[Path] = None,
) -> dict:
    if horizon_minutes != 0 and artifacts_base is None:
        raise ValueError("need artifacts_base if horizon_minutes != 0")
    base = artifacts_base if artifacts_base is not None else ARTIFACTS
    figures_dir = FIGURES_ROOT if artifacts_base is None else (base / "figures")

    raw = load_raw_sorted()
    all_minutes = global_minute_index(raw)
    y_i = minute_series_for_column(raw, col, all_minutes)
    lo, hi = clip_bounds_from_train(y_i, all_minutes)
    y_w = clip_series(y_i, lo, hi)

    save_minute_csv(PROCESSED_DIR / f"col_{col}_minute.csv", y_i, y_w)

    d_all = build_feature_table(y_w, horizon_minutes=horizon_minutes)
    X, y = feature_target_split(d_all)
    n = len(X)
    if n < 500:
        raise RuntimeError(f"col {col}: too few rows ({n})")

    sl_tr, sl_va, sl_te = _split_train_val_test(n)
    X_tr, y_tr = X.iloc[sl_tr], y.iloc[sl_tr]
    X_va, y_va = X.iloc[sl_va], y.iloc[sl_va]
    X_te, y_te = X.iloc[sl_te], y.iloc[sl_te]

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
        ]
    )
    ridge.fit(X_tr, y_tr)
    pred_r_tr = ridge.predict(X_tr)
    pred_r_va = ridge.predict(X_va)
    pred_r_te = ridge.predict(X_te)

    hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
    hgb.fit(X_tr, y_tr)
    pred_h_tr = hgb.predict(X_tr)
    pred_h_va = hgb.predict(X_va)
    pred_h_te = hgb.predict(X_te)

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    pred_rf_tr = rf.predict(X_tr)
    pred_rf_va = rf.predict(X_va)
    pred_rf_te = rf.predict(X_te)

    pack = _three_model_preds(pred_r_tr, pred_r_va, pred_r_te, pred_h_tr, pred_h_va, pred_h_te, pred_rf_tr, pred_rf_va, pred_rf_te)

    mae_r_va = float(mean_absolute_error(y_va, pred_r_va))
    mae_h_va = float(mean_absolute_error(y_va, pred_h_va))
    mae_rf_va = float(mean_absolute_error(y_va, pred_rf_va))
    mae_naive_va = _lag1_baseline_mae(y_va.to_numpy(), X_va)

    blend_a, blend_b, blend_w, _ = _best_two_model_blend(y_va.to_numpy(), pack)
    tra, vaa, tea = pack[blend_a]
    trb, vab, teb = pack[blend_b]
    pred_b_tr = blend_w * tra + (1.0 - blend_w) * trb
    pred_b_va = blend_w * vaa + (1.0 - blend_w) * vab
    pred_b_te = blend_w * tea + (1.0 - blend_w) * teb

    mae_b_te = float(mean_absolute_error(y_te, pred_b_te))

    val_mae_blend = float(mean_absolute_error(y_va, pred_b_va))
    val_maes = {"ridge": mae_r_va, "hgb": mae_h_va, "rf": mae_rf_va, "blend": val_mae_blend}
    order_if_same = {"ridge": 0, "hgb": 1, "rf": 2, "blend": 3}
    sv_winner = _lowest_mae_key(val_maes, order_if_same)

    n_dev = sl_te.start
    X_dev, y_dev = X.iloc[:n_dev], y.iloc[:n_dev]
    walk_means_pack = _mean_val_mae_many_train_sizes(X_dev, y_dev)
    if walk_means_pack is not None:
        wf_means, wf_n_folds = walk_means_pack
        maes_for_final_pick = wf_means
    else:
        wf_means, wf_n_folds = None, 0
        maes_for_final_pick = val_maes

    winner = _lowest_mae_key(maes_for_final_pick, order_if_same)
    if winner == "blend":
        pred_w_tr, pred_w_va, pred_w_te = pred_b_tr, pred_b_va, pred_b_te
    else:
        pred_w_tr, pred_w_va, pred_w_te = pack[winner]
    wf_disagrees_sv = bool(wf_means is not None and sv_winner != winner)

    mae_w_te = float(mean_absolute_error(y_te, pred_w_te))
    test_mae_ridge = float(mean_absolute_error(y_te, pred_r_te))
    test_mae_hgb = float(mean_absolute_error(y_te, pred_h_te))
    test_mae_rf = float(mean_absolute_error(y_te, pred_rf_te))
    test_mae_by_pick = {
        "ridge": test_mae_ridge,
        "hgb": test_mae_hgb,
        "rf": test_mae_rf,
        "blend": mae_b_te,
    }
    test_mae_at_sv_winner = test_mae_by_pick[sv_winner]

    train_mae_ridge = float(mean_absolute_error(y_tr, pred_r_tr))
    train_mae_hgb = float(mean_absolute_error(y_tr, pred_h_tr))
    train_mae_rf = float(mean_absolute_error(y_tr, pred_rf_tr))
    train_mae_blend = float(mean_absolute_error(y_tr, pred_b_tr))
    train_mae_winner = float(mean_absolute_error(y_tr, pred_w_tr))
    val_mae_winner = val_maes[winner]

    out_dir = base / f"col_{col}"
    base.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_tr, n_va = sl_tr.stop, sl_va.stop
    split_lab = np.array(["train"] * n, dtype=object)
    split_lab[n_tr:n_va] = "val"
    split_lab[n_va:n] = "test"

    pred_tbl = pd.DataFrame(
        {
            "time": X.index,
            "split": split_lab,
            "y_true": y.to_numpy(),
            "pred_ridge": np.concatenate([pred_r_tr, pred_r_va, pred_r_te]),
            "pred_hgb": np.concatenate([pred_h_tr, pred_h_va, pred_h_te]),
            "pred_blend": np.concatenate([pred_b_tr, pred_b_va, pred_b_te]),
            "pred_winner": np.concatenate([pred_w_tr, pred_w_va, pred_w_te]),
            "pred_rf": np.concatenate([pred_rf_tr, pred_rf_va, pred_rf_te]),
        }
    )
    pred_tbl.to_csv(out_dir / "predictions.csv", index=False)

    if winner == "blend":
        winner_label = "blend:" + str(blend_a) + "+" + str(blend_b) + " w=" + format(blend_w, ".2f")
    else:
        winner_label = str(winner)

    metrics = {
        "col": col,
        "horizon_min": horizon_minutes,
        "n_rows": n,
        "winsor_lo": lo,
        "winsor_hi": hi,
        "winner_label": winner_label,
        "train_mae_ridge": train_mae_ridge,
        "train_mae_hgb": train_mae_hgb,
        "train_mae_rf": train_mae_rf,
        "train_mae_blend": train_mae_blend,
        "train_mae_winner": train_mae_winner,
        "val_mae_naive_lag1": mae_naive_va,
        "val_mae_ridge": mae_r_va,
        "val_mae_hgb": mae_h_va,
        "val_mae_rf": mae_rf_va,
        "val_mae_blend": val_mae_blend,
        "val_mae_winner": val_mae_winner,
        "sv_winner": sv_winner,
        "wf_disagrees_sv": wf_disagrees_sv,
        "blend_a": blend_a,
        "blend_b": blend_b,
        "blend_w_on_a": blend_w,
        "test_mae_ridge": test_mae_ridge,
        "test_mae_hgb": test_mae_hgb,
        "test_mae_rf": test_mae_rf,
        "test_mae_blend": mae_b_te,
        "test_mae_winner": mae_w_te,
        "test_mae_at_sv_winner": test_mae_at_sv_winner,
        "test_mae_at_wf_winner": mae_w_te,
        "winner": winner,
        "wf_used": wf_means is not None,
        "wf_n_folds": wf_n_folds,
        "wf_mean_val_mae_ridge": wf_means["ridge"] if wf_means else None,
        "wf_mean_val_mae_hgb": wf_means["hgb"] if wf_means else None,
        "wf_mean_val_mae_rf": wf_means["rf"] if wf_means else None,
        "wf_mean_val_mae_blend": wf_means["blend"] if wf_means else None,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump({"model": ridge, "kind": "sklearn_pipeline_ridge"}, out_dir / "model_ridge.joblib")
    joblib.dump({"model": hgb, "kind": "hist_gbrt"}, out_dir / "model_hgb.joblib")
    joblib.dump({"model": rf, "kind": "random_forest"}, out_dir / "model_rf.joblib")
    joblib.dump(
        {
            "blend_a": blend_a,
            "blend_b": blend_b,
            "blend_w_on_a": blend_w,
            "winner": winner,
            "sv_winner": sv_winner,
            "wf_used": wf_means is not None,
            "wf_n_folds": wf_n_folds,
            "wf_disagrees_sv": wf_disagrees_sv,
            "winsor_lo": lo,
            "winsor_hi": hi,
            "horizon_minutes": horizon_minutes,
            "feature_columns": list(X.columns),
        },
        out_dir / "strategy_meta.joblib",
    )

    plot_column_splits(figures_dir, pred_tbl, col, blend_a, blend_b, blend_w, winner)

    return metrics


def train_all(columns: Union[Tuple[int, ...], List[int]]) -> pd.DataFrame:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in columns:
        m = train_one_column(int(col))
        rows.append(m)
    return pd.DataFrame(rows)
