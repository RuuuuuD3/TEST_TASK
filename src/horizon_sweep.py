import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACTS,
    FORECAST_HORIZONS_MINUTES,
    HGB_PARAMS,
    RIDGE_ALPHA,
    RANDOM_STATE,
    TARGET_COLS,
    TRAIN_FRAC,
    VAL_FRAC,
)
from .features import build_feature_table, feature_target_split
from .preprocess import (
    clip_bounds_from_train,
    clip_series,
    global_minute_index,
    load_raw_sorted,
    minute_series_for_column,
)


def _split_train_val_test(n: int) -> Tuple[slice, slice, slice]:
    n_tr = int(np.floor(TRAIN_FRAC * n))
    n_va = int(np.floor((TRAIN_FRAC + VAL_FRAC) * n))
    n_tr = max(1, min(n - 2, n_tr))
    n_va = max(n_tr + 1, min(n - 1, n_va))
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
        ]
    )


def run_column_horizons(col: int, horizons: Tuple[int, ...], with_hgb: bool) -> List[Dict]:
    raw = load_raw_sorted()
    all_minutes = global_minute_index(raw)
    y_i = minute_series_for_column(raw, col, all_minutes)
    lo, hi = clip_bounds_from_train(y_i, all_minutes)
    y_w = clip_series(y_i, lo, hi)

    rows: List[Dict] = []
    for h in horizons:
        d_all = build_feature_table(y_w, horizon_minutes=h)
        X, y = feature_target_split(d_all)
        n = len(X)
        if n < 800:
            rec_fail: Dict = {
                "col": col,
                "horizon_min": h,
                "n_rows": n,
                "test_mae_ridge": float("nan"),
                "test_mae_last_value": float("nan"),
                "note": "too_few_rows",
            }
            if with_hgb:
                rec_fail["test_mae_hgb"] = float("nan")
            rows.append(rec_fail)
            continue

        sl_tr, sl_va, sl_te = _split_train_val_test(n)
        X_tr, y_tr = X.iloc[sl_tr], y.iloc[sl_tr]
        X_te, y_te = X.iloc[sl_te], y.iloc[sl_te]
        idx_te = X_te.index

        last_y = y_w.reindex(idx_te).to_numpy(dtype=float)
        yt = y_te.to_numpy(dtype=float)
        m = np.isfinite(last_y) & np.isfinite(yt)
        mae_naive = float(mean_absolute_error(yt[m], last_y[m])) if np.any(m) else float("nan")

        ridge = _ridge()
        ridge.fit(X_tr, y_tr)
        mae_ridge = float(mean_absolute_error(y_te, ridge.predict(X_te)))

        rec: Dict = {
            "col": col,
            "horizon_min": h,
            "n_rows": n,
            "test_mae_ridge": mae_ridge,
            "test_mae_last_value": mae_naive,
        }
        if with_hgb:
            hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
            hgb.fit(X_tr, y_tr)
            rec["test_mae_hgb"] = float(mean_absolute_error(y_te, hgb.predict(X_te)))
        rows.append(rec)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, nargs="*", default=None)
    p.add_argument("--horizons", type=int, nargs="*", default=None)
    p.add_argument("--with-hgb", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.cols is not None:
        cols = tuple(args.cols)
    else:
        cols = TARGET_COLS
    if args.horizons is not None:
        horizons = tuple(args.horizons)
    else:
        horizons = FORECAST_HORIZONS_MINUTES
    out = args.out if args.out is not None else ARTIFACTS / "horizon_sweep.csv"

    all_rows: List[Dict] = []
    for col in cols:
        all_rows.extend(run_column_horizons(int(col), horizons, args.with_hgb))
    df = pd.DataFrame(all_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(df.to_string(index=False))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
