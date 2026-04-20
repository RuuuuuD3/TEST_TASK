import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .config import ARTIFACTS, TARGET_COLS
from .features import build_feature_table
from .preprocess import clip_series, global_minute_index, load_raw_sorted, minute_series_for_column


def _blend_or_one_pick(meta: dict, pr: np.ndarray, ph: np.ndarray, prf: np.ndarray) -> np.ndarray:
    pack = {"ridge": pr, "hgb": ph, "rf": prf}
    w = float(meta["blend_w_on_a"])
    a, b = meta["blend_a"], meta["blend_b"]
    blend_pred = w * pack[a] + (1.0 - w) * pack[b]
    if meta["winner"] == "blend":
        return blend_pred
    return pack[meta["winner"]]


def load_for_predict(col: int, data_path: Path, artifacts_root: Path):
    col_dir = artifacts_root / f"col_{col}"
    meta_path = col_dir / "strategy_meta.joblib"
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))

    meta = joblib.load(meta_path)
    feat_cols = list(meta["feature_columns"])
    lo, hi = float(meta["winsor_lo"]), float(meta["winsor_hi"])

    ridge = joblib.load(col_dir / "model_ridge.joblib")["model"]
    hgb = joblib.load(col_dir / "model_hgb.joblib")["model"]
    rf = joblib.load(col_dir / "model_rf.joblib")["model"]

    raw = load_raw_sorted(data_path)
    if col not in raw.columns:
        raise ValueError(f"no column {col}")

    all_minutes = global_minute_index(raw)
    y_i = minute_series_for_column(raw, col, all_minutes)
    y_w = clip_series(y_i, lo, hi)

    d_all = build_feature_table(y_w)
    d = d_all.dropna(subset=feat_cols)
    if len(d) == 0:
        raise ValueError("no rows after dropna")
    X = d[feat_cols]
    return meta, X, d, ridge, hgb, rf


def predict_column(col: int, data_path: Path, artifacts_root: Path) -> pd.DataFrame:
    meta, X, d, ridge, hgb, rf = load_for_predict(col, data_path, artifacts_root)

    pr = ridge.predict(X)
    ph = hgb.predict(X)
    prf = rf.predict(X)
    pick = _blend_or_one_pick(meta, pr, ph, prf)

    w = float(meta["blend_w_on_a"])
    a = meta["blend_a"]
    b = meta["blend_b"]
    preds = {"ridge": pr, "hgb": ph, "rf": prf}
    pa = preds[a]
    pb = preds[b]
    blend_only = w * pa + (1.0 - w) * pb

    out = pd.DataFrame(
        {
            "time": d.index,
            "y": d["y"].to_numpy(),
            "pred_ridge": pr,
            "pred_hgb": ph,
            "pred_rf": prf,
            "pred_blend": blend_only,
            "pred_pick": pick,
            "winner": meta["winner"],
            "winner_label": (
                "blend:" + str(a) + "+" + str(b) + " w=" + format(w, ".2f")
                if meta["winner"] == "blend"
                else str(meta["winner"])
            ),
        }
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cols", type=int, nargs="*", default=None)
    args = p.parse_args()

    cols = tuple(args.cols) if args.cols else TARGET_COLS
    out_root = args.out if args.out is not None else args.artifacts / "predict_run"
    out_root.mkdir(parents=True, exist_ok=True)

    for col in cols:
        df = predict_column(int(col), args.data, args.artifacts)
        out_path = out_root / f"pred_col_{col}.csv"
        df.to_csv(out_path, index=False)
        m = df["y"].notna()
        if m.sum() > 0:
            mae = float(mean_absolute_error(df.loc[m, "y"], df.loc[m, "pred_pick"]))
            print(f"col {col}: rows={len(df)} mae(y,pick)={mae:.6f} -> {out_path}")
        else:
            print(f"col {col}: rows={len(df)} (no y in file, mae skipped) -> {out_path}")


if __name__ == "__main__":
    main()
