from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .config import CLIP_HIGH_Q, CLIP_LOW_Q, DATA_PATH, INTERP_LIMIT_MINUTES, SENTINEL_BAD, TIMESTAMP_COL, TRAIN_FRAC


def load_raw_sorted(csv_path: Optional[Path] = None) -> pd.DataFrame:
    path = csv_path if csv_path is not None else DATA_PATH
    df = pd.read_csv(path, header=None, low_memory=False)
    t = pd.to_datetime(df[TIMESTAMP_COL], dayfirst=True, errors="coerce")
    df = df.drop(columns=[TIMESTAMP_COL])
    df.insert(0, "_time", t)
    df = df.dropna(subset=["_time"]).sort_values("_time")
    return df


def global_minute_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    t = df["_time"]
    start = t.min().floor("min")
    end = t.max().floor("min")
    return pd.date_range(start, end, freq="min")


def minute_series_for_column(df: pd.DataFrame, col: Union[int, str], all_minutes: pd.DatetimeIndex) -> pd.Series:
    minute = df["_time"].dt.floor("min")
    v = pd.to_numeric(df[col], errors="coerce").to_numpy()
    v = np.where(v == SENTINEL_BAD, np.nan, v)
    sub = pd.DataFrame({"m": minute, "v": v}).dropna(subset=["m"])
    sub = sub.dropna(subset=["v"])
    y = sub.groupby("m", sort=True)["v"].mean().reindex(all_minutes)
    y = y.interpolate(method="linear", limit=INTERP_LIMIT_MINUTES, limit_direction="both")
    y.name = f"y_{col}"
    return y


def clip_bounds_from_train(y: pd.Series, all_minutes: pd.DatetimeIndex) -> tuple[float, float]:
    if len(all_minutes) == 0:
        return float(np.nan), float(np.nan)
    cut_idx = int(max(0, min(len(all_minutes) - 1, np.floor(TRAIN_FRAC * len(all_minutes)))))
    t_cut = all_minutes[cut_idx]
    y_tr = y.loc[:t_cut].dropna()
    if y_tr.empty:
        return float(y.min()), float(y.max())
    lo = float(y_tr.quantile(CLIP_LOW_Q))
    hi = float(y_tr.quantile(CLIP_HIGH_Q))
    if lo >= hi:
        lo, hi = float(y_tr.min()), float(y_tr.max())
    return lo, hi


def clip_series(y: pd.Series, lo: float, hi: float) -> pd.Series:
    return y.clip(lower=lo, upper=hi)


def save_minute_csv(path, y_before: pd.Series, y_after: pd.Series) -> None:
    out = pd.DataFrame(
        {
            "time": y_before.index,
            "y_after_interp": y_before.values,
            "y_after_clip": y_after.values,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
