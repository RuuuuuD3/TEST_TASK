from typing import Tuple

import numpy as np
import pandas as pd

from .config import LAG_STEPS, ROLL_WINDOWS


def _sin_cos(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    ang = 2.0 * np.pi * values.astype(float) / period
    return np.sin(ang), np.cos(ang)


def build_feature_table(y: pd.Series, horizon_minutes: int = 0) -> pd.DataFrame:
    d = pd.DataFrame(index=y.index)
    yy_obs = y.astype(float)
    h = int(horizon_minutes)
    if h <= 0:
        d["y"] = yy_obs
    else:
        d["y"] = yy_obs.shift(-h)

    idx = pd.DatetimeIndex(d.index)
    hour = idx.hour.astype(float) + idx.minute.astype(float) / 60.0
    dow = idx.dayofweek.astype(float)
    doy = idx.dayofyear.astype(float)

    sh, ch = _sin_cos(hour.to_numpy(), 24.0)
    sd, cd = _sin_cos(dow.to_numpy(), 7.0)
    sy, cy = _sin_cos(doy.to_numpy(), 365.25)

    d["sin_hour"] = sh
    d["cos_hour"] = ch
    d["sin_dow"] = sd
    d["cos_dow"] = cd
    d["sin_doy"] = sy
    d["cos_doy"] = cy

    for lag in LAG_STEPS:
        d[f"lag_{lag}"] = yy_obs.shift(lag)
    for w in ROLL_WINDOWS:
        d[f"roll_mean_{w}"] = yy_obs.shift(1).rolling(window=w, min_periods=1).mean()

    return d


def feature_target_split(d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = []
    for c in d.columns:
        if c != "y":
            feature_cols.append(c)
    dd = d.dropna()
    x_part = dd[feature_cols]
    y_part = dd["y"]
    return x_part, y_part
