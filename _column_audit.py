from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.csv"
SENTINEL = 9999.0


def main() -> None:
    df = pd.read_csv(DATA, header=None, low_memory=False)
    t = pd.to_datetime(df[0], dayfirst=True, errors="coerce")
    df = df.drop(columns=[0])
    df.insert(0, "time", t)
    df = df.dropna(subset=["time"]).sort_values("time")
    df["minute"] = df["time"].dt.floor("min")

    value_cols = [c for c in df.columns if c not in ("time", "minute")]

    full_index = pd.date_range(df["minute"].min(), df["minute"].max(), freq="min")
    n_grid = len(full_index)

    rows = []
    for c in value_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.mask(s == SENTINEL, np.nan)
        sub = pd.DataFrame({"minute": df["minute"], "v": s}).dropna(subset=["v"])
        if sub.empty:
            rows.append((c, 0.0, 0, 0, np.nan, np.nan, np.nan, np.nan, 0.0))
            continue
        g = sub.groupby("minute", as_index=True)["v"].mean()
        aligned = g.reindex(full_index)
        cov = float(aligned.notna().mean())
        n_obs = int(aligned.notna().sum())
        y = aligned.dropna()
        q01, q50, q99 = float(y.quantile(0.01)), float(y.quantile(0.5)), float(y.quantile(0.99))
        iqr = float(y.quantile(0.75) - y.quantile(0.25))
        hi = float(y.quantile(0.75) + 5 * iqr) if iqr > 0 else float(y.max())
        frac_hi = float((y > hi).mean()) if len(y) else 0.0
        rows.append((c, cov, n_obs, n_grid, q01, q50, q99, hi, frac_hi))

    out = pd.DataFrame(
        rows,
        columns=["col", "coverage", "n_obs", "n_grid", "q01", "q50", "q99", "iqr5_hi", "frac_above_fence"],
    ).sort_values("coverage", ascending=False)
    out.to_csv(ROOT / "_column_coverage.csv", index=False)

    for thr in (0.15, 0.25, 0.35, 0.5):
        m = out["coverage"] >= thr
        print(f"coverage>={thr}: n_cols={m.sum()}")

    print("\nTOP 25 по coverage:")
    print(out.head(25).to_string(index=False))
    print("\nBOTTOM 15 по coverage (>0):")
    print(out[out.coverage > 0].tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
