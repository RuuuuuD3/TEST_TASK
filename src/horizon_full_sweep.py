import argparse
import json

import pandas as pd

from .config import ARTIFACTS, FULL_HORIZON_SWEEP_MINUTES, TARGET_COLS
from .preprocess import load_raw_sorted
from .train_column import train_one_column


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", type=str, required=True)
    p.add_argument("--grid", choices=("normal", "targets", "data"), default=None)
    p.add_argument("--horizons", type=int, nargs="*", default=None)
    p.add_argument("--cols", type=int, nargs="*", default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.grid is not None:
        if args.cols is not None or args.horizons is not None:
            p.error("no --cols/--horizons with --grid")
        horizons = FULL_HORIZON_SWEEP_MINUTES
        if args.grid in ("normal", "targets"):
            cols = TARGET_COLS
        else:
            raw = load_raw_sorted()
            cols = tuple(sorted((c for c in raw.columns if c != "_time"), key=int))
    else:
        if not args.horizons:
            p.error("need --horizons or --grid")
        horizons = tuple(args.horizons)
        cols = tuple(args.cols) if args.cols else TARGET_COLS

    root = ARTIFACTS / "full_horizon" / args.run_tag
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in horizons:
        h_root = root / f"h_{h}"
        h_root.mkdir(parents=True, exist_ok=True)
        for col in cols:
            print(f"run_tag={args.run_tag} horizon_min={h} col={col}", flush=True)
            done = h_root / f"col_{col}" / "metrics.json"
            if not args.no_resume and done.is_file():
                with open(done, encoding="utf-8") as f:
                    rows.append(json.load(f))
                print(f"reuse col={col} h={h}", flush=True)
                continue
            try:
                m = train_one_column(int(col), horizon_minutes=int(h), artifacts_base=h_root)
                rows.append(m)
            except Exception as exc:
                print(f"SKIP col={col} h={h}: {exc}", flush=True)
                rows.append(
                    {
                        "col": int(col),
                        "horizon_min": int(h),
                        "n_rows": None,
                        "error": str(exc),
                    }
                )
    summary = pd.DataFrame(rows)
    out_csv = root / "summary_metrics.csv"
    summary.to_csv(out_csv, index=False)
    with pd.option_context("display.max_rows", 200, "display.width", 140):
        print(summary.to_string(index=False))
    print(f"\nwritten {out_csv}")


if __name__ == "__main__":
    main()
