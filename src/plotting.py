from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _fewer_rows(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def _pick_label(pick: str, blend_a: str, blend_b: str, blend_w: float) -> str:
    if pick == "blend":
        return f"blend:{blend_a}+{blend_b} w={blend_w:.2f}"
    return pick


def plot_column_splits(
    figures_dir: Path,
    df: pd.DataFrame,
    col: int,
    blend_a: str,
    blend_b: str,
    blend_w: float,
    pick: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    suf = f"_col_{col}.png"
    plab = _pick_label(pick, blend_a, blend_b, blend_w)
    df = df.sort_values("time").reset_index(drop=True)

    tr = df["split"] == "train"
    va = df["split"] == "val"
    te = df["split"] == "test"

    t_tr_end = df.loc[tr, "time"].max()
    t_va_end = df.loc[va, "time"].max()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axvspan(df["time"].min(), t_tr_end, color="#2ca02c", alpha=0.08)
    ax.axvspan(t_tr_end, t_va_end, color="#1f77b4", alpha=0.08)
    ax.axvspan(t_va_end, df["time"].max(), color="#ff7f0e", alpha=0.08)
    ax.axvline(t_tr_end, color="gray", ls="--", lw=0.9)
    ax.axvline(t_va_end, color="gray", ls="--", lw=0.9)

    d_plot = _fewer_rows(df, 25_000)
    tt = pd.to_datetime(d_plot["time"])
    ax.plot(tt, d_plot["y_true"], color="black", lw=0.75, alpha=0.9, label="y true")
    ax.plot(tt, d_plot["pred_winner"], color="#d62728", lw=0.65, alpha=0.88, label=plab)
    ax.set_title(f"c{col} | y vs {plab} | train/val/test")
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(figures_dir / f"timeline_winner{suf}", dpi=140)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    sub = df.loc[va].copy()
    if not sub.empty:
        sub = _fewer_rows(sub, 15_000)
        tp = pd.to_datetime(sub["time"])
        ax2.plot(tp, sub["y_true"], color="black", lw=1.0, label="y true")
        ax2.plot(tp, sub["pred_ridge"], color="#1f77b4", lw=0.55, alpha=0.88, label="ridge")
        ax2.plot(tp, sub["pred_hgb"], color="#2ca02c", lw=0.55, alpha=0.88, label="hgb")
        ax2.plot(tp, sub["pred_rf"], color="#bcbd22", lw=0.55, alpha=0.9, label="rf")
        ax2.plot(tp, sub["pred_blend"], color="#9467bd", lw=0.65, alpha=0.9, label="blend")
        bt = f"{blend_a}+{blend_b} w={blend_w:.2f}"
        ax2.set_title(f"c{col} | val | best2 {bt} | pick={plab}")
        ax2.set_xlabel("time")
        ax2.set_ylabel("y")
        ax2.legend(loc="best", fontsize=7, ncol=2)
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    else:
        ax2.set_title(f"c{col} | val empty")
    fig2.tight_layout()
    fig2.savefig(figures_dir / f"val_compare{suf}", dpi=140)
    plt.close(fig2)

    tail_n = min(len(df.loc[tr]), 8000)
    df_tr_tail = df.loc[tr].iloc[-tail_n:]
    df_combo = pd.concat([df_tr_tail, df.loc[va], df.loc[te]], axis=0).sort_values("time")
    df_c = _fewer_rows(df_combo, 30_000)
    tc = pd.to_datetime(df_c["time"])
    fig3, ax3 = plt.subplots(figsize=(16, 5))
    ax3.axvspan(df_tr_tail["time"].min(), t_tr_end, color="#2ca02c", alpha=0.07)
    ax3.axvspan(t_tr_end, t_va_end, color="#1f77b4", alpha=0.07)
    ax3.axvspan(t_va_end, df_c["time"].max(), color="#ff7f0e", alpha=0.07)
    ax3.axvline(t_tr_end, color="gray", ls="--", lw=0.9)
    ax3.axvline(t_va_end, color="gray", ls="--", lw=0.9)
    ax3.plot(tc, df_c["y_true"], color="black", lw=0.7, alpha=0.92, label="y true")
    ax3.plot(tc, df_c["pred_winner"], color="#d62728", lw=0.55, alpha=0.88, label=plab)
    ax3.set_title(f"c{col} | tail+val+test | y vs {plab} | n_tail={tail_n}")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_xlabel("time")
    ax3.set_ylabel("y")
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))
    fig3.tight_layout()
    fig3.savefig(figures_dir / f"tail_winner{suf}", dpi=140)
    plt.close(fig3)
