import pandas as pd

from .config import ARTIFACTS, TARGET_COLS
from .train_column import train_all


def main() -> None:
    summary = train_all(TARGET_COLS)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ARTIFACTS / "summary_metrics.csv", index=False)
    win_cols = [
        "col",
        "winner",
        "sv_winner",
        "wf_disagrees_sv",
        "winner_label",
        "blend_a",
        "blend_b",
        "blend_w_on_a",
        "val_mae_ridge",
        "val_mae_hgb",
        "val_mae_rf",
        "val_mae_blend",
        "val_mae_winner",
        "test_mae_winner",
        "test_mae_at_sv_winner",
        "test_mae_at_wf_winner",
    ]
    use_cols = []
    for c in win_cols:
        if c in summary.columns:
            use_cols.append(c)
    wdf = summary[use_cols]
    wdf.to_csv(ARTIFACTS / "winners_by_column.csv", index=False)
    lines = []
    for _, row in wdf.iterrows():
        col = int(row["col"])
        w = row["winner"]
        lab = row["winner_label"]
        line = str(col) + "\t" + str(w) + "\t" + str(lab)
        lines.append(line)
    (ARTIFACTS / "winners_by_column.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print()
    print(wdf[["col", "winner", "winner_label", "blend_a", "blend_b", "blend_w_on_a"]].to_string(index=False))


if __name__ == "__main__":
    main()
