from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES_ROOT = ARTIFACTS / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"

TIMESTAMP_COL = 0
SENTINEL_BAD = 9999.0

COLS_MAIN = (15, 16, 17, 18, 19, 20, 75)
COLS_EXTENDED = (76, 77, 71)
TARGET_COLS = tuple(COLS_MAIN) + tuple(COLS_EXTENDED)

INTERP_LIMIT_MINUTES = 10
CLIP_LOW_Q = 0.001
CLIP_HIGH_Q = 0.999
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

LAG_STEPS = (1, 2, 3, 6, 12, 24, 48, 96, 1440)
ROLL_WINDOWS = (12, 48, 1440)

RANDOM_STATE = 42

HGB_PARAMS = dict(
    max_depth=8,
    learning_rate=0.06,
    max_iter=350,
    min_samples_leaf=40,
    l2_regularization=1e-3,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)

RIDGE_ALPHA = 1.0

RF_PARAMS = dict(
    n_estimators=120,
    max_depth=16,
    min_samples_leaf=4,
    max_samples=0.55,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

BLEND_PAIRS = (("ridge", "hgb"), ("ridge", "rf"), ("hgb", "rf"))

WF_VAL_FRAC = 0.10
WF_MIN_TRAIN_FRAC = 0.40
WF_MAX_FOLDS = 7
WF_MIN_FOLDS = 3

FORECAST_HORIZONS_MINUTES = (1, 5, 15, 60, 240, 1440)
EXTRA_HORIZON_MINUTES = (2, 3, 10, 20, 30, 45, 90, 120, 180, 720)
FULL_HORIZON_SWEEP_MINUTES = tuple(sorted(set(FORECAST_HORIZONS_MINUTES + EXTRA_HORIZON_MINUTES)))
