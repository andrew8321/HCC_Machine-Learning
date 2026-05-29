import os
from pathlib import Path

# ── Data paths ────────────────────────────────────────────────────────────────
# Set HCC_DATA_DIR (or the per-file env vars below) to point at your data.
DATA_DIR = Path(os.environ.get("HCC_DATA_DIR", "data"))

LT_DATA_PATH  = Path(os.environ.get("HCC_LT_DATA",  DATA_DIR / "Data_LT.xlsx"))
SR_DATA_PATH  = Path(os.environ.get("HCC_SR_DATA",  DATA_DIR / "Data_Surgery.xlsx"))
EXT_DATA_PATH = Path(os.environ.get("HCC_EXT_DATA", DATA_DIR / "Data_External.xlsx"))

# ── Output paths ──────────────────────────────────────────────────────────────
MODELS_DIR  = Path(os.environ.get("HCC_MODELS_DIR",  "models"))
RESULTS_DIR = Path(os.environ.get("HCC_RESULTS_DIR", "results"))

# ── Feature schema ────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "sex", "dm", "hypertensive", "hepatitisb", "hepatitisc", "alcohol",
    "performance", "encep", "i_vp", "i_vv", "i_b", "i_n",
    "i_m", "ascites", "cirrhosis",
]
TARGET_COL = "3yr_survival_status"

# ── Shared preprocessing constants ────────────────────────────────────────────
IMPUTER_MAX_ITER     = 100
IMPUTER_RANDOM_STATE = 42
TRAIN_TEST_SPLIT_SEED = 42
TEST_SIZE            = 0.2
BOOTSTRAP_N          = 1000
BOOTSTRAP_SEED       = 42

# ── LT cohort hyperparameters (published — do not change) ─────────────────────
LT_RF_N_ESTIMATORS = 2000
LT_XGB_PARAMS = dict(
    n_estimators=70, max_depth=8, learning_rate=0.0218,
    subsample=0.6468, colsample_bytree=0.5275,
    min_child_weight=1, gamma=3.5, reg_alpha=0.049, reg_lambda=0.023,
    random_state=42, eval_metric="auc", use_label_encoder=False, n_jobs=-1,
)

# ── Surgery cohort hyperparameters (published — do not change) ────────────────
SR_RF_N_ESTIMATORS = 100
SR_XGB_PARAMS = dict(
    learning_rate=0.03, max_depth=6, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=42,
)
SR_CAT_PARAMS = dict(
    iterations=200, learning_rate=0.03, depth=6,
    auto_class_weights="Balanced", random_state=42, verbose=0,
)
