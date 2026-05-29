import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CATEGORICAL_COLS,
    IMPUTER_MAX_ITER,
    IMPUTER_RANDOM_STATE,
    MODELS_DIR,
    RESULTS_DIR,
    SR_CAT_PARAMS,
    SR_DATA_PATH,
    SR_RF_N_ESTIMATORS,
    SR_XGB_PARAMS,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_TEST_SPLIT_SEED,
)
from hcc.evaluation import build_results_table
from hcc.models import (
    build_models,
    compute_optimal_thresholds,
    pos_weight,
    predict_probabilities,
    train_models,
)
from hcc.preprocessing import impute_and_engineer, split_and_scale

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(SR_DATA_PATH, sheet_name="Sheet1")

df, _ = impute_and_engineer(
    df, CATEGORICAL_COLS, TARGET_COL,
    initial_strategy="median",
    random_state=IMPUTER_RANDOM_STATE,
    max_iter=IMPUTER_MAX_ITER,
)

X_train, X_test, y_train, y_test, scaler, num_cols = split_and_scale(
    df, CATEGORICAL_COLS, TARGET_COL,
    test_size=TEST_SIZE,
    random_state=TRAIN_TEST_SPLIT_SEED,
)

joblib.dump(scaler, MODELS_DIR / "scaler_SR.pkl")

spw = pos_weight(y_train)
models_dict = build_models(spw, SR_XGB_PARAMS, SR_RF_N_ESTIMATORS, cat_params=SR_CAT_PARAMS)
train_models(models_dict, X_train, y_train)

for name, model in models_dict.items():
    joblib.dump(model, MODELS_DIR / f"SR_{name.replace(' ', '_')}.pkl")

pred_probs = predict_probabilities(models_dict, X_test)
thresholds = compute_optimal_thresholds(pred_probs, y_test)

with open(MODELS_DIR / "SR_optimal_thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

results_df = build_results_table(pred_probs, thresholds, y_test)
print(results_df.to_string(index=False))
results_df.to_excel(RESULTS_DIR / "SR_model_performance_with_CI.xlsx", index=False)
