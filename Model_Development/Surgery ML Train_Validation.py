import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, precision_recall_curve
)

# ======================= 1. Load & preprocess ===========================
df = pd.read_excel(r"C:\Users\andre\Desktop\HCC연구\Data_수술.xlsx", sheet_name="Sheet1")

# Define categorical and numerical columns
categorical_cols = ["sex", "dm", "hypertensive", "hepatitisb", "hepatitisc", "alcohol",
                    "performance", "encep", "i_vp", "i_vv", "i_b", "i_n", "i_m", "ascites", "cirrhosis"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["3yr_survival_status"]]

# Impute numerical columns
imputer = IterativeImputer(random_state=42, initial_strategy="median")
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Impute categorical columns by probabilistic sampling
for col in categorical_cols:
    probs = df[col].value_counts(normalize=True)
    df.loc[df[col].isna(), col] = np.random.choice(probs.index, size=df[col].isna().sum(), p=probs.values)

# Create BMI
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
df.drop(columns=["height", "weight"], inplace=True)

# Update numerical_cols
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["3yr_survival_status"]]

# ======================= 2. Train-test split & scaling ==================
X = df.drop(columns=["3yr_survival_status"])
y = df["3yr_survival_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ======================= 3. Fit models ================================
model_results = {}
pred_probs = {}

# Logistic Regression
log_model = LogisticRegression(class_weight="balanced", solver="liblinear")
log_model.fit(X_train, y_train)
pred_probs["Logistic Regression"] = log_model.predict_proba(X_test)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
pred_probs["Random Forest"] = rf_model.predict_proba(X_test)[:, 1]

# XGBoost
xgb_model = xgb.XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                              learning_rate=0.03, max_depth=6, n_estimators=200,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=42)
xgb_model.fit(X_train, y_train)
pred_probs["XGBoost"] = xgb_model.predict_proba(X_test)[:, 1]

# SVM
svm_model = SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=42)
svm_model.fit(X_train, y_train)
pred_probs["SVM"] = svm_model.predict_proba(X_test)[:, 1]

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                               random_state=42)
lgb_model.fit(X_train, y_train)
pred_probs["LightGBM"] = lgb_model.predict_proba(X_test)[:, 1]

# CatBoost
cat_model = cb.CatBoostClassifier(iterations=200, learning_rate=0.03, depth=6,
                                  auto_class_weights="Balanced", random_state=42, verbose=0)
cat_model.fit(X_train, y_train)
pred_probs["CatBoost"] = cat_model.predict_proba(X_test)[:, 1]

# ======================= 4. Threshold optimization ============================
optimal_thresholds = {}
for model, probs in pred_probs.items():
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1)
    optimal_thresholds[model] = thresholds[best_idx]

# ======================= 5. Bootstrap CI function =============================
def bootstrap_ci(y_true, y_prob, metric_fn, threshold=None, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        y_t = y_true.iloc[idx]
        p_t = y_prob[idx]
        if threshold is not None:
            p_t = (p_t >= threshold).astype(int)
        score = metric_fn(y_t, p_t)
        scores.append(score)
    return round(np.percentile(scores, 2.5), 2), round(np.percentile(scores, 97.5), 2)

def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# ======================= 6. Compute metrics and CI ============================
summary = []

for model_name, probs in pred_probs.items():
    thresh = optimal_thresholds[model_name]
    y_pred = (probs >= thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    spec = specificity(y_test, y_pred)

    acc_ci = bootstrap_ci(y_test, probs, accuracy_score, threshold=thresh)
    f1_ci = bootstrap_ci(y_test, probs, f1_score, threshold=thresh)
    auc_ci = bootstrap_ci(y_test, probs, roc_auc_score)
    prec_ci = bootstrap_ci(y_test, probs, precision_score, threshold=thresh)
    rec_ci = bootstrap_ci(y_test, probs, recall_score, threshold=thresh)
    spec_ci = bootstrap_ci(y_test, probs, specificity, threshold=thresh)

    summary.append({
        "Model": model_name,
        "Accuracy": round(acc, 2),
        "F1 Score": round(f1, 2),
        "AUROC": round(auc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "Specificity": round(spec, 2),
        "Accuracy 95% CI": f"[{acc_ci[0]}, {acc_ci[1]}]",
        "F1 Score 95% CI": f"[{f1_ci[0]}, {f1_ci[1]}]",
        "AUROC 95% CI": f"[{auc_ci[0]}, {auc_ci[1]}]",
        "Precision 95% CI": f"[{prec_ci[0]}, {prec_ci[1]}]",
        "Recall 95% CI": f"[{rec_ci[0]}, {rec_ci[1]}]",
        "Specificity 95% CI": f"[{spec_ci[0]}, {spec_ci[1]}]",
    })

# ======================= 7. Export ===============================
results_df = pd.DataFrame(summary)
results_df.to_excel("SR_model_performance_with_CI.xlsx", index=False)
print(results_df)
