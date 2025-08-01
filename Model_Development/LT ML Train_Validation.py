import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report,
    precision_score, recall_score, confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib

# Load data
file_path = "C:\\Users\\andre\\Desktop\\HCC연구\\Data_간이식.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Define categorical and numerical columns
categorical_cols = ["sex", "dm", "hypertensive", "hepatitisb", "hepatitisc", "alcohol",
                    "performance", "encep", "i_vp", "i_vv", "i_b", "i_n",
                    "i_m", "ascites", "cirrhosis"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["3yr_survival_status"]]

# Impute numerical columns
imputer_iter = IterativeImputer(max_iter=100, initial_strategy="mean", random_state=42)
df[numerical_cols] = imputer_iter.fit_transform(df[numerical_cols])

# Impute categorical columns
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
for col in existing_categorical_cols:
    probs = df[col].value_counts(normalize=True)
    df.loc[df[col].isna(), col] = np.random.choice(probs.index, p=probs.values, size=df[col].isna().sum())

# Create BMI and drop height/weight
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
df.drop(columns=["height", "weight"], inplace=True)

# Redefine column lists
categorical_cols = [col for col in categorical_cols if col in df.columns]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["3yr_survival_status"]]

# Define features and target
X = df.drop(columns=["3yr_survival_status"])
y = df["3yr_survival_status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = RobustScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
X[numerical_cols] = scaler.transform(X[numerical_cols])
joblib.dump(scaler, "scaler_LT.pkl")

# Train models
results = {}
pred_probs = {}
models = {}

# Logistic Regression
model = LogisticRegression(class_weight="balanced", solver="liblinear")
model.fit(X_train, y_train)
models["Logistic Regression"] = model
pred_probs["Logistic Regression"] = model.predict_proba(X_test)[:, 1]

# Random Forest
model = RandomForestClassifier(n_estimators=2000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
models["Random Forest"] = model
pred_probs["Random Forest"] = model.predict_proba(X_test)[:, 1]

# XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight, n_estimators=70, max_depth=8,
    learning_rate=0.0218, subsample=0.6468, colsample_bytree=0.5275,
    min_child_weight=1, gamma=3.5, reg_alpha=0.049, reg_lambda=0.023,
    random_state=42, eval_metric="auc", use_label_encoder=False, n_jobs=-1
)
model.fit(X_train, y_train)
models["XGBoost"] = model
pred_probs["XGBoost"] = model.predict_proba(X_test)[:, 1]

# SVM
model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
models["SVM"] = model
pred_probs["SVM"] = model.predict_proba(X_test)[:, 1]

# LightGBM
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=scale_pos_weight, random_state=42)
model.fit(X_train, y_train)
models["LightGBM"] = model
pred_probs["LightGBM"] = model.predict_proba(X_test)[:, 1]

# CatBoost
model = cb.CatBoostClassifier(random_state=42, verbose=0, auto_class_weights="Balanced")
model.fit(X_train, y_train)
models["CatBoost"] = model
pred_probs["CatBoost"] = model.predict_proba(X_test)[:, 1]

# Calculate optimal thresholds
optimal_thresholds = {}
for model_name, y_prob in pred_probs.items():
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    optimal_thresholds[model_name] = thresholds[best_idx]

# Bootstrap function
def bootstrap_ci(y_true, y_pred_prob, metric_fn, threshold=None, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        y_true_sample = y_true.iloc[idx]
        y_prob_sample = y_pred_prob[idx]
        if threshold is not None:
            y_pred_sample = (y_prob_sample >= threshold).astype(int)
            score = metric_fn(y_true_sample, y_pred_sample)
        else:
            score = metric_fn(y_true_sample, y_prob_sample)
        scores.append(score)
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

# Create result table
results_df = pd.DataFrame(index=models.keys())

for model_name, y_prob in pred_probs.items():
    threshold = optimal_thresholds[model_name]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    results_df.loc[model_name, "Accuracy"] = accuracy_score(y_test, y_pred)
    results_df.loc[model_name, "F1 Score"] = f1_score(y_test, y_pred)
    results_df.loc[model_name, "AUROC"] = roc_auc_score(y_test, y_prob)
    results_df.loc[model_name, "Precision"] = precision_score(y_test, y_pred)
    results_df.loc[model_name, "Recall"] = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results_df.loc[model_name, "Specificity"] = tn / (tn + fp)

    # Confidence Intervals
    acc_ci = bootstrap_ci(y_test, y_prob, accuracy_score, threshold)
    f1_ci = bootstrap_ci(y_test, y_prob, f1_score, threshold)
    auc_ci = bootstrap_ci(y_test, y_prob, roc_auc_score)
    precision_ci = bootstrap_ci(y_test, y_prob, precision_score, threshold)
    recall_ci = bootstrap_ci(y_test, y_prob, recall_score, threshold)

    def specificity_fn(y_true, y_pred):
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    specificity_ci = bootstrap_ci(y_test, y_prob, specificity_fn, threshold)

    results_df.loc[model_name, "Accuracy 95% CI"] = f"[{acc_ci[0]:.2f}, {acc_ci[1]:.2f}]"
    results_df.loc[model_name, "F1 Score 95% CI"] = f"[{f1_ci[0]:.2f}, {f1_ci[1]:.2f}]"
    results_df.loc[model_name, "AUROC 95% CI"] = f"[{auc_ci[0]:.2f}, {auc_ci[1]:.2f}]"
    results_df.loc[model_name, "Precision 95% CI"] = f"[{precision_ci[0]:.2f}, {precision_ci[1]:.2f}]"
    results_df.loc[model_name, "Recall 95% CI"] = f"[{recall_ci[0]:.2f}, {recall_ci[1]:.2f}]"
    results_df.loc[model_name, "Specificity 95% CI"] = f"[{specificity_ci[0]:.2f}, {specificity_ci[1]:.2f}]"

print(results_df)

results_df.to_excel('LT_bootstrap.xlsx')
