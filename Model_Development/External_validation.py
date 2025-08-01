import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report, 
    precision_recall_curve, roc_curve, confusion_matrix
)


file_path = r"D:\External Validation\통합 문서1.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="Sheet1")

categorical_cols = ["sex", "dm", "hypertensive", "hepatitisb", "hepatitisc", "alcohol",
                    "performance", "encep", "i_vp", "i_vv", "i_b", "i_n",
                    "i_m", "ascites", "cirrhosis"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["3yr_survival_status"]]


scaler = joblib.load("models/scaler_LT.pkl")
df_scaled_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

df_scaled = df.copy()
df_scaled[numerical_cols] = df_scaled_numerical  


import joblib
import json
# Load the saved SVM model and threshold
SVM_model_filename = "LT_new_SVM_model.pkl"
SVM_model = joblib.load(SVM_model_filename)

SVM_threshold_filename = "SVM_optimal_threshold.json"
with open(SVM_threshold_filename, "r") as f:
    optimal_thresholds = json.load(f)

df["survival_probability_LT"] = SVM_model.predict_proba(df_scaled)[:, 1]

best_threshold_SVM = optimal_thresholds["SVM"]

df["risk_group_LT"] = np.where(
    df["survival_probability_LT"] >= best_threshold_SVM, 
    "High Risk", 
    "Low Risk"
)

df.to_excel(file_path, index=False)

# Extract the ground truth labels and predicted probabilities
y_true = df["3yr_survival_status"]  
y_scores = df["survival_probability_surgery"]  

y_pred = (y_scores >= best_threshold_SVM).astype(int)

# Compute evaluation metrics
auroc = roc_auc_score(y_true, y_scores)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)

precision = report["1"]["precision"]  
recall = report["1"]["recall"] 

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

specificity = tn / (tn + fp)  

# Display results
metrics_df = pd.DataFrame({
    "Metric": ["AUROC", "Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score"],
    "Value": [auroc, accuracy, precision, recall, specificity, f1]
})

print(metrics_df)

# Compute ROC curve values
fpr, tpr, _ = roc_curve(y_true, y_scores)

metrics_df

###################################### Surgery #####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report, 
    precision_recall_curve, roc_curve, confusion_matrix
)


file_path = r"D:\External Validation\통합 문서1.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="Sheet1")

categorical_cols = ["sex", "dm", "hypertensive", "hepatitisb", "hepatitisc", "alcohol",
                    "performance", "encep", "i_vp", "i_vv", "i_b", "i_n",
                    "i_m", "ascites", "cirrhosis"]
numerical_cols = [col for col in df.columns if col not in categorical_cols]

scaler = joblib.load("models/scaler_SR.pkl")
df_scaled_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

df_scaled = df.copy()
df_scaled[numerical_cols] = df_scaled_numerical  


import joblib
import json

# Load the saved CatBoost model and threshold
catboost_model_filename = "Surgery_catboost_model.pkl"
catboost_model = joblib.load(catboost_model_filename)

catboost_threshold_filename = "catboost_optimal_threshold.json"
with open(catboost_threshold_filename, "r") as f:
    optimal_thresholds = json.load(f)

df["survival_probability_surgery"] = catboost_model.predict_proba(df_scaled)[:, 1]

best_threshold_cat = optimal_thresholds["CatBoost"]

df["risk_group_Surgery"] = np.where(
    df["survival_probability_surgery"] >= best_threshold_cat, 
    "High Risk", 
    "Low Risk"
)

df.to_excel(file_path, index=False)

y_true = df["3yr_survival_status"]  # Actual labels
y_scores = df["survival_probability_surgery"]  # Predicted probabilities


y_pred = (y_scores >= best_threshold_cat).astype(int)

# Compute evaluation metrics
auroc = roc_auc_score(y_true, y_scores)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)

precision = report["1"]["precision"]  
recall = report["1"]["recall"] 

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

specificity = tn / (tn + fp)  

# Display results
metrics_df = pd.DataFrame({
    "Metric": ["AUROC", "Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score"],
    "Value": [auroc, accuracy, precision, recall, specificity, f1]
})

print(metrics_df)

# Compute ROC curve values
fpr, tpr, _ = roc_curve(y_true, y_scores)

metrics_df
