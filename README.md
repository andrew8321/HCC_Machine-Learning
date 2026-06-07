# HCC Machine Learning

Code for the primary analysis in:

> **Machine Learning Based Selection of Resection versus Transplant Improves Survival in Hepatocellular Carcinoma**  
> *JAMA Network Open*, 2025. https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2839032

---

## Repository Structure

```
HCC_Machine-Learning/
├── config.py                   # Data paths and model hyperparameters
├── requirements.txt
│
├── hcc/                        # Shared Python package
│   ├── preprocessing.py        # Imputation, BMI engineering, train/test split
│   ├── models.py               # 6-model ensemble factory and threshold selection
│   └── evaluation.py           # Bootstrap CI and results table
│
├── scripts/
│   ├── train_lt.py             # Train Liver Transplant models
│   ├── train_surgery.py        # Train Surgical Resection models
│   └── validate_external.py    # External cohort validation
│
├── tests/                      # Unit tests (no patient data required)
│
└── statistical_analysis/       # R scripts for survival and counterfactual analysis
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data

Patient data files are **not** included in this repository. Set environment variables to point scripts at your local data:

```bash
export HCC_LT_DATA=/path/to/Data_LT.xlsx
export HCC_SR_DATA=/path/to/Data_Surgery.xlsx
export HCC_EXT_DATA=/path/to/Data_External.xlsx

# Optional: override output directories (default: models/ and results/)
export HCC_MODELS_DIR=/path/to/models
export HCC_RESULTS_DIR=/path/to/results
```

Alternatively, place your data files in a `data/` directory at the repo root using the default names (`Data_LT.xlsx`, `Data_Surgery.xlsx`, `Data_External.xlsx`).

---

## Running the Models

```bash
# 1. Train Liver Transplant model (saves scaler + models to models/)
python scripts/train_lt.py

# 2. Train Surgical Resection model
python scripts/train_surgery.py

# 3. External validation (requires trained models from steps 1 and 2)
python scripts/validate_external.py
```

---

## Tests

```bash
python -m pytest tests/
```

Tests use synthetic data and require no patient files.

---

## Statistical Analysis

R scripts for survival analysis and counterfactual analysis are in `statistical_analysis/`. Each script is self-contained and documents its required input data at the top.
