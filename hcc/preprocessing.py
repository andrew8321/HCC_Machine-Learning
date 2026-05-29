import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def impute_and_engineer(
    df,
    categorical_cols,
    target_col,
    initial_strategy="mean",
    random_state=42,
    max_iter=100,
):
    """
    Impute missing values and compute BMI from height/weight.

    Numerical columns: IterativeImputer with the given initial_strategy.
    Categorical columns: probabilistic sampling from observed distribution.
    Returns (df_processed, fitted_imputer).

    Note: initial_strategy differs between cohorts ("mean" for LT, "median" for
    Surgery) — pass explicitly rather than relying on a default.
    """
    df = df.copy()

    existing_cat = [c for c in categorical_cols if c in df.columns]
    num_cols = [c for c in df.columns if c not in existing_cat + [target_col]]

    imputer = IterativeImputer(
        max_iter=max_iter, initial_strategy=initial_strategy, random_state=random_state
    )
    df[num_cols] = imputer.fit_transform(df[num_cols])

    rng = np.random.default_rng(random_state)
    for col in existing_cat:
        mask = df[col].isna()
        if mask.any():
            probs = df[col].value_counts(normalize=True)
            df.loc[mask, col] = rng.choice(probs.index, size=mask.sum(), p=probs.values)

    if "height" in df.columns and "weight" in df.columns:
        df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
        df.drop(columns=["height", "weight"], inplace=True)

    return df, imputer


def split_and_scale(
    df,
    categorical_cols,
    target_col,
    test_size=0.2,
    random_state=42,
    scale_full_dataset=False,
):
    """
    Stratified train/test split and RobustScaler on numerical columns.

    scale_full_dataset: if True, also returns X with scaler applied to the
    full feature set (needed by the LT model for external validation).

    Returns (X_train, X_test, y_train, y_test, scaler, numerical_cols).
    When scale_full_dataset=True, also returns X_full as the last element.
    """
    existing_cat = [c for c in categorical_cols if c in df.columns]
    num_cols = [c for c in df.columns if c not in existing_cat + [target_col]]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    scaler = RobustScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    if scale_full_dataset:
        X_full = X.copy()
        X_full[num_cols] = scaler.transform(X_full[num_cols])
        return X_train, X_test, y_train, y_test, scaler, num_cols, X_full

    return X_train, X_test, y_train, y_test, scaler, num_cols
