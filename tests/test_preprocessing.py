import numpy as np
import pandas as pd
import pytest

from hcc.preprocessing import impute_and_engineer, split_and_scale


def _make_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "height": rng.uniform(155, 185, n),
        "weight": rng.uniform(50, 100, n),
        "albumin": rng.uniform(2.5, 4.5, n),
        "sex": rng.choice([0, 1], n).astype(float),
        "dm": rng.choice([0, 1], n).astype(float),
        "3yr_survival_status": rng.choice([0, 1], n),
    })
    return df


def test_bmi_calculation():
    df = _make_df()
    expected_bmi = df["weight"] / (df["height"] / 100) ** 2

    result, _ = impute_and_engineer(df, ["sex", "dm"], "3yr_survival_status", initial_strategy="mean")

    assert "bmi" in result.columns
    assert "height" not in result.columns
    assert "weight" not in result.columns
    np.testing.assert_allclose(result["bmi"].values, expected_bmi.values, rtol=1e-6)


def test_categorical_imputation_no_nan():
    df = _make_df(n=100)
    df.loc[df.sample(10, random_state=1).index, "sex"] = np.nan
    df.loc[df.sample(5, random_state=2).index, "dm"] = np.nan

    result, _ = impute_and_engineer(df, ["sex", "dm"], "3yr_survival_status", initial_strategy="mean")

    assert result["sex"].isna().sum() == 0
    assert result["dm"].isna().sum() == 0


def test_split_and_scale_shapes():
    df = _make_df(n=100)
    df, _ = impute_and_engineer(df, ["sex", "dm"], "3yr_survival_status", initial_strategy="mean")
    X_train, X_test, y_train, y_test, scaler, num_cols = split_and_scale(
        df, ["sex", "dm"], "3yr_survival_status", test_size=0.2
    )

    assert len(X_train) + len(X_test) == 100
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_scale_full_dataset_returns_seven_values():
    df = _make_df(n=60)
    df, _ = impute_and_engineer(df, ["sex", "dm"], "3yr_survival_status", initial_strategy="mean")
    result = split_and_scale(
        df, ["sex", "dm"], "3yr_survival_status", scale_full_dataset=True
    )
    assert len(result) == 7
