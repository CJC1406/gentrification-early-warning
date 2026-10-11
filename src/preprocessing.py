"""
Shared data loading, preprocessing and evaluation helpers.

Every training script uses these so that all models see exactly the
same cleaned features, the same train/test split and the same metrics.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.20

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

NON_FEATURE_COLUMNS = [
    "tract_id",
    "housing_cost_change",
    "gentrified",
]

# Census API "jam values". They are large NEGATIVE numbers that mean
# "no estimate / not applicable / too few observations". They must be
# turned into missing values before imputation or scaling; otherwise a
# single -666666666 dominates the mean and variance of its column.
# https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-and-annotation-values.html
CENSUS_SENTINELS = [
    -999999999,
    -888888888,
    -666666666,
    -555555555,
    -333333333,
    -222222222,
]


def census_api_key():
    """Read the Census API key from the environment (never hard-code it)."""
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise SystemExit(
            "Set the CENSUS_API_KEY environment variable first, e.g.\n"
            "  export CENSUS_API_KEY=your_key_here"
        )
    return key


def clean_census_values(frame):
    """Coerce to numeric and replace Census sentinel codes with NaN."""
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.mask(frame.isin(CENSUS_SENTINELS))
    # Any remaining value at or below -100,000,000 is also a sentinel.
    frame = frame.mask(frame <= -100_000_000)
    return frame.replace([np.inf, -np.inf], np.nan)


def load_xy(path):
    """
    Load a model dataset and return (X, y, tract_ids).

    Columns that are entirely missing (for example a duplicated text
    column such as NAME) are dropped. This uses no label information,
    so it does not leak anything from the test set.
    """
    df = pd.read_csv(path, dtype={"tract_id": str})
    df["tract_id"] = df["tract_id"].str.zfill(11)

    y = df["gentrified"].astype(int)
    X = clean_census_values(df.drop(columns=NON_FEATURE_COLUMNS))

    empty = X.columns[X.isna().all()]
    if len(empty):
        print(f"Dropping {len(empty)} all-missing columns")
        X = X.drop(columns=empty)

    return X, y, df["tract_id"]


def split(X, y, tract_ids):
    """The single, shared stratified train/test split."""
    return train_test_split(
        X,
        y,
        tract_ids,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )


def county_groups(tract_ids):
    """State + county FIPS (first 5 digits) for grouped cross-validation."""
    return pd.Series(tract_ids).astype(str).str[:5].to_numpy()


def make_preprocessor(scale=True):
    """
    Median imputation (+ optional standardisation).

    Wrapped in a Pipeline so that medians and scaling statistics are
    learned from the training rows only.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def evaluate(name, y_true, y_pred, y_prob=None):
    """Classification metrics that stay meaningful when classes are imbalanced."""
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": np.nan,
    }
    if y_prob is not None:
        row["ROC AUC"] = roc_auc_score(y_true, y_prob)
    return row


def majority_baseline(y_train, y_test):
    """Always predict the most common training label."""
    majority = int(pd.Series(y_train).mode()[0])
    pred = np.full(len(y_test), majority)
    prob = np.full(len(y_test), float(majority))
    row = evaluate("Majority-class baseline", y_test, pred)
    row["ROC AUC"] = 0.5
    return row, pred, prob


def print_results(rows):
    table = pd.DataFrame(rows)
    print(
        table.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        )
    )
    return table
