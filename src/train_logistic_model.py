import random

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path("data/raw")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_DIR / "model_dataset.csv"
)

y = df["gentrified"]


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df.drop(
    columns=[
        "tract_id",
        "housing_cost_change",
        "gentrified",
    ]
)

# Convert everything to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace Census missing-value sentinel
X = X.replace(
    999999999,
    np.nan
)

# Median imputation
X = X.fillna(
    X.median()
)

# Final safety check
X = X.fillna(0)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)


print(f"Training samples: {len(X_train):,}")
print(f"Test samples:     {len(X_test):,}")
print(f"Features:         {X.shape[1]:,}")


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    C=0.005,
    max_iter=2000,
    random_state=SEED,
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test_scaled
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print()
print("Logistic Regression results")
print("---------------------------")

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")